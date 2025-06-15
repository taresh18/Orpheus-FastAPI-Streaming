import re
import os
import requests
import json
import time
import asyncio
import threading
import queue
from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

# vLLM imports (conditional)
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from .consts import *
from .utils import format_prompt, split_text_into_sentences, stitch_wav_files, perf_monitor
from .decoder import tokens_decoder, tokens_decoder_sync
from src.logger import get_logger

logger = get_logger()

class BaseTokenGenerator(ABC):
    """Abstract base class for token generators."""
    
    @abstractmethod
    def generate_tokens(self, prompt: str, voice: str = DEFAULT_VOICE, **kwargs) -> Generator[str, None, None]:
        """Generate tokens from text prompt."""
        pass
    
    @abstractmethod
    async def generate_tokens_async(self, prompt: str, voice: str = DEFAULT_VOICE, **kwargs) -> Generator[str, None, None]:
        """Generate tokens asynchronously from text prompt."""
        pass

class APITokenGenerator(BaseTokenGenerator):
    """
    Token generator using external API.
    Handles HTTP requests, streaming, retry logic, and error handling.
    """
    
    def __init__(self):
        """Initialize the API token generator."""
        if not INFER_SERVER_URL:
            raise ValueError("INFER_SERVER_URL not configured for API mode")
        
        self.session = requests.Session()
        logger.info(f"APITokenGenerator initialized with server: {INFER_SERVER_URL}")
    
    def generate_tokens(self, prompt: str, voice: str = DEFAULT_VOICE, 
                       temperature: float = TEMPERATURE, top_p: float = TOP_P, 
                       max_tokens: int = MAX_TOKENS, 
                       repetition_penalty: float = REPETITION_PENALTY) -> Generator[str, None, None]:
        """
        Generate tokens from external API with optimized streaming and retry logic.
        
        Args:
            prompt: Text prompt to generate from
            voice: Voice to use for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            
        Yields:
            Token strings
        """
        start_time = time.time()
        formatted_prompt = format_prompt(prompt, voice)
        logger.info(f"Generating tokens via API for: {formatted_prompt}")
        
        # Create the request payload
        payload = {
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            "model": MODEL_NAME,
            "stream": True 
        }
        
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                response = self.session.post(
                    INFER_SERVER_URL, 
                    headers=HEADERS, 
                    json=payload, 
                    stream=True,
                    timeout=INFER_SERVER_TIMEOUT
                )
                
                if response.status_code != 200:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if response.status_code >= 500:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return
                
                # Process the streamed response
                token_counter = 0
                estimated_reasonable_tokens = min(len(prompt) * APPROX_TOKEN_PER_CHARACTER, MAX_TOKENS // 2)
                logger.info(f"Estimated reasonable token count: {estimated_reasonable_tokens}")
                
                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            
                            if data_str.strip() == '[DONE]':
                                logger.info("Received [DONE] marker - generation complete")
                                break
                                
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    token_chunk = data['choices'][0].get('text', '')
                                    
                                    # Check for end tokens
                                    if any(f'<custom_token_{end_id}>' in token_chunk for end_id in END_TOKEN_IDS):
                                        logger.info(f"End token detected: {token_chunk[:100]}... - stopping")
                                        break
                                    
                                    # Extract custom tokens
                                    tokens_in_chunk = re.findall(CUSTOM_TOKEN_PATTERN, token_chunk)
                                    
                                    for token_text in tokens_in_chunk:
                                        token_counter += 1
                                        perf_monitor.add_tokens()
                                        
                                        # Safety checks
                                        if any(f'custom_token_{end_id}' in token_text for end_id in END_TOKEN_IDS):
                                            logger.info(f"End token detected: {token_text}")
                                            return
                                        
                                        if token_counter >= max_tokens:
                                            logger.info(f"Max tokens reached ({max_tokens})")
                                            return
                                        
                                        if token_counter > estimated_reasonable_tokens:
                                            logger.info(f"Exceeded reasonable token count ({token_counter} > {estimated_reasonable_tokens})")
                                            return
                                        
                                        if token_text:
                                            yield token_text
                                    
                                    # Process non-custom-token content
                                    remaining_text = re.sub(CUSTOM_TOKEN_PATTERN, '', token_chunk).strip()
                                    if remaining_text and not any(str(end_id) in remaining_text for end_id in END_TOKEN_IDS):
                                        for chunk in remaining_text.split():
                                            if chunk:
                                                token_counter += 1
                                                perf_monitor.add_tokens()
                                                
                                                if token_counter >= max_tokens:
                                                    logger.info(f"Max tokens reached ({max_tokens})")
                                                    return
                                                
                                                yield chunk
                                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error: {e}")
                                continue
                            except Exception as e:
                                logger.warning(f"Error processing token chunk: {e}")
                                continue
                
                # Generation completed successfully
                generation_time = time.time() - start_time
                tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
                logger.info(f"API generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
                return
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out after {INFER_SERVER_TIMEOUT} seconds")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Generation failed.")
                    return
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error to API at {INFER_SERVER_URL}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Generation failed.")
                    return
    
    async def generate_tokens_async(self, prompt: str, voice: str = DEFAULT_VOICE, **kwargs) -> Generator[str, None, None]:
        """Async wrapper for token generation."""
        # Convert sync generator to async
        sync_gen = self.generate_tokens(prompt, voice, **kwargs)
        for token in sync_gen:
            yield token


class VLLMTokenGenerator(BaseTokenGenerator):
    """
    Token generator using integrated vLLM engine.
    Provides ultra-low latency generation with immediate model loading.
    """
    
    def __init__(self, model_path: str = None, dtype: str = DTYPE,
                 max_model_len: int = MAX_MODEL_LEN,
                 gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
                 max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
                 max_num_seqs: int = MAX_NUM_SEQS):
        """Initialize the vLLM token generator."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available. Install vLLM to use integrated engine.")
        
        self.model_path = model_path or MODEL_NAME
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        
        self.engine: Optional[AsyncLLMEngine] = None
        self._initialized = False
        
        logger.info(f" VLLMTokenGenerator configured: Model: {self.model_path}, Max Length: {max_model_len}, GPU Memory: {gpu_memory_utilization}")
    
    async def initialize_engine(self):
        """Initialize the vLLM engine"""
        if self._initialized:
            logger.warning("vLLM engine already initialized")
            return
            
        logger.info("Initializing vLLM engine for immediate loading...")
        start_time = time.time()
        
        try:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                disable_log_requests=True,
                enforce_eager=False,
                max_num_seqs=self.max_num_seqs,
                enable_chunked_prefill=True,
                max_num_batched_tokens=self.max_num_batched_tokens,
                enable_prefix_caching=True,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            
            init_time = time.time() - start_time
            logger.info(f" vLLM engine initialized in {init_time:.2f}s")
            
            # Warmup the engine
            await self._warmup_engine()
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise RuntimeError(f"vLLM engine initialization failed: {e}")
    
    async def _warmup_engine(self):
        """
        Warmup the vLLM engine with dummy inference to eliminate cold start latency.
        This ensures the first real inference will be fast.
        """
        logger.info("Warming up vLLM engine...")
        
        try:
            dummy_prompt = f"{DEFAULT_VOICE}: Hello world"
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=100,
                repetition_penalty=1.0,
                skip_special_tokens=False,
            )
            
            request_id = f"warmup-{int(time.time() * 1000)}"
            
            async for request_output in self.engine.generate(
                prompt=dummy_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                pass
            
            logger.info("vLLM engine warmup completed")
            
        except Exception as e:
            logger.warning(f"vLLM engine warmup failed: {e}")
    
    async def generate_tokens_async(self, prompt: str, voice: str = DEFAULT_VOICE,
                                  temperature: float = TEMPERATURE, top_p: float = TOP_P,
                                  max_tokens: int = MAX_TOKENS,
                                  repetition_penalty: float = REPETITION_PENALTY,
                                  request_id: Optional[str] = None) -> Generator[str, None, None]:
        """
        Generate tokens using vLLM engine.
        
        """
        if not self._initialized or self.engine is None:
            raise RuntimeError("vLLM engine not initialized. Call initialize_engine() first.")
        
        formatted_prompt = format_prompt(prompt, voice)
        logger.info(f"vLLM generation for: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        if request_id is None:
            request_id = f"tts-{int(time.time() * 1000)}"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=False,
        )
        
        start_time = time.time()
        token_count = 0
        previous_text = ""
        estimated_reasonable_tokens = min(len(prompt) * 15, max_tokens // 2)
        
        try:
            async for request_output in self.engine.generate(
                prompt=formatted_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                
            ):
                if request_output.outputs:
                    output = request_output.outputs[0]
                    current_text = output.text
                    
                    # Get incremental text
                    if len(current_text) > len(previous_text):
                        new_text = current_text[len(previous_text):]
                        previous_text = current_text
                        
                        # Check for end tokens
                        if any(f'custom_token_{end_id}' in new_text for end_id in END_TOKEN_IDS):
                            logger.info(f"End token detected: {new_text[:100]}...")
                            return
                        
                        if new_text.strip():
                            token_count += 1
                            perf_monitor.add_tokens()
                            
                            # Safety checks
                            if token_count >= max_tokens:
                                logger.info(f"Max tokens reached ({max_tokens})")
                                return
                            
                            if token_count > estimated_reasonable_tokens:
                                logger.info(f"Exceeded reasonable token count ({token_count} > {estimated_reasonable_tokens})")
                                return
                            
                            yield new_text
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
        
        finally:
            generation_time = time.time() - start_time
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            logger.info(f"vLLM generation complete: {token_count} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
    
    def generate_tokens(self, prompt: str, voice: str = DEFAULT_VOICE, **kwargs) -> Generator[str, None, None]:
        """Synchronous wrapper for async token generation."""
        if not self._initialized or self.engine is None:
            raise RuntimeError("vLLM engine not initialized. Call initialize_engine() first.")
        
        # Convert async generator to sync using queue
        token_queue = queue.Queue()
        exception_holder = [None]
        
        async def producer():
            try:
                async for token in self.generate_tokens_async(prompt, voice, **kwargs):
                    token_queue.put(token)
            except Exception as e:
                exception_holder[0] = e
            finally:
                token_queue.put(None)  # Sentinel
        
        def run_producer():
            asyncio.run(producer())
        
        # Start producer in separate thread
        thread = threading.Thread(target=run_producer)
        thread.daemon = True
        thread.start()
        
        # Yield tokens from queue
        while True:
            token = token_queue.get()
            if token is None:  # Sentinel
                break
            if exception_holder[0]:
                raise exception_holder[0]
            yield token
        
        thread.join()


class TTSService:
    """
    High-level TTS service that orchestrates token generation and audio processing.
    Handles both streaming and batch processing with automatic engine selection.
    """
    
    def __init__(self):
        """Initialize the TTS service with appropriate token generator."""
        if USE_INTEGRATED_VLLM:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available but USE_INTEGRATED_VLLM is enabled")
            self.token_generator = VLLMTokenGenerator()
            logger.info("TTSService initialized with integrated vLLM")
        else:
            self.token_generator = APITokenGenerator()
            logger.info("TTSService initialized with external API")
    
    async def initialize(self):
        """Initialize the service (call during startup)."""
        if isinstance(self.token_generator, VLLMTokenGenerator):
            await self.token_generator.initialize_engine()
        logger.info("TTSService ready for ultra-low latency generation")
    
    def generate_speech(self, prompt: str, voice: str = DEFAULT_VOICE, 
                       output_file: Optional[str] = None, use_batching: bool = True,
                       max_batch_chars: int = MAX_BATCH_CHARS, **kwargs) -> List[bytes]:
        """
        Generate speech from text with automatic batching for long texts.
        
        Args:
            prompt: Text to convert to speech
            voice: Voice to use
            output_file: Optional output file path
            use_batching: Whether to use sentence-based batching
            max_batch_chars: Maximum characters per batch
            **kwargs: Additional generation parameters
            
        Returns:
            List of audio chunks as bytes
        """
        logger.info(f"🎵 Generating speech for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        logger.info(f"   Voice: {voice}, Batching: {use_batching}")
        
        start_time = time.time()
        
        # For shorter text, use direct processing
        if not use_batching or len(prompt) < max_batch_chars:
            token_gen = self.token_generator.generate_tokens(prompt, voice, **kwargs)
            result = tokens_decoder_sync(token_gen, output_file)
            
            total_time = time.time() - start_time
            logger.info(f"Speech generation completed in {total_time:.2f} seconds")
            return result
        
        # For longer text, use sentence-based batching
        logger.info(f"Using sentence-based batching for {len(prompt)} characters")
        
        sentences = split_text_into_sentences(prompt)
        logger.info(f"Split into {len(sentences)} segments")
        
        # Create batches
        batches = []
        current_batch = ""
        
        for sentence in sentences:
            if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
                batches.append(current_batch)
                current_batch = sentence
            else:
                if current_batch:
                    current_batch += " "
                current_batch += sentence
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches for processing")
        
        # Process batches
        all_audio_segments = []
        batch_temp_files = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
            
            temp_output_file = None
            if output_file:
                temp_output_file = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
                batch_temp_files.append(temp_output_file)
            
            token_gen = self.token_generator.generate_tokens(batch, voice, **kwargs)
            batch_segments = tokens_decoder_sync(token_gen, temp_output_file)
            all_audio_segments.extend(batch_segments)
        
        # Stitch files if needed
        if output_file and batch_temp_files:
            stitch_wav_files(batch_temp_files, output_file)
            
            # Clean up temporary files
            for temp_file in batch_temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove temporary file {temp_file}: {e}")
        
        total_time = time.time() - start_time
        if all_audio_segments:
            total_bytes = sum(len(segment) for segment in all_audio_segments)
            duration = total_bytes / (2 * 24000)  # 2 bytes per sample at 24kHz
            logger.info(f"Generated {duration:.2f}s of audio in {total_time:.2f}s")
            logger.info(f"Realtime factor: {duration/total_time:.2f}x")
        
        logger.info(f"Batch speech generation completed in {total_time:.2f} seconds")
        return all_audio_segments
    
    async def stream_speech(self, prompt: str, voice: str = DEFAULT_VOICE, **kwargs):
        """
        Stream speech audio chunks as they're generated.
        
        Args:
            prompt: Text to convert to speech
            voice: Voice to use
            **kwargs: Additional generation parameters
            
        Yields:
            Audio chunks as bytes
        """
        logger.info(f"Streaming speech generation for: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        if isinstance(self.token_generator, VLLMTokenGenerator):
            # Use async generation for vLLM
            token_gen = self.token_generator.generate_tokens_async(prompt, voice, **kwargs)
        else:
            # Convert sync to async for API
            sync_gen = self.token_generator.generate_tokens(prompt, voice, **kwargs)
            async def async_wrapper():
                for token in sync_gen:
                    yield token
            token_gen = async_wrapper()
        
        # Stream audio chunks
        async for audio_chunk in tokens_decoder(token_gen):
            if audio_chunk:
                yield audio_chunk


_global_tts_service: Optional[TTSService] = None

def get_tts_service() -> TTSService:
    """Get the global TTS service instance."""
    global _global_tts_service
    if _global_tts_service is None:
        _global_tts_service = TTSService()
    return _global_tts_service

async def initialize_tts_service():
    """Initialize the TTS service during application startup."""
    service = get_tts_service()
    
    # Initialize the token generator (vLLM if enabled)
    await service.initialize()
    
    # Also initialize the SNAC decoder to avoid lazy loading on first request
    from .decoder import get_snac_decoder
    logger.info("Pre-loading SNAC decoder alongside vLLM...")
    snac_start_time = time.time()
    
    # This will trigger the SNAC model loading
    _ = get_snac_decoder()
    
    snac_init_time = time.time() - snac_start_time
    logger.info(f"SNAC decoder pre-loaded in {snac_init_time:.2f}s")
    
    logger.info("TTS Service ready for ultra-low latency generation with both vLLM and SNAC models loaded")

async def stream_speech_from_api(prompt: str, voice: str = DEFAULT_VOICE, **kwargs):
    """
    Stream speech audio chunks as they're generated.
    
    Args:
        prompt: Text to convert to speech
        voice: Voice to use for generation
        **kwargs: Additional generation parameters
        
    Yields:
        Audio chunks as bytes
    """
    service = get_tts_service()
    async for chunk in service.stream_speech(prompt, voice, **kwargs):
        yield chunk 