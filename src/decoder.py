import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys
from typing import List, Optional, Generator, Dict, Tuple, Any
from snac import SNAC

from .consts import *
from .utils import perf_monitor
from src.logger import get_logger

logger = get_logger()

class SnacDecoder:
    """
    High-performance SNAC decoder for Orpheus TTS.
    Handles token-to-audio conversion with CUDA optimization and intelligent caching.
    """
    
    def __init__(self):
        """Initialize the SNAC decoder with model and cache."""
        logger.info("Initializing SnacDecoder...")
        
        # Initialize model
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.model = self.model.to(DEVICE)
        logger.info(f"SNAC model loaded on {DEVICE}")
        
        # Initialize cache for token processing with LRU management
        self.token_id_cache: Dict[Tuple[str, int], int] = {}
        self.cache_access_order: List[Tuple[str, int]] = []
        
        # Warmup the model
        self._warmup_model()
            
    def _warmup_model(self) -> None:
        """
        Warmup the SNAC model with dummy data to ensure it's ready for inference.
        This helps eliminate cold start latency on first real inference.
        """
        logger.info("Warming up SNAC model...")
        
        try:
            # Create dummy codes similar to what we'd get from real tokens
            dummy_frames = 7
            
            codes_0 = torch.randint(0, 1024, (dummy_frames,), dtype=torch.int32, device=DEVICE)
            codes_1 = torch.randint(0, 1024, (dummy_frames * 2,), dtype=torch.int32, device=DEVICE)
            codes_2 = torch.randint(0, 1024, (dummy_frames * 4,), dtype=torch.int32, device=DEVICE)
            
            dummy_codes = [
                codes_0.unsqueeze(0),
                codes_1.unsqueeze(0), 
                codes_2.unsqueeze(0)
            ]
            
            # Run warmup inference
            if CUDA_STREAM is not None:
                with torch.cuda.stream(CUDA_STREAM), torch.inference_mode():
                    _ = self.model.decode(dummy_codes)
            else:
                with torch.inference_mode():
                    _ = self.model.decode(dummy_codes)
            
            logger.info("SNAC model warmup completed")
            
        except Exception as e:
            logger.warning(f"SNAC model warmup failed: {e}")
    
    def _manage_cache_size(self) -> None:
        """
        Manage cache size using LRU eviction when cache gets too large.
        Prevents unbounded memory growth while maintaining performance.
        """
        if len(self.token_id_cache) >= MAX_CACHE_SIZE:
            # Remove oldest 20% of entries to avoid frequent evictions
            evict_count = MAX_CACHE_SIZE // 5
            
            # Remove least recently used entries
            for _ in range(min(evict_count, len(self.cache_access_order))):
                if self.cache_access_order:
                    oldest_key = self.cache_access_order.pop(0)
                    self.token_id_cache.pop(oldest_key, None)
    
    def _update_cache_access(self, cache_key: Tuple[str, int]) -> None:
        """
        Update cache access order for LRU tracking.
        
        Args:
            cache_key: The key to update
            
        Returns: 
            None
        """
        # Move to end (most recently used)
        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)
        self.cache_access_order.append(cache_key)
    
    def _quick_token_filter(self, token_string: str) -> bool:
        """
        Pre-filter to quickly reject obviously invalid tokens.
        
        Args:
            token_string: The token string to check
            
        Returns:
            bool: True if token might be valid, False if definitely invalid
        """
        return (len(token_string) >= 16 and  # Minimum length for "<custom_token_X>"
                token_string.startswith('<custom_token_') and 
                token_string.endswith('>') and
                token_string.count('<') == 1 and  # Ensure no nested tokens
                token_string.count('>') == 1)
    
    def turn_token_into_id(self, token_string: str, index: int) -> Optional[int]:
        """
        Convert token string to ID with optimized caching and LRU eviction.
        
        Args:
            token_string: The token string to convert
            index: Position index used for token offset calculation
            
        Returns:
            int: Token ID if valid, None otherwise
        """        
        # Quick pre-filter to avoid processing obviously invalid tokens
        if not self._quick_token_filter(token_string):
            return None
        
        # Check cache first
        cache_key = (token_string, index % 7)
        if cache_key in self.token_id_cache:
            self._update_cache_access(cache_key)
            return self.token_id_cache[cache_key]
            
        # Process token
        token_string = token_string.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
        
        if last_token_start == -1:
            return None
        
        last_token = token_string[last_token_start:]
        
        if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
            return None
            
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            
            # Cache the result
            self._manage_cache_size()  # Ensure we have space
            self.token_id_cache[cache_key] = token_id
            self._update_cache_access(cache_key)  # Track access for LRU
                
            return token_id
        except (ValueError, IndexError) as e:
            return None
    
    def convert_to_audio(self, multiframe: List[int], count: int) -> Optional[bytes]:
        """
        Generate audio bytes from token ids using snac decoder.
        
        Args:
            multiframe: List of token IDs representing audio frames
            count: Frame count for processing
            
        Returns:
            Audio data as bytes, or None if insufficient frames
        """
        if len(multiframe) < 7:
            return None
      
        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]
        
        # Pre-allocate tensors instead of incrementally building them
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=DEVICE)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=DEVICE)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=DEVICE)
        
        # Use vectorized operations where possible
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=DEVICE)
        
        # Direct indexing is much faster than concatenation in a loop
        for j in range(num_frames):
            idx = j * 7
            
            # Code 0 - single value per frame
            codes_0[j] = frame_tensor[idx]
            
            # Code 1 - two values per frame
            codes_1[j*2] = frame_tensor[idx+1]
            codes_1[j*2+1] = frame_tensor[idx+4]
            
            # Code 2 - four values per frame
            codes_2[j*4] = frame_tensor[idx+2]
            codes_2[j*4+1] = frame_tensor[idx+3]
            codes_2[j*4+2] = frame_tensor[idx+5]
            codes_2[j*4+3] = frame_tensor[idx+6]
        
        # Reshape codes into expected format
        codes = [
            codes_0.unsqueeze(0), 
            codes_1.unsqueeze(0), 
            codes_2.unsqueeze(0)
        ]
        
        # Check tokens are in valid range
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            return None

        # Use CUDA stream for parallel processing if available
        if CUDA_STREAM is not None:
            with torch.cuda.stream(CUDA_STREAM), torch.inference_mode():
                # Decode the audio
                audio_hat = self.model.decode(codes)
                
                # Extract the relevant slice and efficiently convert to bytes
                audio_slice = audio_hat[:, :, 2048:4096]
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            with torch.inference_mode():
                # Decode the audio
                audio_hat = self.model.decode(codes)
                
                # Extract the relevant slice and efficiently convert to bytes
                audio_slice = audio_hat[:, :, 2048:4096]
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
                
        return audio_bytes

# Global SNAC decoder instance
_snac_decoder: Optional[SnacDecoder] = None

def get_snac_decoder() -> SnacDecoder:
    """Get the global SNAC decoder instance."""
    global _snac_decoder
    if _snac_decoder is None:
        _snac_decoder = SnacDecoder()
    return _snac_decoder

async def tokens_decoder(token_gen: Generator[str, None, None]) -> Generator[bytes, None, None]:
    """
    Optimized token decoder with ultra-low TTFB.
    
    Args:
        token_gen: Generator of token strings
        
    Returns:
        Generator of audio chunks as bytes
    """
    decoder = get_snac_decoder()
    buffer = []
    count = 0
    first_chunk_processed = False
    
    async for token_sim in token_gen:       
        token = decoder.turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                # First chunk processing for ultra-low TTFB
                if not first_chunk_processed and count >= MIN_TOKENS_FIRST_CHUNK:
                    # Use exactly 7 tokens for first chunk to minimize TTFB
                    buffer_to_proc = buffer[-MIN_TOKENS_FIRST_CHUNK:]
 
                    logger.info(f"Processing FIRST audio chunk with {len(buffer_to_proc)} tokens")
                    audio_samples = decoder.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
                        perf_monitor.add_audio_chunk()
                
                # Subsequent chunks: Use standard 28-token processing for quality
                elif first_chunk_processed and count % 7 == 0 and count > MIN_TOKENS_SUBSEQUENT:
                    buffer_to_proc = buffer[-MIN_TOKENS_SUBSEQUENT:]
                    audio_samples = decoder.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
                        perf_monitor.add_audio_chunk()
        
        # Check for end tokens
        if any(f'custom_token_{end_id}' in token_sim for end_id in END_TOKEN_IDS):
            logger.info(f"End token detected in decoder: {token_sim[:50]}... - stopping")
            break

def tokens_decoder_sync(token_gen: Generator[str, None, None], 
                       output_file: Optional[str] = None) -> List[bytes]:
    """
    Synchronous token decoder with optimized TTFB approach.
    Processes first audio chunk after just 7 tokens, then uses 28-token sliding window.
    
    Args:
        token_gen: Generator of token strings
        output_file: Optional output file path
        
    Returns:
        List of audio chunks as bytes
    """
    decoder = get_snac_decoder()
    audio_chunks = []
    buffer = []
    count = 0
    first_chunk_processed = False
    
    for token_sim in token_gen:
        if not token_sim or not token_sim.strip():
            continue
            
        # Check for end tokens
        if any(f'custom_token_{end_id}' in token_sim for end_id in END_TOKEN_IDS):
            logger.info(f"End token detected in decoder: {token_sim[:50]}... - stopping")
            break
        
        token = decoder.turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                # CRITICAL OPTIMIZATION: First chunk processing for ultra-low TTFB
                if not first_chunk_processed and count >= MIN_TOKENS_FIRST_CHUNK:
                    # Use exactly 7 tokens for first chunk to minimize TTFB
                    buffer_to_proc = buffer[-MIN_TOKENS_FIRST_CHUNK:]
 
                    logger.info(f"Processing FIRST audio chunk with {len(buffer_to_proc)} tokens for ultra-low TTFB")
                    audio_samples = decoder.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        audio_chunks.append(audio_samples)
                        perf_monitor.add_audio_chunk()
                
                # Subsequent chunks: Use standard 28-token processing for quality
                elif first_chunk_processed and count % 7 == 0 and count > MIN_TOKENS_SUBSEQUENT:
                    buffer_to_proc = buffer[-MIN_TOKENS_SUBSEQUENT:]
                    audio_samples = decoder.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        audio_chunks.append(audio_samples)
                        perf_monitor.add_audio_chunk()
    
    # Save to file if requested
    if output_file and audio_chunks:
        try:
            import wave
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                
                # Write all audio chunks
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk)
            
            logger.info(f"Audio saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving audio to {output_file}: {e}")
    
    return audio_chunks 