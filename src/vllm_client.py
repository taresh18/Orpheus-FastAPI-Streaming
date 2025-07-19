import aiohttp
import asyncio
import orjson
import logging
import os
from typing import AsyncGenerator, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for making requests to VLLM serve server."""
    
    def __init__(self, server_url: str = None):
        """Initialize the VLLM serve client"""
        try:
            # Load all environment variables
            self.server_url = server_url or os.environ["SERVE_URL"]
            self.model_name = os.environ["MODEL_NAME"]
            self.api_key = os.environ["API_KEY"]
            voices_str = os.environ["AVAILABLE_VOICES"]
            
            # Parse and set values
            self.available_voices = [voice.strip() for voice in voices_str.split(',')]
            self.connect_timeout = float(os.environ["CONNECT_TIMEOUT"])
            self.read_timeout = float(os.environ["READ_TIMEOUT"])
            self.total_timeout = float(os.environ["TOTAL_TIMEOUT"])
            self.max_connections = int(os.environ["MAX_CONNECTIONS"])
            self.max_connections_per_host = int(os.environ["MAX_CONNECTIONS_PER_HOST"])
            max_concurrent_requests = int(os.environ["MAX_CONCURRENT_REQUESTS"])
            
            # Store sampling parameters for later use
            self.stop_token_ids = [int(token.strip()) for token in os.environ["STOP_TOKEN_IDS"].split(',')]
            self.max_tokens = int(os.environ["MAX_TOKENS"])
            self.temperature = float(os.environ["TEMPERATURE"])
            self.top_p = float(os.environ["TOP_P"])
            self.repetition_penalty = float(os.environ["REPETITION_PENALTY"])
            self.max_retries = int(os.environ["MAX_RETRIES"])
            self.retry_delay = float(os.environ["RETRY_DELAY"])
            
            # Session for connection pooling
            self._session = None
            self._semaphore = asyncio.Semaphore(max_concurrent_requests)
            
        except KeyError as e:
            raise ValueError(f"Missing required environment variable: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid environment variable value: {e}")
        
        logger.info(f"VLLMClient initialized with server: {self.server_url}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Available voices: {self.available_voices}")
        logger.info(f"Timeouts - Connect: {self.connect_timeout}s, Read: {self.read_timeout}s, Total: {self.total_timeout}s")
        logger.info(f"Max concurrent requests: {max_concurrent_requests}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with proper configuration."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.connect_timeout,
                sock_read=self.read_timeout,
                total=self.total_timeout
            )
            
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                keepalive_timeout=120,
                enable_cleanup_closed=True,
                ttl_dns_cache=600,
                use_dns_cache=True,
                force_close=False
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "text/event-stream"
                }
            )
        
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _format_prompt(self, prompt: str, voice: str = "tara") -> str:
        """Format the prompt for the VLLM serve API."""
        adapted_prompt = f"{voice}: {prompt}"
        return f"<custom_token_3><|begin_of_text|>{adapted_prompt}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    
    async def generate_tokens(self, prompt: str, voice: str = "tara") -> AsyncGenerator[str, None]:
        """Generate tokens asynchronously from the VLLM serve server.
        
        Args:
            prompt: The text prompt to generate tokens for
            voice: The voice to use for generation
            
        Yields:
            Generated token strings
        """
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' is not available. Available voices: {self.available_voices}")
        
        formatted_prompt = self._format_prompt(prompt, voice)
                
        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stream": True,  # Always stream for better performance
            "stop_token_ids": self.stop_token_ids
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                # Use semaphore to limit concurrent requests
                async with self._semaphore:
                    session = await self._get_session()

                logger.info(f"Sending request to {self.server_url}/completions")
                
                # Use regular aiohttp with improved SSE parsing
                async with session.post(
                    f"{self.server_url}/completions",
                    json=payload,
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"VLLM serve API error: {response.status} - {error_text}")
                        raise Exception(f"VLLM serve API error: {response.status} - {error_text}")
                    
                    # Process the streaming response with improved parsing
                    buffer = ""
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            buffer += line_str + '\n'
                            
                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if line.startswith('data: '):
                                    data_str = line[6:]
                                    
                                    if data_str.strip() == '[DONE]':
                                        return
                                    
                                    try:
                                        data = orjson.loads(data_str)
                                        if 'choices' in data and len(data['choices']) > 0:
                                            token_text = data['choices'][0].get('text', '')
                                            if token_text:
                                                yield token_text
                                    except orjson.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON from stream: {e}")
                    
                    # If we get here, the request was successful
                    break
                                        
            except asyncio.TimeoutError:
                logger.error(f"Request to VLLM serve timed out (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    logger.info(f"Retrying request after timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception("Request to VLLM serve timed out after all retries")
            except aiohttp.ClientError as e:
                logger.error(f"HTTP client error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    logger.info(f"Retrying request after HTTP error (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"HTTP client error after all retries: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in generate_tokens (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    logger.info(f"Retrying request after unexpected error (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 