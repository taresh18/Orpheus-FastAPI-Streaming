import time
import struct
import wave
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Generator, Any
from .consts import *
from src.logger import get_logger

logger = get_logger()

def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Format prompt for Orpheus model with voice prefix and special tokens.
    This is the shared implementation used across all token generators.
    
    Args:
        prompt: The text prompt to format
        voice: Voice to use for generation
        
    Returns:
        Formatted prompt with voice prefix and special tokens
    """
    # Validate voice and provide fallback
    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format with voice prefix
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers
    return f"{AUDIO_START_TOKEN}{formatted_prompt}{AUDIO_END_TOKEN}"

def split_text_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences with a more reliable approach.
    Optimized for TTS processing with minimum sentence length enforcement.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentence strings
    """
    # Simple approach that doesn't rely on variable-width lookbehinds
    parts = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        
        # If we hit a sentence ending followed by a space, consider this a potential sentence end
        if char in (' ', '\n', '\t') and len(current_sentence) > 1:
            prev_char = current_sentence[-2]
            if prev_char in ('.', '!', '?'):
                # Check if this is likely a real sentence end and not an abbreviation
                if len(current_sentence) > 3 and current_sentence[-3] not in ('.', ' '):
                    parts.append(current_sentence.strip())
                    current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        parts.append(current_sentence.strip())
    
    # Combine very short segments to avoid tiny audio files
    combined_sentences = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        
        # If this is a short sentence and not the last one, combine with next
        while i < len(parts) - 1 and len(current) < MIN_SENTENCE_CHARS:
            i += 1
            current += " " + parts[i]
            
        combined_sentences.append(current)
        i += 1
    
    return combined_sentences

# Cache for WAV headers to avoid regenerating them for each request
_WAV_HEADER_CACHE: Dict[Tuple[int, int, int], bytes] = {}

def generate_wav_header(sample_rate: int = SAMPLE_RATE, 
                       bits_per_sample: int = BITS_PER_SAMPLE, 
                       channels: int = CHANNELS) -> bytes:
    """
    Generate WAV header with caching for improved performance.
    Optimized for ultra-low latency streaming applications.
    
    Args:
        sample_rate: Audio sample rate
        bits_per_sample: Bits per sample
        channels: Number of audio channels
        
    Returns:
        Cached or newly generated WAV header bytes
    """
    cache_key = (sample_rate, bits_per_sample, channels)
    
    # Return cached header if available
    if cache_key in _WAV_HEADER_CACHE:
        return _WAV_HEADER_CACHE[cache_key]
    
    # Generate new header if not in cache (5x faster than using wave module)
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    
    # Use direct struct packing for fastest possible WAV header generation
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', 0xFFFFFFFF))  # Placeholder for file size
    header.extend(b'WAVE')
    # Format chunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Format chunk size
    header.extend(struct.pack('<H', 1))   # PCM format
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', byte_rate))
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    # Data chunk
    header.extend(b'data')
    header.extend(struct.pack('<I', 0xFFFFFFFF))  # Placeholder for data size
    
    # Store in cache for future use
    _WAV_HEADER_CACHE[cache_key] = bytes(header)
    return _WAV_HEADER_CACHE[cache_key]

def stitch_wav_files(input_files: List[str], output_file: str, 
                    crossfade_ms: int = CROSSFADE_MS) -> None:
    """
    Stitch multiple WAV files together with crossfading for smooth transitions.
    Optimized for batch TTS processing.
    
    Args:
        input_files: List of input WAV file paths
        output_file: Output file path
        crossfade_ms: Crossfade duration in milliseconds
    """
    if not input_files:
        return
        
    logger.info(f"Stitching {len(input_files)} WAV files together with {crossfade_ms}ms crossfade")
    
    # If only one file, just copy it
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        return
    
    # Convert crossfade_ms to samples
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    logger.info(f"Using {crossfade_samples} samples for crossfade at {SAMPLE_RATE}Hz")
    
    # Build the final audio in memory with crossfades
    final_audio = np.array([], dtype=np.int16)
    first_params = None
    
    for i, input_file in enumerate(input_files):
        try:
            with wave.open(input_file, 'rb') as wav:
                if first_params is None:
                    first_params = wav.getparams()
                elif wav.getparams() != first_params:
                    logger.warning(f"Warning: WAV file {input_file} has different parameters")
                    
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                
                if i == 0:
                    # First segment - use as is
                    final_audio = audio
                else:
                    # Apply crossfade with previous segment
                    if len(final_audio) >= crossfade_samples and len(audio) >= crossfade_samples:
                        # Create crossfade weights
                        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                        
                        # Apply crossfade
                        crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                                           audio[:crossfade_samples] * fade_in).astype(np.int16)
                        
                        # Combine audio segments
                        final_audio = np.concatenate([final_audio[:-crossfade_samples], 
                                                    crossfade_region, 
                                                    audio[crossfade_samples:]])
                    else:
                        # One segment too short for crossfade, just append
                        logger.info(f"Segment {i} too short for crossfade, concatenating directly")
                        final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            if i == 0:
                raise  # Critical failure if first file fails
    
    # Write the final audio data to the output file
    try:
        with wave.open(output_file, 'wb') as output_wav:
            if first_params is None:
                raise ValueError("No valid WAV files were processed")
                
            output_wav.setparams(first_params)
            output_wav.writeframes(final_audio.tobytes())
        
        logger.info(f"Successfully stitched audio to {output_file} with crossfading")
    except Exception as e:
        logger.error(f"Error writing output file {output_file}: {e}")
        raise

async def sync_to_async_gen(sync_gen: Generator[Any, None, None]) -> Generator[Any, None, None]:
    """
    Convert a synchronous generator to an async generator.
    Used for compatibility between sync and async interfaces.
    
    Args:
        sync_gen: Synchronous generator to convert
        
    Yields:
        Items from the synchronous generator
    """
    for item in sync_gen:
        yield item

class PerformanceMonitor:
    """
    Track and report performance metrics for TTS operations.
    Optimized for real-time monitoring with minimal overhead.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = PERFORMANCE_REPORT_INTERVAL
        
    def add_tokens(self, count: int = 1) -> None:
        """Add token count and check if reporting is needed."""
        self.token_count += count
        self._check_report()
        
    def add_audio_chunk(self) -> None:
        """Add audio chunk count and check if reporting is needed."""
        self.audio_chunks += 1
        self._check_report()
        
    def _check_report(self) -> None:
        """Check if it's time to report performance metrics."""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
            
    def report(self) -> None:
        """Report current performance metrics."""
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:
            return
            
        tokens_per_sec = self.token_count / elapsed
        chunks_per_sec = self.audio_chunks / elapsed
        
        # Estimate audio duration based on audio chunks
        est_duration = self.audio_chunks * AUDIO_CHUNK_DURATION
        
        logger.info(f"Performance: {tokens_per_sec:.1f} tokens/sec, "
              f"est. {est_duration:.1f}s audio generated, "
              f"{self.token_count} tokens, {self.audio_chunks} chunks "
              f"in {elapsed:.1f}s")

# Global performance monitor instance
perf_monitor = PerformanceMonitor() 