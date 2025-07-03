import requests
import time
import statistics
import os
import struct
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env file
SERVER_HOST = "localhost"
SERVER_PORT = "9090"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
TEST_TEXT = "Hi How are you doing today? So happy to see you again."
VOICE = "tara"
NUM_RUNS = 5
WARMUP_TEXT = "Doing warmup"
OUTPUT_DIR = "outputs"

# Audio parameters for WAV header generation
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1

# Create outputs directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_wav_header(sample_rate=SAMPLE_RATE, bits_per_sample=BITS_PER_SAMPLE, channels=CHANNELS, data_size=0):
    """Generate WAV header for PCM audio data."""
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    
    # Calculate file size (header + data)
    file_size = 36 + data_size
    
    # Build WAV header
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', file_size))
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
    header.extend(struct.pack('<I', data_size))
    
    return bytes(header)

def run_single_test(text, voice, save_file=None, session=None):
    """Run a single TTS test and return timing metrics.

    Parameters
    ----------
    text : str
        Prompt text to send to the TTS service.
    voice : str
        Voice name.
    save_file : str or None, optional
        If provided, the received raw PCM stream will be wrapped with a WAV header
        and written to this path.
    session : requests.Session or None, optional
        If supplied, the session's ``post`` method will be used, allowing the
        underlying TCP connection to be kept alive across multiple runs.  If
        ``None`` (default) a standalone ``requests.post`` call is made.
    """
    start_time = time.time()
    ttfb = None

    # Use the provided session for connection-pooled requests, or fall back to
    # the top-level ``requests`` module.
    http = session or requests

    try:
        response = http.post(
            f"{BASE_URL}/v1/audio/speech/stream",
            json={
                "input": text,
                "voice": voice,
            },
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            bytes_received = 0
            audio_data = bytearray()
            
            # Initialize an iterator for the response content with a more efficient chunk size.
            content_iterator = response.iter_content(chunk_size=4096)
            
            # Process chunks and capture TTFB when the first non-empty chunk arrives
            for chunk in content_iterator:
                if not chunk:
                    # Skip keep-alive chunks or empty payloads
                    continue
                if ttfb is None:
                    ttfb = time.time() - start_time  # First byte received
                audio_data.extend(chunk)
                bytes_received += len(chunk)
            
            # No audio was received – treat as failure
            if bytes_received == 0:
                return {'success': False, 'error': 'No audio data received (zero bytes)'}
            
            # Save the complete audio as a WAV file if requested.
            if save_file:
                print(f" -> Saving {len(audio_data)} bytes to {save_file}", end="")
                wav_header = generate_wav_header(data_size=len(audio_data))
                with open(save_file, 'wb') as f:
                    f.write(wav_header)
                    f.write(audio_data)
            
            total_time = time.time() - start_time
            
            bytes_per_second = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
            audio_duration = len(audio_data) / bytes_per_second if bytes_per_second > 0 else 0

            return {
                'success': True,
                'ttfb': ttfb,
                'total_time': total_time,
                'bytes_received': bytes_received,
                'audio_duration': audio_duration
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f"Request failed: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"An unexpected error occurred: {e}"}

def main():
    print("TTS Streaming Benchmark")
    print(f"Text: '{TEST_TEXT}' ({len(TEST_TEXT)} characters)")
    print(f"Voice: {VOICE}")
    print(f"Runs: {NUM_RUNS}")
    print(f"Base URL: {BASE_URL}")
    print("Note: Converting raw audio streams to WAV format with proper headers")
    
    # Create a single session so every run reuses the same HTTP connection.
    with requests.Session() as session:

        # Warmup run
        print("\nRunning warmup...")
        warmup_result = run_single_test(WARMUP_TEXT, VOICE, session=session)
        if warmup_result['success']:
            print(f"Warmup complete: TTFB {warmup_result['ttfb']:.3f}s, Total {warmup_result['total_time']:.3f}s")
        else:
            print(f"Warmup failed: {warmup_result['error']}")
        
        # Benchmark runs
        print(f"\nStarting {NUM_RUNS} benchmark runs...")
        results = []
        
        for run in range(1, NUM_RUNS + 1):
            print(f"Run {run}/{NUM_RUNS}...", end=" ")
            
            # Save as .wav files with proper headers
            audio_filename = f"{OUTPUT_DIR}/run_{run}.wav"
            result = run_single_test(TEST_TEXT, VOICE, audio_filename, session=session)
            
            if result['success']:
                print(f" TTFB: {result['ttfb']:.3f}s, Total: {result['total_time']:.3f}s")
                results.append(result)
            else:
                print(f" Failed: {result['error']}")
        
        # Calculate and display statistics
        if results:
            successful_runs = len(results)
            ttfb_times = [r['ttfb'] for r in results if r['ttfb'] is not None]
            total_times = [r['total_time'] for r in results]
            audio_durations = [r['audio_duration'] for r in results if r.get('audio_duration')]
            rtfs = [r['total_time'] / r['audio_duration'] for r in results if r.get('audio_duration', 0) > 0]
            
            print(f"\n--- Benchmark Results ({successful_runs}/{NUM_RUNS} successful) ---")
            
            if audio_durations:
                avg_duration = statistics.mean(audio_durations)
                print("\nAudio Characteristics:")
                print(f"  - Sample Rate: {SAMPLE_RATE}Hz, {BITS_PER_SAMPLE}-bit, {CHANNELS} channel(s)")
                print(f"  - Average Audio Duration: {avg_duration:.3f}s")
            
            print("\nPerformance Metrics:")
            if ttfb_times:
                ttfb_mean = statistics.mean(ttfb_times)
                ttfb_stdev = statistics.stdev(ttfb_times) if len(ttfb_times) > 1 else 0
                print(f"  - TTFB (Time to First Byte):")
                print(f"    - Mean: {ttfb_mean:.3f}s, StdDev: {ttfb_stdev:.3f}s, Min: {min(ttfb_times):.3f}s, Max: {max(ttfb_times):.3f}s")
            
            if total_times:
                total_mean = statistics.mean(total_times)
                total_stdev = statistics.stdev(total_times) if len(total_times) > 1 else 0
                print(f"  - Total Generation Time:")
                print(f"    - Mean: {total_mean:.3f}s, StdDev: {total_stdev:.3f}s, Min: {min(total_times):.3f}s, Max: {max(total_times):.3f}s")

            if rtfs:
                rtf_mean = statistics.mean(rtfs)
                rtf_stdev = statistics.stdev(rtfs) if len(rtfs) > 1 else 0
                print(f"  - RTF (Real-Time Factor):")
                print(f"    - Mean: {rtf_mean:.3f}, StdDev: {rtf_stdev:.3f}, Min: {min(rtfs):.3f}, Max: {max(rtfs):.3f}")
                print(f"    (Lower is better; RTF < 1 means faster than real-time generation)")

            print(f"\nAudio for each run saved in '{OUTPUT_DIR}/'")
        
        else:
            print("\nNo successful runs completed.")

if __name__ == "__main__":
    main() 