import requests
import time
import statistics
import os
import struct
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env file
SERVER_HOST = os.getenv("SERVER_HOST")
SERVER_PORT = os.getenv("SERVER_PORT")
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

def run_single_test(text, voice, save_file=None):
    """Run a single TTS test and return timing metrics."""
    start_time = time.time()
    ttfb = None
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/audio/speech/stream",
            json={
                "input": text,
                "voice": voice,
                "response_format": "wav"
            },
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            bytes_received = 0
            audio_data = bytearray()
            
            # Collect all audio data first
            for chunk in response.iter_content(chunk_size=1):
                if chunk:
                    # TTFB now marks first audio chunk received (no WAV header expected)
                    if ttfb is None:
                        ttfb = time.time() - start_time
                    audio_data.extend(chunk)
                    bytes_received += len(chunk)
            
            # Save as WAV file if requested
            if save_file:
                wav_header = generate_wav_header(data_size=len(audio_data))
                with open(save_file, 'wb') as f:
                    f.write(wav_header)
                    f.write(audio_data)
            
            total_time = time.time() - start_time
            
            return {
                'success': True,
                'ttfb': ttfb,
                'total_time': total_time,
                'bytes_received': bytes_received
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("TTS Streaming Benchmark")
    print(f"Text: '{TEST_TEXT}' ({len(TEST_TEXT)} characters)")
    print(f"Voice: {VOICE}")
    print(f"Runs: {NUM_RUNS}")
    print(f"Base URL: {BASE_URL}")
    print("Note: Converting raw audio streams to WAV format with proper headers")
    
    # Warmup run
    print("\nRunning warmup...")
    warmup_result = run_single_test(WARMUP_TEXT, VOICE)
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
        result = run_single_test(TEST_TEXT, VOICE, audio_filename)
        
        if result['success']:
            print(f"TTFB: {result['ttfb']:.3f}s, Total: {result['total_time']:.3f}s")
            results.append(result)
        else:
            print(f"Failed: {result['error']}")
    
    # Calculate and display statistics
    if results:
        successful_runs = len(results)
        ttfb_times = [r['ttfb'] for r in results if r['ttfb'] is not None]
        total_times = [r['total_time'] for r in results]
        
        print(f"\nResults ({successful_runs}/{NUM_RUNS} successful):")
        print(f"TTFB - Mean: {statistics.mean(ttfb_times):.3f}s, Min: {min(ttfb_times):.3f}s, Max: {max(ttfb_times):.3f}s")
        print(f"Total - Mean: {statistics.mean(total_times):.3f}s, Min: {min(total_times):.3f}s, Max: {max(total_times):.3f}s")
        
        print(f"\nAudio files saved as WAV format with proper headers")
        print(f"Audio specs: {SAMPLE_RATE}Hz, {BITS_PER_SAMPLE}-bit, {CHANNELS} channel(s)")
    
    else:
        print("\nNo successful runs completed.")

if __name__ == "__main__":
    main() 