import requests
import time
import statistics
import os
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

# WAV header size (44 bytes for standard PCM WAV)
WAV_HEADER_SIZE = 44

# Create outputs directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            
            if save_file:
                with open(save_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            if ttfb is None and bytes_received + len(chunk) > WAV_HEADER_SIZE:
                                ttfb = time.time() - start_time
                            if save_file:
                                f.write(chunk)
                            bytes_received += len(chunk)
            else:
                # Warmup run - don't save file
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        if ttfb is None and bytes_received + len(chunk) > WAV_HEADER_SIZE:
                            ttfb = time.time() - start_time
                        bytes_received += len(chunk)
            
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
    
    else:
        print("\nNo successful runs completed.")

if __name__ == "__main__":
    main() 