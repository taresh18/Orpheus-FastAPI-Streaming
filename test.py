import requests
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import AutoTokenizer
import torch

model_name = "dst19/jess-voice-merged"

base_url = "http://0.0.0.0:9090/v1"
api_key = "tensorrt_llm"
REQUEST_TIMEOUT = 5  # Reduced timeout for faster failure detection

print(f"starting test")

tokenizer = AutoTokenizer.from_pretrained(model_name)

def _format_prompt(prompt, voice="tara"):
    # This formatting is specific to the Orpheus model
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[ 128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    prompt_string = tokenizer.decode(all_input_ids[0])
    return prompt_string

try:
    prompt = f"tara: Hey What's up? How are you doing? So happy to see you again!"
    formatted_prompt = f"<custom_token_3><|begin_of_text|>{prompt}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    # stop_token = "<custom_token_2>"
    stop_token = 128258
    
    payload = {
        "model": model_name,
        "prompt": formatted_prompt,
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "stream": True,  # Always stream for better performance
        "stop_token_ids": [stop_token]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Connection": "keep-alive",  # Keep connection alive
        "Accept": "text/event-stream"  # Explicitly request SSE
    }
    
    st = time.time()
    
    # Create a session with optimized connection pooling
    session = requests.Session()
    
    # Configure connection pooling for better performance
    adapter = HTTPAdapter(
        pool_connections=10,  # Number of connection pools
        pool_maxsize=10,      # Max connections per pool
        max_retries=Retry(
            total=1,          # Only 1 retry for faster failure
            backoff_factor=0.1,  # Minimal backoff
            status_forcelist=[500, 502, 503, 504]
        )
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Use localhost instead of 0.0.0.0 for better performance
    local_url = base_url.replace("0.0.0.0", "127.0.0.1")
    
    try:
        response = session.post(
            f"{local_url}/completions", 
            headers=headers, 
            json=payload, 
            stream=True,
            timeout=(1, REQUEST_TIMEOUT)  # (connect_timeout, read_timeout)
        )
        
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Error details: {response.text}")
        
        # Process the streamed response with better buffering
        buffer = ""
        token_counter = 0
        num_chunks = 0
        first_chunk_time = None
        
        # Iterate through the response to get tokens
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove the 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        break
                        
                    try:
                        data = json.loads(data_str)
                        num_chunks += 1
                        
                        # Record time of first chunk
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        
                        if num_chunks == 28:
                            end_time = time.time()
                                            
                        if 'choices' in data and len(data['choices']) > 0:
                            token_text = data['choices'][0].get('text', '')
                            token_counter += 1
                            # print(f"Token: '{token_text}'")

                            if token_text:
                               if token_text == stop_token:
                                   print(f"Stop token found: '{token_text}'")
                                   break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
        
        # Generation completed successfully
        generation_time = time.time() - st
        tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
        print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
        
        if first_chunk_time:
            ttfb = (first_chunk_time - st) * 1000
            print(f"TTFB (first chunk): {ttfb:.1f} ms")
        
        print(f"ttfb (chunk 28): {(end_time - st)*1000:.1f} ms")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

except Exception as e:
    print(f"Error: {e}")

