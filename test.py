from openai import OpenAI
import time

model_name = "dst19/jess-voice-merged"

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="tensorrt_llm",
)

print(f"starting test")

try:
    # prompt = _format_prompt("Hey What's up? How are you doing? So happy to see you again!")
    prompt = "<custom_token_3><|begin_of_text|>tara: Hey What's up? How are you doing? So happy to see you again!<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    print(f"Prompt: {prompt}")
    st = time.time()
    response = client.completions.create(
        model="dst19/jess-voice-merged",
        prompt=prompt,
        max_tokens=1024,
        stream=True,
        temperature=0.4,
        # stop_token_ids=128258,
        # repetition_penalty=1.1,
        top_p=0.9,
    )

    num_chunks = 0
    print("Streaming response:")
    for chunk in response:
        num_chunks += 1
        if chunk.choices and chunk.choices[0].text:
            print(f"Text: '{chunk.choices[0].text}'")
        if chunk.choices and chunk.choices[0].finish_reason:
            print(f"Finish reason: {chunk.choices[0].finish_reason}")
        if num_chunks == 28:
            end_time = time.time()
    print()  # Add a newline at the end

    print(f"ttfb: {(end_time - st)*1000} ms")

except Exception as e:
    print(f"Error: {e}")

