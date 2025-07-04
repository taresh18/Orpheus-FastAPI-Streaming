# Orpheus TTS

- RTX-4090, cuda12.8
- 200ms ttfb on fp16 using vllm
- 160 ms ttfb on fp16 using trt-llm

## installation
- `sudo apt-get -y install libopenmpi-dev`
- `pip install -r requirements.txt` # use virtual env with python3.10
- update start.sh with your hf-token
- `bash start.sh`
