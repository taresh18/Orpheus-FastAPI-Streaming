# Orpheus TTS

- RTX-4090, cuda12.8
- 200ms ttfb on fp16 using vllm
- 160 ms ttfb on fp16 using trt-llm

## installation
- `sudo apt-get -y install libopenmpi-dev`
- `conda create -n trt python=3.10` or use a virtual env with python3.10
- `pip install -r requirements.txt`
- update start.sh with your hf-token
- `bash start.sh`
