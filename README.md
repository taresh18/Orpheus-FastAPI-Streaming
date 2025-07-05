# Orpheus TTS

- RTX-4090, cuda12.8
- 200ms ttfb on fp16 using vllm
- 160 ms ttfb on fp16 using trt-llm

## installation
- `sudo apt-get -y install libopenmpi-dev`
- `conda create -n trt python=3.10 && conda activate trt` or use a virtual env with python3.10
- `pip install -r requirements.txt`
- update start.sh with your hf-token
- `bash start.sh`

### Note: 
The hf repo of model `canopylabs/orpheus-3b-0.1-ft` has optimiser / fsdp files which are not required for inference. However trt-llm download all the files so ensure you have enough storage (~100GB) available in the pod / machine. Or you can stop the process midway (after tokeniser and model safetensors are downloaded) and cleanup these extra files and set `export TRANSFORMERS_OFFLINE=1` in start.sh and start teh process again
