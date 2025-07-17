#!/bin/bash
set -e

export HF_HOME="/workspace/hf"
# export HF_TOKEN="<your-token-here>"
export TRANSFORMERS_OFFLINE=0

trtllm-serve serve dst19/jess-voice-merged --host 0.0.0.0 --port 9090 --max_batch_size 4 --max_num_tokens 8192 --max_seq_len 2048 --max_beam_width 1 --kv_cache_free_gpu_memory_fraction 0.8 --trust_remote_code