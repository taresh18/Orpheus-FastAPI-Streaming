#!/bin/bash
set -e

export HF_HOME="/workspace/hf"
# export HF_TOKEN="<your-token-here>"
export TRANSFORMERS_OFFLINE=0
uvicorn main:app --host 0.0.0.0 --port 9090
