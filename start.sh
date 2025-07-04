export HF_HOME="/workspace/hf"
export HF_TOKEN="<your-token-here>"
#export TRANSFORMERS_OFFLINE=1
uvicorn main:app --host localhost --port 9090
