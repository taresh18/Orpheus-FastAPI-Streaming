import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Integration mode - determines whether to use integrated vLLM or external API
USE_INTEGRATED_VLLM = os.environ.get("USE_INTEGRATED_VLLM", "false").lower() == "true"

# Generation parameters from environment variables
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
REPETITION_PENALTY = 1.1

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "models/Orpheus-3b-FT-Q4_K_M.gguf")
DTYPE = os.environ.get("DTYPE", "auto")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.6"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "512"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "4"))

# Audio settings
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "24000"))
BITS_PER_SAMPLE = 16
CHANNELS = 1

# Voice configuration
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]

# Token patterns and prefixes
CUSTOM_TOKEN_PREFIX = "<custom_token_"
CUSTOM_TOKEN_PATTERN = r'<custom_token_\d+>'

# Special tokens for prompt formatting
AUDIO_START_TOKEN = "<|audio|>"
AUDIO_END_TOKEN = "<|eot_id|>"

# Server configuration
INFER_SERVER_URL = os.environ.get("INFER_SERVER_URL")
INFER_SERVER_TIMEOUT = int(os.environ.get("INFER_SERVER_TIMEOUT", "120"))

# API request settings
HEADERS = {"Content-Type": "application/json"}
MAX_RETRIES = 3

# Validation warning for external API mode
if not INFER_SERVER_URL and not USE_INTEGRATED_VLLM:
    print("WARNING: INFER_SERVER_URL not set. API calls will fail until configured.")

# Processing settings
NUM_WORKERS = 8
APPROX_TOKEN_PER_CHARACTER = 15

# Batch processing settings
MAX_BATCH_CHARS = 1000
MIN_SENTENCE_CHARS = 20

# Audio processing settings
CROSSFADE_MS = 50

# CUDA settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None

# Cache settings
MAX_CACHE_SIZE = 10000

# FastAPI server settings
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5005"))

# Performance monitoring settings
PERFORMANCE_REPORT_INTERVAL = 2.0  # seconds
AUDIO_CHUNK_DURATION = 0.085  # seconds per chunk estimate 

MIN_TOKENS_FIRST_CHUNK = 7 # generate first audio chunk after 7 tokens
MIN_TOKENS_SUBSEQUENT = 28 # generate subsequent with a 28 token sliding window