import asyncio
import torch
import os
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from transformers import AutoTokenizer
from .decoder import tokens_decoder
import logging

logger = logging.getLogger(__name__)

class OrpheusModelTRT:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "canopylabs/orpheus-3b-0.1-ft")
        
        # Load available voices from environment variables
        voices_str = os.getenv("AVAILABLE_VOICES", "tara,zoe,jess,zac,leo,mia,julia,leah")
        self.available_voices = [voice.strip() for voice in voices_str.split(',')]

        # Load sampling parameters from environment variables
        self.temperature = float(os.getenv("TRT_TEMPERATURE", 0.4))
        self.top_p = float(os.getenv("TRT_TOP_P", 0.9))
        self.max_tokens = int(os.getenv("TRT_MAX_TOKENS", 1024))
        self.repetition_penalty = float(os.getenv("TRT_REPETITION_PENALTY", 1.1))
        stop_token_ids_str = os.getenv("TRT_STOP_TOKEN_IDS", "128258")
        self.stop_token_ids = [int(token_id.strip()) for token_id in stop_token_ids_str.split(',')]

        # Load engine parameters from environment variables
        self.dtype = os.getenv("TRT_DTYPE", "bfloat16")
        self.max_input_len = int(os.getenv("TRT_MAX_INPUT_LEN", 1024))
        self.max_batch_size = int(os.getenv("TRT_MAX_BATCH_SIZE", 4))
        self.max_seq_len = int(os.getenv("TRT_MAX_SEQ_LEN", 8192))
        self.enable_chunked_prefill = os.getenv("TRT_ENABLE_CHUNKED_PREFILL", "True").lower() == 'true'
        self.max_beam_width = int(os.getenv("TRT_MAX_BEAM_WIDTH", 1))
        self.max_num_tokens = int(os.getenv("TRT_MAX_NUM_TOKENS", 8192))
        self.free_gpu_memory_fraction = float(os.getenv("TRT_FREE_GPU_MEMORY_FRACTION", 0.6))
        self.max_kv_cache_tokens = int(os.getenv("TRT_KV_CACHE_MAX_TOKENS", 512))
        logger.info("--- TRT-LLM Sampling Parameters Loaded ---")
        logger.info(f"Model Name: {self.model_name}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Top P: {self.top_p}")
        logger.info(f"Max Tokens: {self.max_tokens}")
        logger.info(f"Repetition Penalty: {self.repetition_penalty}")
        logger.info(f"Stop Token IDs: {self.stop_token_ids}")
        logger.info(f"Model dtype: {self.dtype}")
        logger.info(f"Max Input Len: {self.max_input_len}")
        logger.info(f"Max Batch Size: {self.max_batch_size}")
        logger.info(f"Max Seq Len: {self.max_seq_len}")
        logger.info(f"Enable Chunked Prefill: {self.enable_chunked_prefill}")
        logger.info(f"Max Beam Width: {self.max_beam_width}")
        logger.info(f"Max Num Tokens: {self.max_num_tokens}")
        logger.info(f"Free GPU Memory Fraction: {self.free_gpu_memory_fraction}")
        logger.info(f"Max KV Cache Tokens: {self.max_kv_cache_tokens}")
        logger.info("-----------------------------------------")

        self.engine = self._setup_engine()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def _setup_engine(self):
        return LLM(model=self.model_name,
                    dtype=self.dtype,
                    max_input_len=self.max_input_len,
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.max_seq_len,
                    enable_chunked_prefill=self.enable_chunked_prefill,
                    trust_remote_code=True,
                    max_beam_width=self.max_beam_width,
                    max_num_tokens=self.max_num_tokens,
                    kv_cache_config=KvCacheConfig(
                        free_gpu_memory_fraction=self.free_gpu_memory_fraction,
                        # max_tokens=self.max_kv_cache_tokens,
                        # max_attention_window=[32]
                    )
        )
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt, voice="tara"):
        # This formatting is specific to the Orpheus model
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string

    async def generate_tokens_async(self, prompt, voice):
        self.validate_voice(voice)
        prompt_string = self._format_prompt(prompt, voice)
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            repetition_penalty=self.repetition_penalty,
        )

        async for output in self.engine.generate_async(
            prompt_string,
            sampling_params=sampling_params,
            streaming=True
        ):
            yield output.outputs[0].text
    
    async def generate_speech_async(self, prompt, voice):
        """
        Generates speech by first generating tokens and then decoding them into audio asynchronously.
        This function is a simple wrapper to stay consistent with the original engine class.
        """
        token_generator = self.generate_tokens_async(prompt, voice)
        async for audio_chunk in tokens_decoder(token_generator):
            yield audio_chunk 