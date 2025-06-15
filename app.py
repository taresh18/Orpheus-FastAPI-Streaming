import os
import time
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import io

# Load environment variables from .env file
load_dotenv(override=True)

# Import from refactored TTS engine
from src.engines import get_tts_service, initialize_tts_service, stream_speech_from_api
from src.consts import *
from src.utils import generate_wav_header
from src.logger import get_logger

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Orpheus TTS Server...")
    
    # Configuration summary
    if USE_INTEGRATED_VLLM:
        logger.info("Using integrated vLLM engine")
    else:
        api_url = os.environ.get("INFER_SERVER_URL")
        logger.info(f"Using external API at: {api_url}")
    
    try:
        await initialize_tts_service()
    except Exception as e:
        logger.error(f"Failed to initialize TTS service: {e}")
        if USE_INTEGRATED_VLLM:
            raise RuntimeError(f"TTS service initialization failed: {e}")
    
    yield
    # shutdown
    logger.info("Shutting down Orpheus TTS Server...")

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FastAPI-Streaming",
    description="High-performance Text-to-Speech server using Orpheus-FastAPI-Streaming",
    version="2.0.0",
    lifespan=lifespan
)

# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class StreamingSpeechRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


async def _create_audio_stream_generator(text: str, voice: str):
    """
    Create an optimized audio stream generator for real-time TTS.
    Uses the refactored TTS service for ultra-low latency streaming.
    """
    # Validate inputs
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    if voice not in AVAILABLE_VOICES:
        logger.warning(f" Invalid voice '{voice}', using default '{DEFAULT_VOICE}'")
        voice = DEFAULT_VOICE
    
    logger.info(f"Starting streaming generation for: '{text[:50]}{'...' if len(text) > 50 else ''}' (voice: {voice})")
    
    # Send WAV header first for immediate playback
    wav_header = generate_wav_header(SAMPLE_RATE, 16, 1)
    yield wav_header
    
    # Stream audio chunks as they're generated
    try:
        chunk_count = 0
        start_time = time.time()
        
        async for audio_chunk in stream_speech_from_api(text, voice):
            if audio_chunk:
                chunk_count += 1
                yield audio_chunk
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Streaming complete: {chunk_count} chunks in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        # In streaming mode, we can't send HTTP error codes after headers are sent
        # The client will detect the incomplete stream
        raise


@app.post("/v1/audio/speech/stream")
async def create_speech_stream_v1(request: StreamingSpeechRequest):
    """
    OpenAI-compatible streaming endpoint.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    return StreamingResponse(
        _create_audio_stream_generator(request.input, request.voice),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering for true streaming
        }
    )


@app.post("/api/tts/stream")
async def create_speech_stream_api(request: Request):
    """
    streaming endpoint for real-time TTS.
    """
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)
    
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' parameter")
    
    return StreamingResponse(
        _create_audio_stream_generator(text, voice),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/v1/audio/speech")
async def create_speech_v1(request: SpeechRequest):
    """OpenAI-compatible speech generation endpoint."""
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    try:
        # Use the TTS service for generation
        tts_service = get_tts_service()
        audio_chunks = tts_service.generate_speech(
            prompt=request.input,
            voice=request.voice
        )
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Combine all chunks
        audio_data = b''.join(audio_chunks)
        
        # Create WAV file in memory
        wav_header = generate_wav_header(SAMPLE_RATE, 16, 1)
        full_audio = wav_header + audio_data
        
        return StreamingResponse(
            io.BytesIO(full_audio),
            media_type="audio/wav",
            headers={"Content-Length": str(len(full_audio))}
        )
        
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.post("/api/tts")
async def create_speech_api(request: Request):
    """Custom TTS endpoint with file output support."""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)
    output_file = data.get("output_file")
    
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' parameter")
    
    try:
        # Use the TTS service for generation
        tts_service = get_tts_service()
        audio_chunks = tts_service.generate_speech(
            prompt=text,
            voice=voice,
            output_file=output_file
        )
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        response_data = {
            "success": True,
            "message": f"Generated {len(audio_chunks)} audio chunks",
            "voice": voice,
            "text_length": len(text)
        }
        
        if output_file:
            response_data["output_file"] = output_file
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.get("/api/voices")
async def get_voices():
    """Get available voices."""
    return {
        "voices": AVAILABLE_VOICES,
        "default": DEFAULT_VOICE,
        "count": len(AVAILABLE_VOICES)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "5005"))
    
    # Start with reload enabled
    uvicorn.run("app:app", host=host, port=port, reload=True)
