from src.logger import setup_logger
setup_logger()

import time
import os
from src.trt_engine import OrpheusModelTRT
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import json
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings
import asyncio

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    input: str = "Hey there, looks like you forgot to provide a prompt!"
    voice: str = "tara"


class TTSStreamRequest(BaseModel):
    input: str
    voice: str = "tara"
    continue_: bool = Field(True, alias="continue")
    segment_id: str


class VoiceDetail(BaseModel):
    name: str
    description: str
    language: str
    gender: str
    accent: str
    preview_url: Optional[str] = None


class VoicesResponse(BaseModel):
    voices: List[VoiceDetail]
    default: str
    count: int
    
engine: OrpheusModelTRT = None
VOICE_DETAILS: List[VoiceDetail] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS engine on application startup."""
    global engine, VOICE_DETAILS
    engine = OrpheusModelTRT()

    # Dynamically generate voice details from the loaded engine
    VOICE_DETAILS = [
        VoiceDetail(
            name=voice,
            description=f"A standard {voice} voice.",
            language="en",
            gender="unknown",
            accent="american"
        ) for voice in engine.available_voices
    ]
    yield
    # Clean up the model and other resources if needed

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    start_time = time.perf_counter()

    async def generate_audio_stream():
        first_chunk = True
        try:
            audio_generator = engine.generate_speech_async(
                prompt=data.input,
                voice=data.voice,
            )

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                yield chunk
        except Exception:
            logger.exception("An error occurred during audio generation")


    return StreamingResponse(generate_audio_stream(), media_type='audio/pcm')


# @app.websocket("/v1/audio/speech/stream/ws")
# async def tts_stream_ws(websocket: WebSocket):
#     await websocket.accept()

#     request_queue = asyncio.Queue()

#     async def receive_task():
#         """Receives messages from the client and puts them in a queue."""
#         try:
#             while True:
#                 message = await websocket.receive_text()
#                 try:
#                     data = TTSStreamRequest.parse_raw(message)
#                     await request_queue.put(data)
#                     if not data.continue_:
#                         await request_queue.put(None)  # Sentinel to signal end
#                         break
#                 except Exception as e:
#                     logger.error(f"Error parsing incoming message: {e}")
#                     await websocket.send_json({"error": "invalid message format"})
#                     await request_queue.put(None)  # End processing on error
#                     break
#         except WebSocketDisconnect:
#             logger.info("Client disconnected during receive.")
#             await request_queue.put(None)  # Ensure processor task also exits
#         except Exception as e:
#             logger.error(f"An unexpected error occurred in receive_task: {e}")
#             await request_queue.put(None)


#     async def process_task():
#         """Processes requests from the queue and generates audio."""
#         is_final_packet = False
#         while True:
#             data = await request_queue.get()
#             if data is None:  # Sentinel received
#                 is_final_packet = True
#                 break

#             try:
#                 await websocket.send_json({"type": "start", "segment_id": data.segment_id})

#                 if data.input and data.input.strip():
#                     logger.info(f"Generating audio for input: '{data.input.strip()}'")
#                     audio_generator = engine.generate_speech_async(
#                         prompt=data.input.strip(),
#                         voice=data.voice,
#                     )

#                     async for chunk in audio_generator:
#                         await websocket.send_bytes(chunk)
#                 else:
#                     logger.info("Empty or whitespace-only input received, skipping audio generation.")

#                 await websocket.send_json({"type": "end", "segment_id": data.segment_id})
#             except Exception as e:
#                 logger.exception("Error during audio generation in process task.")
#                 await websocket.send_json({"error": str(e)})
#                 break  # Stop processing on error

#         if is_final_packet and websocket.client_state == WebSocketState.CONNECTED:
#             logger.info("Final packet processed, sending done message.")
#             await websocket.send_json({"done": True})

#     # Run both tasks concurrently
#     processing_task_handle = asyncio.create_task(process_task())
#     receiving_task_handle = asyncio.create_task(receive_task())

#     done, pending = await asyncio.wait(
#         [processing_task_handle, receiving_task_handle],
#         return_when=asyncio.FIRST_COMPLETED,
#     )

#     for task in pending:
#         task.cancel()

#     logger.info("Closing websocket connection.")
#     if websocket.client_state == WebSocketState.CONNECTED:
#         try:
#             await websocket.close()
#         except Exception:
#             pass # Ignore errors on close, connection might already be gone


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            try:
                data = TTSStreamRequest.parse_raw(message)
            except Exception as e:
                logger.error(f"Error parsing incoming websocket message: {e}")
                await websocket.send_json({"error": "invalid message format"})
                break

            try:
                await websocket.send_json({"type": "start", "segment_id": data.segment_id})

                if data.input and data.input.strip():
                    logger.info(f"Generating audio for input: '{data.input.strip()}'")
                    audio_generator = engine.generate_speech_async(
                        prompt=data.input.strip(),
                        voice=data.voice,
                    )

                    async for chunk in audio_generator:
                        await websocket.send_bytes(chunk)
                else:
                    logger.info("Empty or whitespace-only input received, skipping audio generation.")
                
                await websocket.send_json({"type": "end", "segment_id": data.segment_id})

                if not data.continue_:
                    await websocket.send_json({"done": True})
                    break

            except Exception as e:
                logger.exception("An error occurred during audio generation in websocket.")
                await websocket.send_json({"error": str(e), "done": True})
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.get("/api/voices", response_model=VoicesResponse)
async def get_voices():
    """Get available voices with detailed information."""
    default_voice = engine.available_voices[0] if engine and engine.available_voices else "tara"
    return {
        "voices": VOICE_DETAILS,
        "default": default_voice,
        "count": len(VOICE_DETAILS)
    }
