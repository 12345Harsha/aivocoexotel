#!/usr/bin/env python3
# bridge_server.py (audioop-based resampling, Exotel <-> Aivoco bridge)

import os
import asyncio
import json
import logging
import base64
import audioop
from typing import Optional, Dict

import socketio  # python-socketio client
from websockets.server import serve
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

# -------------------- Load .env --------------------
load_dotenv(override=True)

# -------------------- Config --------------------
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", 8080))

AIVOCO_URL = os.getenv("AIVOCO_ENDPOINT", "wss://sts.aivoco.on.cloud.vispark.in")
AIVOCO_KEY = os.getenv("AIVOCO_API_KEY") or ""
VOICE_CHOICE = os.getenv("VOICE_CHOICE", "female")
SYSTEM_MSG = os.getenv("SYSTEM_MESSAGE", "You are a helpful AI assistant.")

# Sample rates
EXOTEL_RATE = int(os.getenv("SAMPLE_RATE_EXOTEL", 8000))   # Exotel audio rate (8k)
AIVOCO_IN   = int(os.getenv("SAMPLE_RATE_AIVOCO", 16000))  # Aivoco mic input (16k)
AIVOCO_OUT  = int(os.getenv("SAMPLE_RATE_OUTPUT", 24000))  # Aivoco speaker output (24k)
FRAME_MS    = int(os.getenv("EXOTEL_FRAME_MS", 100))

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")

# -------------------- Audio utils --------------------
def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def resample_pcm16_bytes_audioop(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
    if not pcm_bytes or src_sr == dst_sr:
        return pcm_bytes
    try:
        converted, _ = audioop.ratecv(pcm_bytes, 2, 1, src_sr, dst_sr, None)
    except Exception as e:
        log.exception("resample failed (%d->%d): %s", src_sr, dst_sr, e)
        return b""
    return converted

def resample_pcm16_bytes(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
    return resample_pcm16_bytes_audioop(pcm_bytes, src_sr, dst_sr)

# -------------------- Aivoco client per session --------------------
class AivocoSession:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.sio = socketio.AsyncClient(reconnection=True)
        self.ready_evt = asyncio.Event()
        self.session_active = False
        self.out_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._register_handlers()

    def _register_handlers(self):
        @self.sio.event
        async def connect():
            log.info("[%s] (Aivoco) socket connected", self.call_id)
            payload = {
                "event": "start_call",
                "auth_key": AIVOCO_KEY,
                "system_message": SYSTEM_MSG,
                "voice_choice": VOICE_CHOICE,
                "custom_functions": []
            }
            await self.sio.emit("start_call", payload)

        @self.sio.event
        async def disconnect():
            log.info("[%s] (Aivoco) socket disconnected", self.call_id)
            self.session_active = False

        @self.sio.on("auth_success")
        async def _auth_success(data):
            log.info("[%s] ✅ auth_success: %s", self.call_id, data)

        @self.sio.on("auth_failed")
        async def _auth_failed(data):
            log.error("[%s] ❌ auth_failed: %s", self.call_id, data)
            self.ready_evt.set()

        @self.sio.on("session_ready")
        async def _session_ready(data):
            log.info("🚀 [%s] Aivoco session is ready: %s", self.call_id, data)
            self.session_active = True
            self.ready_evt.set()

        @self.sio.on("session_ended")
        async def _session_ended(data):
            log.info("[%s] 📴 Aivoco session ended: %s", self.call_id, data)
            self.session_active = False

        # -------------------- Old audio handlers (optional) --------------------
        @self.sio.on("audio_response")
        async def _audio_response_file(data):
            log.info("✅ [%s] [Aivoco] audio_response event received", self.call_id)
            await self._handle_incoming_audio_event("audio_response", data)

        @self.sio.on("audio_chunk")
        async def _audio_chunk(data):
            log.info("[%s] 🎶 audio_chunk: received keys=%s", self.call_id, list(data.keys() if isinstance(data, dict) else []))
            await self._handle_incoming_audio_event("audio_chunk", data)

        # -------------------- Catch-all handler for all audio --------------------
        @self.sio.on("*")
        async def catch_all(event, data=None):
            # Log everything
            log.info("[%s] 📨 (Aivoco raw event) %s | data keys=%s | data preview=%s",
                     self.call_id,
                     event,
                     list(data.keys()) if isinstance(data, dict) else type(data),
                     str(data)[:200])
            # Handle audio in any event
            audio_keys = ("audio_response", "audio_chunk", "audio_data", "chunk", "audio")
            if isinstance(data, dict):
                for k in audio_keys:
                    if k in data and data[k]:
                        try:
                            pcm = _b64d(data[k])
                            log.info("[%s] 🔊 catch_all decoded audio for key=%s: %d bytes",
                                     self.call_id, k, len(pcm))
                            await self.out_queue.put(pcm)
                        except Exception as e:
                            log.error("[%s] base64 decode failed for key=%s: %s", self.call_id, k, e)

        @self.sio.on("text_response")
        async def _text_response(data):
            txt = (data or {}).get("text")
            if txt:
                log.info("[%s] 💬 text_response: %s", self.call_id, txt)

        @self.sio.on("error")
        async def _error(data):
            log.error("[%s] ⚠️ Aivoco error: %s", self.call_id, data)

    async def _handle_incoming_audio_event(self, event_name: str, data):
        if not data:
            log.warning("[%s] %s event had no data", self.call_id, event_name)
            return

        candidates = []
        if isinstance(data, dict):
            for k in ("audio_data", "audio_chunk", "chunk", "audio", "data"):
                if k in data and data[k]:
                    candidates.append((k, data[k]))
        else:
            candidates.append(("raw", data))

        if not candidates:
            log.warning("[%s] %s event had no valid audio key", self.call_id, event_name)
            return

        key, b64 = candidates[0]
        try:
            pcm = _b64d(b64)
            log.info("[%s] 🔊 %s.%s decoded: %d base64 chars -> %d PCM bytes",
                     self.call_id, event_name, key, len(b64), len(pcm))
        except Exception as e:
            log.error("[%s] %s: base64 decode failed: %s", self.call_id, event_name, e)
            return

        await self.out_queue.put(pcm)

    async def start(self):
        if not AIVOCO_KEY:
            log.error("[%s] No AIVOCO_KEY provided!", self.call_id)
            return
        log.info("[%s] Connecting to Aivoco at %s", self.call_id, AIVOCO_URL)
        url = f"{AIVOCO_URL}?auth_key={AIVOCO_KEY}"
        headers = {"Authorization": f"Bearer {AIVOCO_KEY}"}
        await self.sio.connect(url, transports=["websocket"], headers=headers)
        try:
            await asyncio.wait_for(self.ready_evt.wait(), timeout=30)
        except asyncio.TimeoutError:
            log.warning("[%s] ⚠️ Aivoco session_ready timeout", self.call_id)

    async def stop(self):
        try:
            await self.sio.emit("stop_call")
        except Exception:
            pass
        if self.sio.connected:
            await self.sio.disconnect()
        self.session_active = False

    async def send_audio_from_exotel(self, pcm16_8k: bytes):
        if not self.session_active or not self.sio.connected:
            return
        pcm16_16k = resample_pcm16_bytes(pcm16_8k, EXOTEL_RATE, AIVOCO_IN)
        if not pcm16_16k:
            return
        b64 = _b64e(pcm16_16k)
        log.info("[%s] ↑ forwarding %d PCM bytes (%d b64 chars) from Exotel to Aivoco",
                 self.call_id, len(pcm16_16k), len(b64))
        await self.sio.emit("audio_data", {"audio_data": b64})

# -------------------- Exotel handler --------------------
class Bridge:
    def __init__(self):
        self.sessions: Dict[object, AivocoSession] = {}
        self.pump_tasks: Dict[object, asyncio.Task] = {}

    async def pump_aivoco_to_exotel(self, ws, stream_sid: str, aivoco: AivocoSession):
        bytes_per_100ms = int(EXOTEL_RATE * (FRAME_MS / 1000.0)) * 2
        buf = bytearray()
        try:
            while True:
                try:
                    pcm_from_aivoco = await asyncio.wait_for(aivoco.out_queue.get(), timeout=0.1)
                    log.info("[%s] ↓ got %d PCM bytes from Aivoco", stream_sid, len(pcm_from_aivoco))
                    pcm_8k = resample_pcm16_bytes(pcm_from_aivoco, AIVOCO_OUT, EXOTEL_RATE)
                    if pcm_8k:
                        buf.extend(pcm_8k)
                except asyncio.TimeoutError:
                    pass

                while len(buf) >= bytes_per_100ms:
                    chunk = bytes(buf[:bytes_per_100ms])
                    del buf[:bytes_per_100ms]
                    await ws.send(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": _b64e(chunk)}
                    }))
                    log.info("[%s] → sent %d bytes chunk to Exotel", stream_sid, len(chunk))
                await asyncio.sleep(0.01)
        except Exception as e:
            log.info("[%s] pump exiting: %s", stream_sid, e)

    async def handle(self, ws, path):
        if path not in ("/exotel", "/"):
            await ws.close()
            return
        call_id = "call"
        stream_sid = None
        aivoco: Optional[AivocoSession] = None
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                ev = msg.get("event")
                if ev == "connected":
                    aivoco = AivocoSession(call_id)
                    self.sessions[ws] = aivoco
                    await aivoco.start()
                    log.info("[%s] bridge ready", call_id)
                elif ev == "start":
                    stream_sid = msg.get("start", {}).get("streamSid")
                    if aivoco and ws not in self.pump_tasks:
                        self.pump_tasks[ws] = asyncio.create_task(
                            self.pump_aivoco_to_exotel(ws, stream_sid, aivoco)
                        )
                elif ev == "media":
                    payload_b64 = (msg.get("media") or {}).get("payload")
                    if not payload_b64 or not aivoco:
                        continue
                    try:
                        pcm16_8k = _b64d(payload_b64)
                        log.info("[%s] ← received %d PCM bytes from Exotel", call_id, len(pcm16_8k))
                    except Exception as e:
                        log.error("[%s] base64 decode failed: %s", call_id, e)
                        continue
                    await aivoco.send_audio_from_exotel(pcm16_8k)
                elif ev == "stop":
                    break
        except ConnectionClosed:
            log.info("[%s] Exotel websocket closed", call_id)
        finally:
            if ws in self.pump_tasks:
                self.pump_tasks[ws].cancel()
            if aivoco:
                await aivoco.stop()
            self.sessions.pop(ws, None)

# -------------------- Entry --------------------
async def main():
    bridge = Bridge()
    async with serve(
        bridge.handle,
        BRIDGE_HOST,
        BRIDGE_PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=10_000_000,
    ):
        log.info("Bridge listening on ws://%s:%d/exotel", BRIDGE_HOST, BRIDGE_PORT)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
