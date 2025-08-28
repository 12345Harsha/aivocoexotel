#!/usr/bin/env python3
# bridge_server.py — Exotel (8k PCM16 base64) <-> Aivoco (Socket.IO) bridge
# With detailed logging for both directions of audio flow.

import os, asyncio, json, logging, base64, audioop, time
from typing import Optional, Dict, Tuple, List

import numpy as np
import socketio
from websockets.server import serve
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

# -------------------- Env / Config --------------------
load_dotenv(override=True)

BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", 8080))

AIVOCO_URL   = os.getenv("AIVOCO_ENDPOINT", "wss://sts.aivoco.on.cloud.vispark.in")
AIVOCO_KEY   = os.getenv("AIVOCO_API_KEY", "")
VOICE_CHOICE = os.getenv("VOICE_CHOICE", "female")
SYSTEM_MSG   = os.getenv("SYSTEM_MESSAGE", "You are a helpful AI assistant.")

# Audio params
EXOTEL_RATE      = int(os.getenv("SAMPLE_RATE_EXOTEL", 8000))    # inbound from Exotel
AIVOCO_IN_RATE   = int(os.getenv("SAMPLE_RATE_AIVOCO", 16000))   # what we SEND to Aivoco
DEFAULT_AIVO_OUT = int(os.getenv("SAMPLE_RATE_OUTPUT", 24000))   # typical TTS rate if not sent
FRAME_MS_EXO_OUT = int(os.getenv("EXOTEL_FRAME_MS", 100))        # kept for reference; not used in stateless path
FRAME_MS_TO_AIVO = int(os.getenv("AIVOCO_FRAME_MS", 20))         # chunk size we emit to Aivoco (ms)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE_FRAMES = os.getenv("VERBOSE_FRAMES", "0") == "1"  # log each individual frame/chunk
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

# -------------------- Helpers --------------------
def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def resample_pcm16_bytes(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Resample raw PCM16 mono using audioop.ratecv (stateless per block)."""
    if not pcm or src_sr == dst_sr:
        return pcm
    try:
        out, _ = audioop.ratecv(pcm, 2, 1, src_sr, dst_sr, None)
        return out
    except Exception as e:
        log.exception("resample %d->%d failed: %s", src_sr, dst_sr, e)
        return b""

def max_amp_0_1(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    arr = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    return float(np.max(np.abs(arr))) / 32768.0

def chunk_bytes(raw: bytes, frame_ms: int, sr: int, bytes_per_sample=2, ch=1):
    step = int(sr * (frame_ms/1000.0)) * bytes_per_sample * ch
    if step <= 0:  # safety
        step = max(2 * ch, 320)  # ~10ms at 8k mono
    for i in range(0, len(raw), step):
        yield raw[i:i+step]

def _coerce_bytes(x) -> Optional[bytes]:
    """Accept bytes/bytearray/memoryview or list[int] as PCM bytes."""
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, list) and x and all(isinstance(v, int) and 0 <= v < 256 for v in x):
        try: return bytes(x)
        except Exception: return None
    return None

# -------------------- Per-session flow stats --------------------
class FlowStats:
    def __init__(self):
        self.t0 = time.time()
        # Exotel -> Bridge
        self.exo_rx_frames = 0
        self.exo_rx_bytes  = 0
        # Bridge -> AIVOCO (tx)
        self.aivo_tx_frames = 0
        self.aivo_tx_bytes  = 0
        # AIVOCO -> Bridge (rx)
        self.aivo_rx_chunks = 0
        self.aivo_rx_bytes  = 0
        self.aivo_rx_sr     = DEFAULT_AIVO_OUT
        # Bridge -> Exotel (tx)
        self.exo_tx_frames = 0
        self.exo_tx_bytes  = 0

    def snapshot(self) -> str:
        dt = time.time() - self.t0
        return (f"[{dt:5.1f}s] "
                f"Exotel⇢Bridge: {self.exo_rx_frames} frames, {human_bytes(self.exo_rx_bytes)} | "
                f"Bridge⇢AIVOCO: {self.aivo_tx_frames} chunks, {human_bytes(self.aivo_tx_bytes)} (->16k) | "
                f"AIVOCO⇢Bridge: {self.aivo_rx_chunks} chunks, {human_bytes(self.aivo_rx_bytes)} (@{self.aivo_rx_sr} Hz) | "
                f"Bridge⇢Exotel: {self.exo_tx_frames} frames, {human_bytes(self.exo_tx_bytes)} (@8k)")

# -------------------- Aivoco Session --------------------
class AivocoSession:
    def __init__(self, call_id: str, stats: FlowStats):
        self.call_id = call_id
        self.sio = socketio.AsyncClient(reconnection=True)
        self.ready_evt = asyncio.Event()
        self.session_ready = False

        self.stats = stats

        # Outbound audio to Exotel: queue of (sr, pcm16 bytes)
        self.out_queue: "asyncio.Queue[Tuple[int, bytes]]" = asyncio.Queue()
        self._last_rx_sr = DEFAULT_AIVO_OUT

        self._register_handlers()

    def _register_handlers(self):
        @self.sio.event
        async def connect():
            log.info("[%s] Aivoco connected → start_call", self.call_id)
            await self.sio.emit("start_call", {
                "event": "start_call",
                "auth_key": AIVOCO_KEY,
                "system_message": SYSTEM_MSG,
                "voice_choice": VOICE_CHOICE,
                "custom_functions": []
            })

        @self.sio.event
        async def disconnect():
            log.info("[%s] Aivoco disconnected", self.call_id)
            self.session_ready = False

        @self.sio.on("auth_success")
        async def auth_success(_):
            log.info("[%s] auth_success", self.call_id)

        @self.sio.on("auth_failed")
        async def auth_failed(data):
            log.error("[%s] auth_failed: %s", self.call_id, data)
            self.ready_evt.set()

        @self.sio.on("session_ready")
        async def session_ready(data):
            log.info("🚀 [%s] session_ready", self.call_id)
            self.session_ready = True
            self.ready_evt.set()

        @self.sio.on("text_response")
        async def text_response(data):
            txt = (data or {}).get("text")
            if txt:
                log.info("[%s] text_response: %s", self.call_id, txt)

        # Named audio events (often only the first chunk)
        @self.sio.on("audio_response")
        async def audio_response(data):
            await self._ingest_named("audio_response", data)

        @self.sio.on("audio_chunk")
        async def audio_chunk(data):
            await self._ingest_named("audio_chunk", data)

        # Catch-all: subsequent **binary** chunks (primary ingest)
        @self.sio.on("*")
        async def catch_all(event, data):
            if event in {"connect","disconnect","auth_success","auth_failed","session_ready","session_ended","text_response","error","message"}:
                return
            buf = _coerce_bytes(data)
            if buf:
                self.stats.aivo_rx_chunks += 1
                self.stats.aivo_rx_bytes  += len(buf)
                if VERBOSE_FRAMES:
                    log.debug("[%s] AIVOCO→Bridge (binary) %d bytes (sr=%d)", self.call_id, len(buf), self._last_rx_sr)
                await self.out_queue.put((self._last_rx_sr, buf))
                return
            if isinstance(data, dict):
                sr = None
                for k in ("sample_rate_hz","sample_rate","sr","rate"):
                    v = data.get(k)
                    if isinstance(v,int) and v>0:
                        sr = v; break
                if sr:
                    self._last_rx_sr = sr
                    self.stats.aivo_rx_sr = sr
                for k in ("audio_data","audio_chunk","chunk","audio","data"):
                    v = data.get(k)
                    if isinstance(v, str) and v:
                        try:
                            pcm = _b64d(v)
                            self.stats.aivo_rx_chunks += 1
                            self.stats.aivo_rx_bytes  += len(pcm)
                            if VERBOSE_FRAMES:
                                log.debug("[%s] AIVOCO→Bridge (b64) %d bytes (sr=%d)", self.call_id, len(pcm), self._last_rx_sr)
                            await self.out_queue.put((self._last_rx_sr, pcm))
                            break
                        except Exception as e:
                            log.debug("[%s] b64 decode failed in catch_all: %s", self.call_id, e)
                    else:
                        buf = _coerce_bytes(v)
                        if buf:
                            self.stats.aivo_rx_chunks += 1
                            self.stats.aivo_rx_bytes  += len(buf)
                            if VERBOSE_FRAMES:
                                log.debug("[%s] AIVOCO→Bridge (embedded bytes) %d bytes (sr=%d)", self.call_id, len(buf), self._last_rx_sr)
                            await self.out_queue.put((self._last_rx_sr, buf))
                            break

        @self.sio.on("error")
        async def error(data):
            log.error("[%s] aivoco error: %s", self.call_id, data)

    async def _ingest_named(self, evt: str, data):
        if not isinstance(data, dict):
            return
        for k in ("sample_rate_hz","sample_rate","sr","rate"):
            v = data.get(k)
            if isinstance(v,int) and v>0:
                self._last_rx_sr = v
                self.stats.aivo_rx_sr = v
                break
        for k in ("audio_data","audio_chunk","chunk","audio","data"):
            v = data.get(k)
            if isinstance(v, str) and v:
                try:
                    pcm = _b64d(v)
                    self.stats.aivo_rx_chunks += 1
                    self.stats.aivo_rx_bytes  += len(pcm)
                    if VERBOSE_FRAMES:
                        log.debug("[%s] AIVOCO→Bridge (%s b64) %d bytes (sr=%d)", self.call_id, evt, len(pcm), self._last_rx_sr)
                    await self.out_queue.put((self._last_rx_sr, pcm))
                    break
                except Exception as e:
                    log.debug("[%s] b64 decode failed in %s: %s", self.call_id, evt, e)
            else:
                buf = _coerce_bytes(v)
                if buf:
                    self.stats.aivo_rx_chunks += 1
                    self.stats.aivo_rx_bytes  += len(buf)
                    if VERBOSE_FRAMES:
                        log.debug("[%s] AIVOCO→Bridge (%s bytes) %d bytes (sr=%d)", self.call_id, evt, len(buf), self._last_rx_sr)
                    await self.out_queue.put((self._last_rx_sr, buf))
                    break

    async def start(self):
        if not AIVOCO_KEY:
            log.error("[%s] Missing AIVOCO_API_KEY", self.call_id); return
        try:
            await self.sio.connect(AIVOCO_URL, transports=["websocket"])
            log.info("[%s] Connected to AIVoco: %s", self.call_id, AIVOCO_URL)
        except Exception as e:
            log.exception("[%s] socketio.connect failed: %s", self.call_id, e); return
        try:
            await asyncio.wait_for(self.ready_evt.wait(), timeout=30)
        except asyncio.TimeoutError:
            log.error("[%s] session_ready timeout", self.call_id)

    async def stop(self):
        try: await self.sio.emit("stop_call")
        except Exception: pass
        if self.sio.connected:
            try: await self.sio.disconnect()
            except Exception: pass
        self.session_ready = False
        log.info("[%s] Aivoco session stopped", self.call_id)

    async def send_audio_from_exotel(self, pcm16_8k: bytes, src_sr: int = EXOTEL_RATE):
        """Exotel → resample to 16k → emit 'audio_data' in ~FRAME_MS_TO_AIVO chunks with SR metadata."""
        if not (self.session_ready and self.sio.connected):
            log.debug("[%s] drop exotel frame (session not ready)", self.call_id)
            return
        pcm16_16k = resample_pcm16_bytes(pcm16_8k, src_sr, AIVOCO_IN_RATE)
        if not pcm16_16k:
            return
        for chunk in chunk_bytes(pcm16_16k, FRAME_MS_TO_AIVO, AIVOCO_IN_RATE):
            if not chunk:
                continue
            amp = max_amp_0_1(chunk)
            has_audio = amp > 0.01
            try:
                # optional: ACK callback (will log if server acks)
                def _ack_cb(resp=None):
                    log.debug("[%s] Aivoco ACK audio_data: %s", self.call_id, str(resp)[:120])

                await self.sio.emit("audio_data", {
                    "audio_data": _b64e(chunk),
                    "has_audio": has_audio,
                    "max_amplitude": float(amp),
                    "sample_rate_hz": AIVOCO_IN_RATE
                }, callback=_ack_cb)

                self.stats.aivo_tx_frames += 1
                self.stats.aivo_tx_bytes  += len(chunk)
                if VERBOSE_FRAMES:
                    log.debug("[%s] Bridge→AIVOCO %d bytes (sr=16k, has_audio=%s, amp=%.3f)",
                              self.call_id, len(chunk), has_audio, amp)
            except Exception as e:
                log.error("[%s] emit audio_data failed: %s", self.call_id, e)
                break

# -------------------- WebSocket Bridge (Exotel side) --------------------
class Bridge:
    def __init__(self):
        self.sessions: Dict[object, AivocoSession] = {}
        self.pumps: Dict[object, asyncio.Task] = {}
        self.stats_map: Dict[object, FlowStats] = {}

    async def pump_aivoco_to_exotel(self, ws, stream_sid: str, aivoco: AivocoSession, stats: FlowStats):
        """Immediately forward each Aivoco chunk to Exotel without buffering."""
        try:
            while True:
                out_sr, pcm = await aivoco.out_queue.get()
                pcm_8k = resample_pcm16_bytes(pcm, out_sr, EXOTEL_RATE)
                if not pcm_8k:
                    continue

                # send directly to Exotel
                await ws.send(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": _b64e(pcm_8k)}
                }))

                stats.exo_tx_frames += 1
                stats.exo_tx_bytes  += len(pcm_8k)

                if VERBOSE_FRAMES:
                    log.debug("[stream:%s] Bridge→Exotel sent %d bytes (frame %d)",
                              stream_sid, len(pcm_8k), stats.exo_tx_frames)
        except Exception as e:
            log.info("[stream:%s] pump → Exotel exiting: %s", stream_sid, e)

    async def stats_reporter(self, ws):
        """Periodically print a one-line summary for this session."""
        stats = self.stats_map.get(ws)
        if not stats:
            return
        try:
            while True:
                await asyncio.sleep(1.5)
                log.info("FLOW %s", stats.snapshot())
        except asyncio.CancelledError:
            return

    async def handle(self, ws, path):
        if path not in ("/exotel", "/"):
            await ws.close(); return

        call_id = "call"
        stream_sid = None
        aivoco: Optional[AivocoSession] = None

        # per-connection stats & reporter
        stats = FlowStats()
        self.stats_map[ws] = stats
        reporter_task = asyncio.create_task(self.stats_reporter(ws))

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                ev = msg.get("event")

                if ev == "connected":
                    log.info("[Exotel] connected event received. Starting Aivoco session…")
                    aivoco = AivocoSession(call_id, stats)
                    self.sessions[ws] = aivoco
                    await aivoco.start()
                    log.info("[%s] bridge: Aivoco ready=%s", call_id, aivoco.session_ready)

                elif ev == "start":
                    stream_sid = (msg.get("start") or {}).get("streamSid", "exo-stream")
                    log.info("[Exotel] start streamSid=%s", stream_sid)
                    if aivoco and ws not in self.pumps:
                        self.pumps[ws] = asyncio.create_task(
                            self.pump_aivoco_to_exotel(ws, stream_sid, aivoco, stats)
                        )

                elif ev == "media":
                    media = (msg.get("media") or {})
                    b64 = media.get("payload")
                    if not (aivoco and b64):
                        continue
                    try:
                        pcm8k = _b64d(b64)
                    except Exception as e:
                        log.exception("[%s] base64 decode failed: %s", call_id, e)
                        continue
                    src_sr = media.get("sample_rate_hz") or EXOTEL_RATE
                    try:
                        src_sr = int(src_sr)
                    except Exception:
                        src_sr = EXOTEL_RATE

                    stats.exo_rx_frames += 1
                    stats.exo_rx_bytes  += len(pcm8k)
                    if VERBOSE_FRAMES:
                        log.debug("[stream:%s] Exotel→Bridge %d bytes (sr=%d, frame %d)",
                                  stream_sid, len(pcm8k), src_sr, stats.exo_rx_frames)

                    await aivoco.send_audio_from_exotel(pcm8k, src_sr)

                elif ev == "stop":
                    log.info("[Exotel] stop event received")
                    break

        except ConnectionClosed:
            log.info("[%s] Exotel websocket closed", call_id)
        finally:
            if ws in self.pumps:
                self.pumps[ws].cancel()
            reporter_task.cancel()
            if aivoco:
                await aivoco.stop()
            self.sessions.pop(ws, None)
            self.stats_map.pop(ws, None)
            log.info("Session closed. Final %s", stats.snapshot())

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
