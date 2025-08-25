#!/usr/bin/env python3
# mic_client.py
import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import time

BRIDGE_URL = "ws://127.0.0.1:8080/exotel"
STREAM_SID = "mic-test"

# ----- helpers -----
def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def play_pcm16_8k(pcm: bytes):
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(x, 8000, blocking=False)

# ----- client -----
async def client():
    async with websockets.connect(BRIDGE_URL) as ws:
        print("✅ Connected to bridge")

        # 1. Send "connected" event like Exotel
        await ws.send(json.dumps({
            "event": "connected",
            "start": {"streamSid": STREAM_SID, "call_sid": "local-call-123"}
        }))

        # 2. Send "start" event
        await ws.send(json.dumps({
            "event": "start",
            "start": {"streamSid": STREAM_SID}
        }))

        # 3. Sender: capture mic, detect silence, and stream
        async def sender():
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            silence_threshold = 200  # adjust if too sensitive
            silence_duration = 1.0   # seconds
            last_speech_time = time.time()

            def callback(indata, frames, time_info, status):
                if status:
                    print("⚠️ Mic status:", status)
                loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

            with sd.RawInputStream(samplerate=8000, channels=1, dtype="int16", callback=callback):
                print("🎤 Speak into the mic... (auto silence detection ON)")
                while True:
                    chunk = await q.get()
                    await ws.send(json.dumps({
                        "event": "media",
                        "streamSid": STREAM_SID,
                        "media": {"payload": b64e(chunk)}
                    }))
                    print(f"📤 Sent {len(chunk)} bytes of audio")

                    # Silence detection
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                    if rms > silence_threshold:
                        last_speech_time = time.time()
                    else:
                        if time.time() - last_speech_time > silence_duration:
                            print("🤫 Silence detected → sending end_of_utterance")
                            await ws.send(json.dumps({
                                "event": "mark",
                                "streamSid": STREAM_SID,
                                "mark": {"name": "end_of_utterance"}
                            }))
                            last_speech_time = time.time() + 999  # prevent spam until speech resumes

        # 4. Receiver: play audio from Aivoco
        async def receiver():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                print("⬅️ Got message from bridge:", msg)  # 👀 Debug line

                if msg.get("event") == "media":
                    b64 = msg.get("media", {}).get("payload")
                    if b64:
                        pcm = b64d(b64)
                        print(f"🔊 Playing {len(pcm)} bytes of audio from bridge")
                        play_pcm16_8k(pcm)

        await asyncio.gather(sender(), receiver())

if __name__ == "__main__":
    asyncio.run(client())
