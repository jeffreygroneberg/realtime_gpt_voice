"""
Azure OpenAI Realtime API — Voice Console Chat with Token Usage Logging & Tracing
==================================================================================
A speech-in / speech-out console app that connects to the GPT-4o Realtime API
via WebSocket.  Captures microphone audio, streams it to Azure, plays back the
assistant's voice response, and logs detailed token usage on the console.

Tracing is enabled via OpenTelemetry and Azure Monitor / Application Insights,
so all spans are visible in the AI Foundry portal under Tracing.

Requires:
    pip install openai python-dotenv websockets numpy sounddevice \
        azure-ai-projects azure-monitor-opentelemetry azure-identity

Environment variables (loaded from .env):
    AZURE_OPENAI_ENDPOINT                  – e.g. https://ais-realtimegpt.openai.azure.com/
    AZURE_OPENAI_API_KEY                   – your API key
    AZURE_OPENAI_DEPLOYMENT                – e.g. gpt-4o-realtime-preview
    AZURE_OPENAI_API_VERSION               – e.g. 2025-04-01-preview
    APPLICATIONINSIGHTS_CONNECTION_STRING   – App Insights connection string
"""

import asyncio
import base64
import os
import sys

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# OpenTelemetry tracing
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

load_dotenv()

# ── Tracing setup ────────────────────────────────────────────────────────────
# Enable content capture so prompts/responses are visible in traces
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

_appinsights_cs = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if _appinsights_cs:
    configure_azure_monitor(connection_string=_appinsights_cs)
    print("[tracing] Azure Monitor configured — traces will appear in App Insights / AI Foundry.")
else:
    print("[tracing] APPLICATIONINSIGHTS_CONNECTION_STRING not set — tracing disabled.")

tracer = trace.get_tracer(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

# Audio settings – the Realtime API uses PCM16, 24 kHz, mono
SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION_MS = 100  # send audio every 100 ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# ── Session-level token counters ─────────────────────────────────────────────
session_tokens = {
    "total": 0,
    "input": 0,
    "output": 0,
    "input_text": 0,
    "input_audio": 0,
    "input_cached": 0,
    "output_text": 0,
    "output_audio": 0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def print_banner() -> None:
    print("\n" + "=" * 64)
    print("  Azure OpenAI Realtime Voice Chat")
    print(f"  Endpoint   : {ENDPOINT}")
    print(f"  Deployment : {DEPLOYMENT}")
    print(f"  API Version: {API_VERSION}")
    print("=" * 64)
    print("  Speak into your microphone — the assistant will reply by voice.")
    print("  Transcripts and token usage are logged to the console.")
    print("  Press Ctrl+C to quit.")
    print("=" * 64 + "\n")


def log_token_usage(usage) -> None:
    """Pretty-print token usage from a response.done event."""
    if not usage:
        return

    total = getattr(usage, "total_tokens", 0) or 0
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0

    inp_details = getattr(usage, "input_token_details", None)
    out_details = getattr(usage, "output_token_details", None)

    inp_text = (getattr(inp_details, "text_tokens", 0) or 0) if inp_details else 0
    inp_audio = (getattr(inp_details, "audio_tokens", 0) or 0) if inp_details else 0
    inp_cached = (getattr(inp_details, "cached_tokens", 0) or 0) if inp_details else 0
    out_text = (getattr(out_details, "text_tokens", 0) or 0) if out_details else 0
    out_audio = (getattr(out_details, "audio_tokens", 0) or 0) if out_details else 0

    # Update session totals
    session_tokens["total"] += total
    session_tokens["input"] += inp
    session_tokens["output"] += out
    session_tokens["input_text"] += inp_text
    session_tokens["input_audio"] += inp_audio
    session_tokens["input_cached"] += inp_cached
    session_tokens["output_text"] += out_text
    session_tokens["output_audio"] += out_audio

    # Per-response
    print("\n┌─── Token Usage (this response) ───────────────────────────┐")
    print(f"│  Total tokens     : {total:>8}")
    print(f"│  Input tokens     : {inp:>8}  (text: {inp_text}, audio: {inp_audio}, cached: {inp_cached})")
    print(f"│  Output tokens    : {out:>8}  (text: {out_text}, audio: {out_audio})")
    print("├─── Session Totals ────────────────────────────────────────┤")
    print(f"│  Total tokens     : {session_tokens['total']:>8}")
    print(f"│  Input tokens     : {session_tokens['input']:>8}  (text: {session_tokens['input_text']}, audio: {session_tokens['input_audio']}, cached: {session_tokens['input_cached']})")
    print(f"│  Output tokens    : {session_tokens['output']:>8}  (text: {session_tokens['output_text']}, audio: {session_tokens['output_audio']})")
    print("└───────────────────────────────────────────────────────────┘\n")


# ── Audio playback ──────────────────────────────────────────────────────────
class AudioPlayer:
    """Accumulates PCM16 audio chunks and plays them back without blocking the
    event loop.  Playback runs in a thread executor so async tasks keep working."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def add_chunk(self, b64_audio: str) -> None:
        self._chunks.append(base64.b64decode(b64_audio))

    def _play_sync(self) -> None:
        """Blocking playback — called inside an executor thread."""
        if not self._chunks:
            return
        raw = b"".join(self._chunks)
        audio = np.frombuffer(raw, dtype=np.int16)
        sd.play(audio, samplerate=SAMPLE_RATE, blocksize=4096)
        sd.wait()
        self._chunks.clear()

    async def play_async(self) -> None:
        """Non-blocking wrapper for playback."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_sync)


# ── Microphone capture → WebSocket ──────────────────────────────────────────
@tracer.start_as_current_span("stream_microphone")
async def stream_microphone(
    connection, stop_event: asyncio.Event, mute_event: asyncio.Event
) -> None:
    """Continuously capture mic audio and send it to the Realtime API.

    When *mute_event* is set the mic data is silently discarded so that
    the assistant's own speaker output is not fed back into the model.
    """
    loop = asyncio.get_event_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"  ⚠ audio input status: {status}", file=sys.stderr)
        # Drop audio at the source when muted — thread-safe check
        if mute_event.is_set():
            return
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.tobytes())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
        callback=_callback,
    )

    with stream:
        while not stop_event.is_set():
            try:
                pcm_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            # Double-check mute flag (for anything already queued)
            if mute_event.is_set():
                continue
            # Send base64-encoded PCM16 to the Realtime API
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            await connection.input_audio_buffer.append(audio=b64)


# ── Event processing ────────────────────────────────────────────────────────
@tracer.start_as_current_span("process_events")
async def process_events(
    connection, stop_event: asyncio.Event, mute_event: asyncio.Event
) -> None:
    """Listen for server events: transcripts, audio, token usage."""
    player = AudioPlayer()
    assistant_transcript_buf: list[str] = []  # buffer assistant text until complete
    user_transcript_buf: list[str] = []       # buffer user text that arrives late

    async for event in connection:
        if stop_event.is_set():
            break

        # ── User speech detection ──
        if event.type == "input_audio_buffer.speech_started":
            print("�  [listening …]", flush=True)

        elif event.type == "input_audio_buffer.speech_stopped":
            print("🎙  [processing …]", flush=True)

        # ── User input transcript (may arrive while assistant is responding) ──
        elif event.type == "conversation.item.input_audio_transcription.completed":
            transcript = getattr(event, "transcript", "") or ""
            if transcript.strip():
                # Store it — we'll print it in the right order later
                user_transcript_buf.append(transcript.strip())

        elif event.type == "conversation.item.input_audio_transcription.delta":
            pass  # ignore deltas, we use the completed event

        # ── Assistant audio transcript — buffer, don't print yet ──
        elif event.type == "response.audio_transcript.delta":
            assistant_transcript_buf.append(event.delta)

        elif event.type == "response.audio_transcript.done":
            # Now print everything in the right order:
            # 1) User transcript first (if we have it)
            for ut in user_transcript_buf:
                print(f"\n🗣  You: {ut}", flush=True)
            user_transcript_buf.clear()
            # 2) Then the complete assistant response
            full_response = "".join(assistant_transcript_buf)
            if full_response.strip():
                print(f"\n🤖 Assistant: {full_response.strip()}", flush=True)
            assistant_transcript_buf.clear()

        # ── Assistant audio data ──
        elif event.type == "response.audio.delta":
            # Mute mic as soon as assistant starts producing audio
            if not mute_event.is_set():
                mute_event.set()
            player.add_chunk(event.delta)

        elif event.type == "response.audio.done":
            # Play back audio (non-blocking) then unmute mic
            await player.play_async()
            mute_event.clear()

        # ── Response complete → log tokens + emit traced span ──
        elif event.type == "response.done":
            usage = getattr(event.response, "usage", None)
            log_token_usage(usage)

            # Record a dedicated span with token attributes for App Insights
            if usage:
                with tracer.start_as_current_span("realtime_response") as span:
                    total = getattr(usage, "total_tokens", 0) or 0
                    inp = getattr(usage, "input_tokens", 0) or 0
                    out = getattr(usage, "output_tokens", 0) or 0

                    inp_details = getattr(usage, "input_token_details", None)
                    out_details = getattr(usage, "output_token_details", None)

                    span.set_attribute("gen_ai.system", "openai")
                    span.set_attribute("gen_ai.operation.name", "realtime")
                    span.set_attribute("gen_ai.request.model", DEPLOYMENT)
                    span.set_attribute("gen_ai.usage.total_tokens", total)
                    span.set_attribute("gen_ai.usage.input_tokens", inp)
                    span.set_attribute("gen_ai.usage.output_tokens", out)

                    if inp_details:
                        span.set_attribute("gen_ai.usage.input_tokens.text",
                                           getattr(inp_details, "text_tokens", 0) or 0)
                        span.set_attribute("gen_ai.usage.input_tokens.audio",
                                           getattr(inp_details, "audio_tokens", 0) or 0)
                        span.set_attribute("gen_ai.usage.input_tokens.cached",
                                           getattr(inp_details, "cached_tokens", 0) or 0)
                    if out_details:
                        span.set_attribute("gen_ai.usage.output_tokens.text",
                                           getattr(out_details, "text_tokens", 0) or 0)
                        span.set_attribute("gen_ai.usage.output_tokens.audio",
                                           getattr(out_details, "audio_tokens", 0) or 0)

                    # Include the transcript if available
                    full_text = "".join(assistant_transcript_buf) if assistant_transcript_buf else ""
                    if full_text.strip():
                        span.set_attribute("gen_ai.response.text", full_text.strip())

        # ── Errors ──
        elif event.type == "error":
            err = getattr(event, "error", event)
            print(f"\n❌ Error: {err}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────
@tracer.start_as_current_span("realtime_voice_session")
async def main() -> None:
    print_banner()

    client = AsyncAzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    print("⏳ Connecting to Realtime API …")
    async with client.realtime.connect(model=DEPLOYMENT) as connection:
        # Configure session for voice conversation
        await connection.session.update(
            session={
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "instructions": (
                    "You are a helpful, concise voice assistant. "
                    "Respond naturally and briefly."
                ),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True,
                },
            }
        )
        print("✅ Connected! Start speaking …\n")

        stop_event = asyncio.Event()
        mute_event = asyncio.Event()  # set while assistant audio is playing

        # Run mic capture and event processing concurrently
        mic_task = asyncio.create_task(stream_microphone(connection, stop_event, mute_event))
        event_task = asyncio.create_task(process_events(connection, stop_event, mute_event))

        try:
            # Wait until one of them finishes (usually won't unless error)
            done, pending = await asyncio.wait(
                [mic_task, event_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Propagate exceptions
            for t in done:
                t.result()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop_event.set()
            mic_task.cancel()
            event_task.cancel()
            # Suppress CancelledError noise
            for t in [mic_task, event_task]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    # Session summary
    print("\n" + "=" * 64)
    print("  SESSION SUMMARY")
    print("=" * 64)
    print(f"  Total tokens consumed : {session_tokens['total']}")
    print(f"  Input tokens          : {session_tokens['input']}")
    print(f"    ├ text              : {session_tokens['input_text']}")
    print(f"    ├ audio             : {session_tokens['input_audio']}")
    print(f"    └ cached            : {session_tokens['input_cached']}")
    print(f"  Output tokens         : {session_tokens['output']}")
    print(f"    ├ text              : {session_tokens['output_text']}")
    print(f"    └ audio             : {session_tokens['output_audio']}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)
