# Azure OpenAI Realtime API — Voice Console Chat

A speech-in / speech-out console application that connects to the **Azure OpenAI GPT-4o Realtime API** over WebSocket. You speak into your microphone, the model processes your audio in real time, and replies with synthesised speech — all without a separate STT or TTS service.

## How It Works

### The Realtime API Is Event-Driven

Unlike the standard Chat Completions REST API (send a request, get a response), the Realtime API operates over a **persistent WebSocket connection**. Both client and server exchange a stream of typed **events** — there is no single request/response cycle. This is why the code is structured around an async event loop rather than simple HTTP calls.

The application runs **two concurrent async tasks**:

| Task | Purpose |
|---|---|
| `stream_microphone` | Captures PCM16 audio from the microphone in 100 ms chunks, base64-encodes each chunk, and sends it to the model via `input_audio_buffer.append` events. |
| `process_events` | Listens for server-sent events and reacts to each type accordingly (see below). |

### Server Events We Handle

| Event | What it means | What we do |
|---|---|---|
| `input_audio_buffer.speech_started` | VAD detected the user started speaking | Print a listening indicator |
| `input_audio_buffer.speech_stopped` | VAD detected silence — utterance complete | Print a processing indicator |
| `conversation.item.input_audio_transcription.completed` | Whisper transcript of what the user said | Buffer it for ordered printing |
| `response.audio_transcript.delta` | Incremental text of the assistant's reply | Buffer partial text |
| `response.audio_transcript.done` | Assistant transcript is complete | Print user transcript, then assistant transcript |
| `response.audio.delta` | A chunk of the assistant's audio reply (base64 PCM16) | Mute the mic (to avoid feedback), accumulate audio |
| `response.audio.done` | All audio chunks sent | Play back the accumulated audio, unmute the mic |
| `response.done` | Full response finished — includes token usage | Log token counts, emit an OpenTelemetry span |
| `error` | Something went wrong server-side | Print the error |

### Why Mute the Microphone During Playback?

The model uses **server-side Voice Activity Detection (VAD)** to decide when you've finished speaking. If the assistant's speaker output is picked up by the microphone, the model would interpret its own voice as new user input — creating an echo loop. The code sets a `mute_event` flag as soon as assistant audio arrives and clears it after playback finishes.

### Token Usage Tracking

Each `response.done` event carries a `usage` object with detailed token counts:

- **Input tokens** — broken down into *text*, *audio*, and *cached* tokens
- **Output tokens** — broken down into *text* and *audio* tokens

Audio tokens are significantly more expensive than text tokens, so visibility into this breakdown matters for cost control. The app tracks both **per-response** and **cumulative session** totals, printing them in a formatted table after every assistant turn. A final session summary is printed on exit.

### OpenTelemetry Tracing

When an Application Insights connection string is provided, the app configures **Azure Monitor** as the OpenTelemetry exporter. Each response generates a `realtime_response` span carrying `gen_ai.*` semantic attributes (model, token counts, transcript). These spans appear in the **AI Foundry Tracing** view, giving you production observability without a separate logging pipeline.

> **Note:** The standard `OpenAIInstrumentor` only instruments REST-based SDK calls (chat completions, embeddings, etc.) and does **not** cover the WebSocket-based Realtime API. That's why this app creates manual spans instead.

## External Libraries

| Library | Why it's needed |
|---|---|
| **[openai](https://github.com/openai/openai-python)** | The official OpenAI Python SDK. Provides `AsyncAzureOpenAI` and the `client.realtime.connect()` context manager that establishes the WebSocket session and exposes a typed async iterator over server events. |
| **[sounddevice](https://python-sounddevice.readthedocs.io/)** | Python bindings for [PortAudio](http://www.portaudio.com/). Used to capture microphone input (`sd.InputStream`) and play back audio (`sd.play`). Requires the `portaudio` system library (`brew install portaudio` on macOS). |
| **[numpy](https://numpy.org/)** | Used to convert raw PCM16 byte buffers into typed arrays that `sounddevice` can play. Also enables efficient audio concatenation. |
| **[python-dotenv](https://github.com/theskumar/python-dotenv)** | Loads environment variables from a `.env` file so you don't have to export them manually in every shell session. |
| **[websockets](https://websockets.readthedocs.io/)** | The underlying WebSocket transport used by the OpenAI SDK's Realtime client. It is not imported directly in `main.py` but is a required runtime dependency. |
| **[azure-monitor-opentelemetry](https://learn.microsoft.com/python/api/overview/azure/monitor-opentelemetry-readme)** | One-liner setup (`configure_azure_monitor`) that wires OpenTelemetry traces, metrics, and logs to Azure Application Insights. |
| **[azure-identity](https://learn.microsoft.com/python/api/overview/azure/identity-readme)** | Provides `DefaultAzureCredential` and other credential classes for Azure authentication. Included for scenarios where you switch from API-key auth to Entra ID / managed identity. |
| **[azure-ai-projects](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)** | SDK for Azure AI Foundry project-level operations. Listed as a dependency for broader AI Foundry integration (e.g. evaluations, prompt flow). |

## Prerequisites

- Python 3.10+
- A microphone (built-in or USB)
- `portaudio` system library — on macOS: `brew install portaudio`
- An Azure OpenAI resource with a **`gpt-4o-realtime-preview`** deployment (available in Sweden Central or East US 2)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-realtime-preview
AZURE_OPENAI_API_VERSION=2025-04-01-preview
APPLICATIONINSIGHTS_CONNECTION_STRING=<optional — enables tracing>
```

## Run

```bash
python main.py
```

Speak into your microphone. The assistant replies through your speakers and transcripts plus token usage are printed to the console. Press **Ctrl+C** to exit and see a session summary.
