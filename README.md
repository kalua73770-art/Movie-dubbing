# Hindi Movie Dubbing — Cloud First

Heavy local ML has been removed. The server uses Gemini APIs plus FFmpeg.

## Pipeline

FFmpeg audio extraction → Gemini 3.5 Transcribe (transcription + diarization + timestamps) → Gemini Flash-Lite translation → Gemini TTS → FFmpeg timeline fitting → original-audio ducking → final MP4.

Diarization + word timestamps are requested, so transcription chunks are kept under 30 minutes.

## API keys

Set GEMINI_API_KEYS as comma-separated keys:

GEMINI_API_KEYS=key1,key2,key3

The client rotates through every configured key after a failed call. Keys are never written to logs.

## Render

The repo contains Dockerfile and render.yaml. Create a Render Web Service from this repo and keep GEMINI_API_KEYS as a secret environment variable.

## Local

pip install .
export GEMINI_API_KEYS="key1,key2"
uvicorn app:app --host 0.0.0.0 --port 10000

Then open /.

## Job files

Each job is stored under WORK_DIR/jobs/<job_id>/:
- pipeline.log
- project.json
- chunks/
- tts/
- segments/
- audio/
