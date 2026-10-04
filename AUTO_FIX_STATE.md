# Automatic Fix State

## Audio-quality verification — run #94

Fix signature: tts-batch-splitting-plus-roman-hindi-pronunciation
Attempt: 1/3.

Evidence:
- Multi-line TTS splitting produced clips much shorter than their dialogue slots; for example ~1.09s generated for a 9.2s source line.
- Generated Hindi text was Roman/Hinglish, which can encourage English-phonetic pronunciation.

Fix:
- Single-turn TTS is now the default for production.
- Hindi translation is requested in Devanagari.
- TTS explicitly requests natural Indian-Hindi phonetics.
- Single-turn clips below 75% of their target duration are rejected instead of silently producing clipped dialogue.


## TTS API request-shape failure — 2026-10-04

Fix signature: tts-speech-annotations-unsupported
Attempt: 1/3.

Evidence: the test repeatedly returned HTTP 400 `invalid_request` because `speech_metadata` annotations are not supported by `gemini-2.5-flash-preview-tts`. The old retry policy then retried the same deterministic request across five API keys, multiplying latency before the run was cancelled.

Fix:
- Single-speaker TTS input no longer sends speech annotations; the voice remains selected through `speech_config`.
- Deterministic `request` errors stop key-by-key retries instead of repeating the same invalid request.


## TTS single-speaker configuration failure — 2026-10-04

Fix signature: tts-single-speaker-config-schema
Attempt: 1/3.

Evidence: run #99 returned HTTP 400 with `multi_speaker_voice_config.speaker_voice_configs` requiring exactly 2 configs, even though each failing request contained `segments=1 speakers=1`.

Root cause: the single-speaker branch constructed the multi-speaker `{"speakers":[...]}` shape. Gemini's TTS API expects single-speaker `speech_config` to be an array such as `[{"voice":"Kore"}]`; the `speakers` object is for two-speaker generation. citeturn573792search0turn573792search3

Fix: single-speaker TTS now emits the single-speaker array voice config. This is committed in `6b219deb6b71fe123c4fabc27451cc2f94f4e03c`.
