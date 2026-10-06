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


## Production TTS routing correction — 2026-10-05

Fix signature: tts-single-mode-was-still-using-batch-splitting
Attempt: 1/3.

Evidence: despite `TTS_MODE=single`, run #100 still executed `TTS batch` and `_split_batch_audio`, proving the orchestrator had not actually routed production single-turn mode through `GeminiService.tts_segment()`. This allowed silence-boundary extraction to clip dialogue and caused a 5h29m run before cancellation.

Fix:
- Production `TTS_MODE=single` now calls `_generate_segment_audio()` / `tts_segment()` once per dialogue line.
- Batch synthesis remains available only when explicitly configured as `TTS_MODE=batch`.
- Single-turn TTS uses the fixed single-speaker voice config and no unsupported speech annotations.
- Existing 75% coverage QA remains enabled, and duration rewrite attempts remain 3.


## Duration/prosody recovery fix — 2026-10-06

Fix signature: tts-short-response-fatal-no-rewrite-budget
Attempt: 1/3.

Evidence: run #102 failed on `seg_000002` after Gemini returned only 0.682s for a 1.600s target (42.6% coverage). The orchestrator raised immediately even though the existing duration-rewrite helper was designed to regenerate short dialogue. The helper was also artificially capped to one rewrite attempt.

Fix:
- Honor up to 3 bounded duration-rewrite attempts.
- Rotate preferred TTS model/key between rewrite attempts.
- TTS now receives the target duration and requested pace explicitly and is told not to rush.
- Low duration after bounded retries is recorded as a quality warning rather than failing the entire movie, so one imperfect line cannot prevent the final MP4 from being produced.
- This follows the same high-level principles Meta describes for expressive translation: preserve vocal style, speech rate, rhythm and pauses, while lip syncing remains a separate downstream stage. citeturn778645search1turn778645search0
