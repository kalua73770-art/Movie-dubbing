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
