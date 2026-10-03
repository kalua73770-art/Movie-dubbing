# Automatic Fix State

Signature: `gemini-transcribe-foreground-interaction-hang`
Status: active
Consecutive automatic fix attempts: 1/3

## Trigger
Movie Dubbing Test #92 (run `37096193388`) reached transcription and stayed inside
`client.interactions.create()` for the full 330-minute GitHub job timeout. The visible
terminal exception was `Operation was canceled`; no dubbed-video artifact was produced.

## Root cause
The transcription path used a synchronous Gemini Interactions request. Unlike the
already-hardened Antigravity path, it did not use background execution plus polling, so
the SDK could keep the request open far beyond the intended 600-second HTTP timeout.
Google's current Interactions API documentation explicitly supports `background=True`
with `interactions.get()` polling for long-running work.

## Fix applied
Changed Gemini transcription to:
1. create the transcription interaction with `background=True`;
2. poll the interaction ID every 5 seconds;
3. enforce a bounded 1800-second transcription wait budget;
4. delete the uploaded audio only after the interaction reaches a terminal state;
5. fail fast with a clear terminal-status/timeout error instead of allowing a job to
   hang until the GitHub runner limit.

## Verification required
The next Movie Dubbing Test must reach transcription completion and then produce
`test-output/dubbed_hindi_latest.mp4` and the `dubbed-video-latest` artifact.

## Retry policy
Do not automatically apply another fix for this same signature after three consecutive
failed verification runs. If the new background interaction itself fails because of
Gemini/API availability, credentials, quota, or another external-service condition,
pause automatic fixing and record the human action required.

## Verification fix 2

Run #93 showed that `background=True` is not supported by the Gemini 3.5 Transcribe
endpoint: every key returned HTTP 400 stating that audio input modality is not enabled
for the generated `gemini-3.5-transcribe-agent` target. The background change was therefore
rolled back for Transcribe only.

The Transcribe path now uses the documented synchronous Interactions REST endpoint with
an explicit 600-second HTTP client timeout. This keeps the supported Transcribe contract
while ensuring a stalled network request can fail and rotate to the next key rather than
holding the GitHub runner for hours.


## Verification trigger

The follow-up verification run is intentionally triggered only after the code fix is on `main`. No parallel Movie Dubbing Test run was active at trigger time.
