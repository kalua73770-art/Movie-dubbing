from pathlib import Path
from hindi_dubbing.core.audio.processor import run_cmd

def mix_and_duck(video: Path, dialogue: Path, out: Path, duck_db=-40.0, log=None):
    """
    Temporarily suppress the original soundtrack whenever Hindi dialogue is present.

    This is intentionally aggressive for the current MVP: because we do not have
    reliable speech/music separation, the original speech AND any BGM underneath
    the dialogue are muted/strongly ducked while the Hindi line plays. Outside
    dialogue, the original soundtrack stays at full level.
    """
    threshold = max(0.000976563, min(1.0, 10 ** (duck_db / 20)))
    ratio = 20
    filter_complex = (
        f"[0:a:0][1:a:0]sidechaincompress="
        f"threshold={threshold:.6f}:ratio={ratio}:attack=2:release=80:"
        f"makeup=1:knee=1:detection=peak:level_sc=2:mix=1[duck];"
        "[duck][1:a:0]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y",
        "-i", video,
        "-i", dialogue,
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out,
    ], log)
    return out
