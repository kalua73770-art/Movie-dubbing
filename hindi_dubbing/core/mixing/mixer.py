from array import array
from pathlib import Path
import wave

from hindi_dubbing.core.audio.processor import run_cmd


def _build_mute_mask(segments: list[dict], duration: float, out: Path, sample_rate: int = 1000):
    """
    Build a tiny control WAV: 1 outside dialogue and 0 during every dialogue
    segment. 1 kHz gives 1 ms timing resolution while keeping the mask small.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    total_samples = max(1, int(round(duration * sample_rate)))
    ordered = sorted(
        (
            max(0.0, float(s["start"])),
            min(duration, float(s["end"]))
        )
        for s in segments
        if float(s.get("end", 0)) > float(s.get("start", 0))
    )

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        cursor = 0
        one = array("h", [32767] * sample_rate).tobytes()
        zero = array("h", [0] * sample_rate).tobytes()

        def write_value(count: int, block: bytes):
            remaining = max(0, count)
            block_len = len(block)
            while remaining:
                n = min(remaining, sample_rate)
                wf.writeframes(block[: n * 2])
                remaining -= n

        for start, end in ordered:
            a = min(total_samples, max(cursor, int(round(start * sample_rate))))
            b = min(total_samples, max(a, int(round(end * sample_rate))))
            write_value(a - cursor, one)
            write_value(b - a, zero)
            cursor = b

        write_value(total_samples - cursor, one)

    return out


def mix_and_duck(
    video: Path,
    dialogue: Path,
    out: Path,
    segments: list[dict] | None = None,
    log=None,
):
    """
    MVP mixing mode: the original soundtrack is exactly muted during the
    detected dialogue windows, then restored between lines.

    This intentionally mutes BGM as well during dialogue because the source
    speech/music tracks are not separated yet. That is preferable to leaving
    the original English speech audible underneath the Hindi dub.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    mask = out.parent / f"{out.stem}_mute_mask.wav"

    if segments:
        _build_mute_mask(segments, max(0.0, max(s["end"] for s in segments)), mask)
        if log:
            log.info("Built exact dialogue mute mask: %s", mask)

        filter_complex = (
            "[0:a:0]aresample=48000,aformat=sample_fmts=fltp,"
            "pan=stereo|c0=c0|c1=c0[orig];"
            "[1:a:0]aresample=48000,aformat=sample_fmts=fltp,"
            "pan=stereo|c0=c0|c1=c0[dlg];"
            "[2:a:0]aresample=48000,aformat=sample_fmts=fltp,"
            "pan=stereo|c0=c0|c1=c0[mask];"
            "[orig][mask]amultiply[duck];"
            "[duck][dlg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
        run_cmd([
            "ffmpeg", "-y",
            "-i", video,
            "-i", dialogue,
            "-i", mask,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            out,
        ], log)
    else:
        # Backward-compatible fallback if a caller does not supply segments.
        filter_complex = (
            "[0:a:0][1:a:0]sidechaincompress="
            "threshold=0.001:ratio=20:attack=2:release=80:"
            "makeup=1:knee=1:detection=peak:level_sc=2:mix=1[duck];"
            "[duck][1:a:0]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
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

    try:
        mask.unlink()
    except Exception:
        pass
    return out
