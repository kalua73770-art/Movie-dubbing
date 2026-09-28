from pathlib import Path
from hindi_dubbing.core.audio.processor import run_cmd


def mix_final(video: Path, dialogue: Path, background: Path | None, out: Path, segments=None, log=None):
    """
    Prefer separated Music+SFX stems. If separation is unavailable, fall back to
    exact muting of the original soundtrack during dialogue windows.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    if background is not None and background.exists():
        # Gentle dialogue ducking over the reconstructed background bed.
        filter_complex = (
            "[0:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[bg];"
            "[1:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[dlg];"
            "[bg][dlg]sidechaincompress=threshold=0.05:ratio=2:attack=10:release=120:"
            "makeup=1:knee=1:detection=peak:level_sc=2:mix=1[duck];"
            "[duck][dlg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
        run_cmd([
            "ffmpeg", "-y",
            "-i", background,
            "-i", dialogue,
            "-filter_complex", filter_complex,
            "-map", "0:a:0",
            "-map", "1:a:0",
            "-map", "[a]",
            "-map", "0:a:0",
            "-f", "null", "-"
        ], log) if False else None

        run_cmd([
            "ffmpeg", "-y",
            "-i", video,
            "-i", background,
            "-i", dialogue,
            "-filter_complex",
            "[1:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[bg];"
            "[2:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[dlg];"
            "[bg][dlg]sidechaincompress=threshold=0.05:ratio=2:attack=10:release=120:"
            "makeup=1:knee=1:detection=peak:level_sc=2:mix=1[duck];"
            "[duck][dlg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            out,
        ], log)
        return out

    if not segments:
        raise RuntimeError("No separated background stem and no dialogue segments available for fallback mute")

    duration = max(float(s["end"]) for s in segments)
    mask = out.parent / f"{out.stem}_mute_mask.wav"

    # 1 kHz control signal: 1 outside speech, 0 during speech.
    import wave
    from array import array

    total = max(1, int(round(duration * 1000)))
    with wave.open(str(mask), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1000)
        one = array("h", [32767] * 1000).tobytes()
        zero = array("h", [0] * 1000).tobytes()
        cursor = 0

        def write(count, block):
            while count > 0:
                n = min(count, 1000)
                wf.writeframes(block[:n * 2])
                count -= n

        for seg in sorted(segments, key=lambda x: x["start"]):
            a = max(cursor, int(round(float(seg["start"]) * 1000)))
            b = min(total, max(a, int(round(float(seg["end"]) * 1000))))
            write(a - cursor, one)
            write(b - a, zero)
            cursor = b
        write(total - cursor, one)

    run_cmd([
        "ffmpeg", "-y",
        "-i", video,
        "-i", dialogue,
        "-i", mask,
        "-filter_complex",
        "[0:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[orig];"
        "[1:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[dlg];"
        "[2:a:0]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0[mask];"
        "[orig][mask]amultiply[duck];"
        "[duck][dlg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
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
