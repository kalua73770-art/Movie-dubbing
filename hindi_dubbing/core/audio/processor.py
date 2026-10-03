from pathlib import Path
import subprocess, wave, re

def run_cmd(cmd, log=None):
    if log: log.info("$ %s", " ".join(map(str,cmd)))
    r=subprocess.run([str(x) for x in cmd],capture_output=True,text=True)
    if r.returncode:
        msg=(r.stderr or r.stdout or "command failed").strip()
        if log: log.error(msg[-4000:])
        raise RuntimeError(msg)
    return r.stdout

def ffprobe_duration(path: Path) -> float:
    out=run_cmd(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",path]).strip()
    return float(out)

def extract_audio(video: Path, wav: Path, log=None):
    wav.parent.mkdir(parents=True,exist_ok=True)
    run_cmd(["ffmpeg","-y","-i",video,"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",wav],log)

def split_audio(audio: Path, out_dir: Path, duration: float, chunk_seconds: int, overlap: int, log=None):
    out_dir.mkdir(parents=True,exist_ok=True)
    chunks=[]; start=0.0; idx=0
    step=max(1,chunk_seconds-overlap)
    while start<duration:
        end=min(duration,start+chunk_seconds)
        out=out_dir/f"chunk_{idx:04d}.wav"
        if not out.exists():
            run_cmd(["ffmpeg","-y","-ss",f"{start:.3f}","-t",f"{end-start:.3f}","-i",audio,"-ac","1","-ar","16000","-c:a","pcm_s16le",out],log)
        chunks.append((out,start,end))
        if end>=duration: break
        start+=step; idx+=1
    return chunks

def probe_wav(path: Path):
    with wave.open(str(path),"rb") as wf:
        return wf.getframerate(),wf.getnchannels(),wf.getsampwidth(),wf.getnframes()/wf.getframerate()


def audio_quality(path: Path):
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
    if width != 2 or not raw:
        return 0.0, 0.0
    import array
    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0, 0.0
    peak = max(abs(int(x)) for x in samples) / 32768.0
    rms = (sum(float(x) * float(x) for x in samples) / len(samples)) ** 0.5 / 32768.0
    return rms, peak


def usable_speech(path: Path, min_rms: float = 0.002, min_peak: float = 0.015):
    rms, peak = audio_quality(path)
    return rms >= min_rms and peak >= min_peak, rms, peak

def fit_audio(src: Path, dst: Path, target: float, log=None):
    src_d=probe_wav(src)[3]
    if target<=0: raise ValueError("target duration must be > 0")
    ratio=src_d/target if src_d else 1.0
    filters=[]
    while ratio<0.5:
        filters.append("atempo=0.5"); ratio/=0.5
    while ratio>2.0:
        filters.append("atempo=2.0"); ratio/=2.0
    filters.append(f"atempo={ratio:.6f}")
    dst.parent.mkdir(parents=True,exist_ok=True)
    run_cmd(["ffmpeg","-y","-i",src,"-af",",".join(filters),"-t",f"{target:.6f}","-ac","1","-ar","24000","-c:a","pcm_s16le",dst],log)

def silence_ranges(path: Path, log=None, noise="-42dB", min_silence=0.35):
    cmd=["ffmpeg","-hide_banner","-i",path,"-af",f"silencedetect=noise={noise}:d={min_silence}","-f","null","-"]
    if log: log.info("$ %s"," ".join(map(str,cmd)))
    r=subprocess.run(cmd,capture_output=True,text=True)
    text=(r.stderr or "")
    sil=[]; start=None
    for line in text.splitlines():
        m=re.search(r"silence_start: ([0-9.]+)",line)
        if m: start=float(m.group(1)); continue
        m=re.search(r"silence_end: ([0-9.]+)",line)
        if m and start is not None: sil.append((start,float(m.group(1)))); start=None
    dur=probe_wav(path)[3]
    ranges=[]; cur=0.0
    for s,e in sil:
        if s>cur+0.03: ranges.append((cur,s))
        cur=e
    if cur<dur-0.03: ranges.append((cur,dur))
    return ranges

def extract_range(src: Path, dst: Path, start: float, end: float, log=None):
    dst.parent.mkdir(parents=True,exist_ok=True)
    run_cmd(["ffmpeg","-y","-ss",f"{start:.4f}","-t",f"{max(0,end-start):.4f}","-i",src,"-ac","1","-ar","24000","-c:a","pcm_s16le",dst],log)

def assemble_track(items, total_duration: float, out: Path, chunk_seconds: float = 30.0, log=None):
    """
    Build the dialogue timeline with real sample mixing.

    The previous implementation truncated a new line whenever it overlapped the
    previous line. That could silently delete dialogue. This version mixes
    overlapping lines instead of cutting them away.
    """
    import numpy as np

    out.parent.mkdir(parents=True, exist_ok=True)
    sr = 24000
    sw = 2
    normalized = []

    for start, end, path in sorted(items, key=lambda x: float(x[0])):
        if end <= start:
            continue
        with wave.open(str(path), "rb") as src:
            src_sr = src.getframerate()
            channels = src.getnchannels()
            raw = src.readframes(src.getnframes())
        if channels != 1:
            data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).astype(np.float32)
            data = np.mean(data, axis=1)
        else:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if src_sr != sr:
            converted = out.parent / f".resample_{Path(path).stem}_{sr}.wav"
            run_cmd([
                "ffmpeg", "-y", "-i", path,
                "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", converted
            ], log)
            with wave.open(str(converted), "rb") as src2:
                data = np.frombuffer(src2.readframes(src2.getnframes()), dtype=np.int16).astype(np.float32)
            try:
                converted.unlink()
            except Exception:
                pass
        data /= 32768.0
        normalized.append((float(start), float(end), data))

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sw)
        wf.setframerate(sr)

        chunk_frames = max(1, int(round(chunk_seconds * sr)))
        total_frames = max(1, int(round(total_duration * sr)))

        for chunk_start in range(0, total_frames, chunk_frames):
            chunk_end = min(total_frames, chunk_start + chunk_frames)
            buf = np.zeros(chunk_end - chunk_start, dtype=np.float32)

            for item_start, item_end, data in normalized:
                if item_end <= chunk_start / sr or item_start >= chunk_end / sr:
                    continue

                src_start = max(0, int(round((chunk_start / sr - item_start) * sr)))
                dst_start = max(0, int(round((item_start - chunk_start / sr) * sr)))
                available = min(
                    len(data) - src_start,
                    len(buf) - dst_start,
                )
                if available <= 0:
                    continue

                block = data[src_start:src_start + available].copy()

                # Tiny edge fades prevent clicks when individual TTS files are placed.
                fade = min(int(sr * 0.018), available // 2)
                if fade > 0:
                    block[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                    block[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

                buf[dst_start:dst_start + available] += block

            peak = float(np.max(np.abs(buf))) if len(buf) else 0.0
            if peak > 0.97:
                buf *= 0.97 / peak

            wf.writeframes((np.clip(buf, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())


def trim_edge_silence(src: Path, dst: Path, log=None):
    """Remove only leading/trailing silence from generated TTS; keep internal pauses."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y", "-i", src,
        "-af",
        "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:"
        "stop_periods=1:stop_duration=0.08:stop_threshold=-45dB",
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", dst
    ], log)
    return dst
