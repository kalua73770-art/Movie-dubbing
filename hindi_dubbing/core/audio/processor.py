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

def assemble_track(items, total_duration: float, out: Path):
    out.parent.mkdir(parents=True,exist_ok=True)
    sr=24000; sw=2
    with wave.open(str(out),"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(sw); wf.setframerate(sr)
        cursor=0.0
        silence_block=b"\0"* (sr*sw)
        for start,end,path in sorted(items,key=lambda x:x[0]):
            if start>cursor:
                frames=int((start-cursor)*sr)
                while frames:
                    n=min(frames,sr); wf.writeframes(silence_block[:n*sw]); frames-=n
            with wave.open(str(path),"rb") as src:
                data=src.readframes(src.getnframes())
            overlap=max(0.0,cursor-start)
            if overlap:
                cut=int(overlap*sr)*sw; data=data[cut:]
            wf.writeframes(data); cursor=max(cursor,end)
        if cursor<total_duration:
            frames=int((total_duration-cursor)*sr)
            while frames:
                n=min(frames,sr); wf.writeframes(silence_block[:n*sw]); frames-=n
