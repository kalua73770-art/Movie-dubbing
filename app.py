from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from hindi_dubbing.pipeline.orchestrator import run_pipeline
from hindi_dubbing.settings import settings

app = FastAPI(title="Hindi Movie Dubbing")
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
lock = threading.Lock()

HTML = """<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hindi Movie Dubbing</title>
<style>
body{font-family:system-ui,Arial;background:#0b1220;color:#eef2ff;max-width:760px;margin:40px auto;padding:0 16px}
.card{background:#121b2e;border:1px solid #263453;border-radius:18px;padding:24px;box-shadow:0 15px 50px #0005}
button{background:#4f7cff;color:white;border:0;border-radius:10px;padding:12px 18px;font-weight:700;cursor:pointer}
input[type=file]{width:100%;padding:14px;border:1px dashed #53678f;border-radius:10px;background:#0e1627;color:#fff}
pre{white-space:pre-wrap;background:#090f1b;padding:14px;border-radius:12px;max-height:360px;overflow:auto}
progress{width:100%;height:14px}
a{color:#8fb1ff}
.small{opacity:.75}
</style>
</head>
<body>
<div class="card">
<h1>🎬 Hindi Movie Dubbing</h1>
<p class="small">Gemini Transcribe → Flash-Lite → Gemini TTS → FFmpeg</p>
<input id="video" type="file" accept="video/*">
<br><br><button onclick="start()">Start Dubbing</button>
<div id="status" style="margin-top:20px"></div>
<progress id="bar" value="0" max="100"></progress>
<pre id="log">Waiting…</pre>
<div id="download" style="margin-top:14px"></div>
</div>
<script>
let timer=null;
async function start(){
  const f=document.getElementById('video').files[0];
  if(!f){alert('Movie select karo');return;}
  document.getElementById('status').textContent='Uploading…';
  const fd=new FormData(); fd.append('video',f);
  const r=await fetch('/api/run',{method:'POST',body:fd});
  const j=await r.json();
  if(!r.ok){document.getElementById('status').textContent=j.detail||'Upload failed';return;}
  poll(j.job_id);
}
async function poll(id){
  if(timer)clearInterval(timer);
  const tick=async()=>{
    const r=await fetch('/api/job/'+id); const j=await r.json();
    document.getElementById('status').textContent=(j.status||'')+' — '+(j.stage||'');
    document.getElementById('log').textContent=j.logs||j.message||'';
    if(j.output)document.getElementById('download').innerHTML='<a href="'+j.output+'">⬇️ Download dubbed movie</a>';
  };
  await tick(); timer=setInterval(tick,3000);
}
</script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/health")
def health():
    return {"ok": True, "service": "movie-dubbing"}

def _tail(path: Path, n=120):
    if not path.exists():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception as e:
        return f"log read error: {e}"

@app.post("/api/run")
async def start_job(video: UploadFile = File(...)):
    with lock:
        running = [j for j in jobs.values() if j["status"] == "running"]
    if running:
        raise HTTPException(409, "A dubbing job is already running")

    job_id = uuid.uuid4().hex[:12]
    work = Path(settings.work_dir) / "jobs" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / (video.filename or "input.mp4")
    with src.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    out = Path(settings.output_dir) / f"{job_id}_hindi.mp4"
    jobs[job_id] = {
        "status": "running", "stage": "queued", "message": "Job queued",
        "source": str(src), "output_path": str(out), "log_path": str(work / "pipeline.log")
    }
    executor.submit(_run, job_id, src, out)
    return {"job_id": job_id}

def _run(job_id, src, out):
    def cb(stage, msg):
        with lock:
            jobs[job_id]["stage"] = stage
            jobs[job_id]["message"] = msg
    try:
        run_pipeline(src, out, progress=cb, job_id=job_id)
        with lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["stage"] = "done"
    except Exception as e:
        with lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["stage"] = "error"
            jobs[job_id]["message"] = str(e)

@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        work = Path(settings.work_dir) / "jobs" / job_id
        manifest = work / "project.json"
        if manifest.exists():
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                job["log_path"] = str(work / "pipeline.log")
                job["output_path"] = str(Path(settings.output_dir) / f"{job_id}_hindi.mp4")
            except Exception:
                job = None
    if not job:
        raise HTTPException(404, "Job not found")

    payload = dict(job)
    payload["logs"] = _tail(Path(job.get("log_path", "")))
    out = Path(job.get("output_path", ""))
    payload["output"] = f"/api/download/{job_id}" if out.exists() else None
    return JSONResponse(payload)

@app.get("/api/download/{job_id}")
def download(job_id: str):
    with lock:
        job = jobs.get(job_id)
    path = Path(job["output_path"]) if job else Path(settings.output_dir) / f"{job_id}_hindi.mp4"
    if not path.exists():
        raise HTTPException(404, "Output not ready")
    return FileResponse(path, filename=path.name, media_type="video/mp4")
