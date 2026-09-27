import argparse
from pathlib import Path
from hindi_dubbing.pipeline.orchestrator import run_pipeline

def main():
    p=argparse.ArgumentParser(description="Cloud-first Hindi movie dubbing")
    p.add_argument("input_video")
    p.add_argument("output_video", nargs="?", default=None)
    a=p.parse_args()
    src=Path(a.input_video)
    out=Path(a.output_video or f"{src.stem}_hindi.mp4")
    run_pipeline(src, out, progress=lambda s,m: print(f"[{s}] {m}", flush=True))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
