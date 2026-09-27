from pathlib import Path
from hindi_dubbing.core.audio.processor import run_cmd

def mix_and_duck(video: Path, dialogue: Path, out: Path, duck_db=-18.0, log=None):
    # Sidechain-compress the original audio using the Hindi dialogue as the trigger.
    threshold=10 ** (duck_db/20)
    ratio=10
    filter_complex=(f"[0:a:0][1:a:0]sidechaincompress=threshold={threshold:.6f}:ratio={ratio}:attack=10:release=300:makeup=1[duck];"
                    "[duck][1:a:0]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    out.parent.mkdir(parents=True,exist_ok=True)
    run_cmd(["ffmpeg","-y","-i",video,"-i",dialogue,"-filter_complex",filter_complex,"-map","0:v:0","-map","[a]","-c:v","copy","-c:a","aac","-b:a","192k","-shortest",out],log)
    return out
