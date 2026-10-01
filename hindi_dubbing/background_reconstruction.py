from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import numpy as np

from hindi_dubbing.core.audio.processor import ffprobe_duration, run_cmd


def speech_blocks(segments: list[dict], duration: float, merge_gap: float = 0.35, pad: float = 0.08):
    ordered = sorted(
        (
            max(0.0, float(s["start"]) - pad),
            min(duration, float(s["end"]) + pad),
            s.get("id", ""),
        )
        for s in segments
        if float(s.get("end", 0)) > float(s.get("start", 0))
    )
    blocks = []
    for start, end, _ in ordered:
        if not blocks or start - blocks[-1]["end"] > merge_gap:
            blocks.append({"start": start, "end": end})
        else:
            blocks[-1]["end"] = max(blocks[-1]["end"], end)
    return blocks


def _detect_scene_cuts(video: Path, log=None, threshold: float = 0.35):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video,
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    if log:
        log.info("$ %s", " ".join(map(str, cmd)))
    result = run_cmd(cmd, log=None)
    # showinfo writes to stderr; run_cmd captures stderr only when it fails, so use a
    # direct call here to retain showinfo output without treating it as an error.
    import subprocess

    proc = subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        text=True,
    )
    text = proc.stderr or proc.stdout or ""
    cuts = []
    for line in text.splitlines():
        match = re.search(r"pts_time:([0-9.]+)", line)
        if match:
            cuts.append(float(match.group(1)))
    return cuts


def _scene_ranges(duration: float, cuts: list[float]):
    points = [0.0]
    for cut in sorted(cuts):
        if 0.5 < cut < duration - 0.5:
            if not points or abs(cut - points[-1]) > 0.5:
                points.append(cut)
    points.append(duration)
    return [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
        if points[i + 1] - points[i] > 0.25
    ]


def _clean_ranges(scene_start, scene_end, blocks):
    result = []
    cursor = scene_start
    for block in blocks:
        start = max(scene_start, float(block["start"]))
        end = min(scene_end, float(block["end"]))
        if end <= scene_start or start >= scene_end:
            continue
        if start > cursor + 0.25:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if scene_end > cursor + 0.25:
        result.append((cursor, scene_end))
    return result


def _choose_sample(clean_ranges, block_start, block_end, target_len=6.0):
    candidates = []
    for start, end in clean_ranges:
        length = end - start
        if length < 0.8:
            continue
        if end <= block_start:
            distance = block_start - end
            sample_start = max(start, end - min(target_len, length))
            sample_end = end
        elif start >= block_end:
            distance = start - block_end
            sample_start = start
            sample_end = min(end, start + min(target_len, length))
        else:
            continue
        score = distance + max(0.0, target_len - (sample_end - sample_start)) * 0.15
        candidates.append((score, sample_start, sample_end, distance))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, start, end, distance = candidates[0]
    return {
        "start": start,
        "end": end,
        "distance": distance,
    }


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    return sample_rate, channels, data.astype(np.float32) / 32768.0


def _write_wav(path: Path, sample_rate: int, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    flat = pcm.reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(flat.tobytes())


def _loop_with_crossfade(sample_path: Path, duration: float, out: Path, log=None, crossfade_ms: int = 120):
    sr, channels, sample = _read_wav(sample_path)
    target_frames = max(1, int(round(duration * sr)))
    if sample.shape[0] == 0:
        raise RuntimeError(f"Empty music sample: {sample_path}")

    overlap = max(1, min(int(sr * crossfade_ms / 1000.0), sample.shape[0] // 4))
    output = sample[:min(target_frames, sample.shape[0])].copy()

    while output.shape[0] < target_frames:
        remaining = target_frames - output.shape[0]
        take = min(remaining, sample.shape[0])
        add = sample[:take].copy()
        ov = min(overlap, output.shape[0], add.shape[0])
        if ov:
            fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
            output[-ov:] = output[-ov:] * (1.0 - fade) + add[:ov] * fade
            add = add[ov:]
        if add.shape[0] == 0:
            break
        output = np.concatenate([output, add], axis=0)

    if output.shape[0] < target_frames:
        pad = np.zeros((target_frames - output.shape[0], channels), dtype=np.float32)
        output = np.concatenate([output, pad], axis=0)
    else:
        output = output[:target_frames]

    # Very short edge fades avoid a click at the replacement boundaries.
    edge = min(int(sr * 0.05), output.shape[0] // 4)
    if edge > 0:
        output[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
        output[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None]
    _write_wav(out, sr, output)


def reconstruct_background(
    source_audio: Path,
    source_video: Path,
    speech_segments: list[dict],
    work_dir: Path,
    log=None,
):
    """
    Reconstruct the soundtrack without a separation model.

    Dialogue windows are replaced with loops made only from nearby clean,
    dialogue-free material in the same detected scene. Clean parts of the
    original soundtrack remain byte-for-time equivalent except for re-encoding.
    """
    duration = ffprobe_duration(source_video)
    blocks = speech_blocks(speech_segments, duration)

    cuts = []
    try:
        cuts = _detect_scene_cuts(source_video, log=log)
    except Exception as exc:
        if log:
            log.warning("Scene detection unavailable; using dialogue-local cue boundaries: %s", str(exc)[:600])

    scenes = _scene_ranges(duration, cuts) or [(0.0, duration)]
    samples_dir = work_dir / "background_reconstruction" / "samples"
    pieces_dir = work_dir / "background_reconstruction" / "pieces"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pieces_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "background_reconstruction.json"
    manifest = {
        "version": 1,
        "duration": duration,
        "scenes": scenes,
        "speech_blocks": blocks,
        "cues": [],
        "pieces": [],
    }

    def scene_for(t):
        for i, (start, end) in enumerate(scenes):
            if start <= t < end or (i == len(scenes) - 1 and t <= end):
                return i, start, end
        return len(scenes) - 1, scenes[-1][0], scenes[-1][1]

    cursor = 0.0
    concat_files = []
    piece_index = 0

    for block_index, block in enumerate(blocks):
        block_start = max(cursor, block["start"])
        block_end = max(block_start, block["end"])
        if block_end <= block_start:
            continue

        scene_id, scene_start, scene_end = scene_for((block_start + block_end) / 2.0)
        clean = _clean_ranges(scene_start, scene_end, blocks)
        chosen = _choose_sample(clean, block_start, block_end, target_len=6.0)

        if chosen is None:
            # Last-resort fallback: search the closest clean material outside the scene.
            all_clean = []
            for sc_start, sc_end in scenes:
                all_clean.extend(_clean_ranges(sc_start, sc_end, blocks))
            chosen = _choose_sample(
                all_clean,
                block_start,
                block_end,
                target_len=5.0,
            )
            if chosen:
                chosen["fallback_scene"] = True

        if chosen is None:
            raise RuntimeError(
                f"No dialogue-free music sample available for block {block_index}"
            )

        cue_key = f"scene_{scene_id:04d}_block_{block_index:04d}"
        sample_path = samples_dir / f"{cue_key}_sample.wav"
        replacement_path = pieces_dir / f"{cue_key}_replacement.wav"

        if not sample_path.exists():
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{chosen['start']:.4f}",
                    "-t",
                    f"{chosen['end'] - chosen['start']:.4f}",
                    "-i",
                    source_audio,
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-c:a",
                    "pcm_s16le",
                    sample_path,
                ],
                log,
            )

        if not replacement_path.exists():
            _loop_with_crossfade(
                sample_path,
                block_end - block_start,
                replacement_path,
                log,
            )

        if block_start > cursor + 0.001:
            clean_path = pieces_dir / f"clean_{piece_index:06d}.wav"
            piece_index += 1
            if not clean_path.exists():
                run_cmd(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        f"{cursor:.4f}",
                        "-t",
                        f"{block_start - cursor:.4f}",
                        "-i",
                        source_audio,
                        "-ac",
                        "2",
                        "-ar",
                        "48000",
                        "-c:a",
                        "pcm_s16le",
                        clean_path,
                    ],
                    log,
                )
            concat_files.append(clean_path)

        concat_files.append(replacement_path)
        piece_index += 1

        manifest["cues"].append(
            {
                "block": block_index,
                "scene": scene_id,
                "target_start": block_start,
                "target_end": block_end,
                "sample_start": chosen["start"],
                "sample_end": chosen["end"],
                "fallback_scene": bool(chosen.get("fallback_scene")),
                "sample": str(sample_path),
                "replacement": str(replacement_path),
            }
        )
        cursor = block_end

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if cursor < duration - 0.001:
        clean_path = pieces_dir / f"clean_{piece_index:06d}.wav"
        if not clean_path.exists():
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{cursor:.4f}",
                    "-t",
                    f"{duration - cursor:.4f}",
                    "-i",
                    source_audio,
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-c:a",
                    "pcm_s16le",
                    clean_path,
                ],
                log,
            )
        concat_files.append(clean_path)

    if not concat_files:
        raise RuntimeError("Background reconstruction produced no audio pieces")

    concat_list = pieces_dir / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in concat_files),
        encoding="utf-8",
    )

    output = work_dir / "audio" / "background_reconstructed.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c:a",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "48000",
            output,
        ],
        log,
    )

    manifest["pieces"] = [str(p) for p in concat_files]
    manifest["output"] = str(output)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if log:
        log.info(
            "Background reconstruction ready: blocks=%d scenes=%d output=%s",
            len(blocks),
            len(scenes),
            output,
        )
    return output
