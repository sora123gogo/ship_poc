import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd

TRACKS_JSONL = Path("data/outputs/tracks.jsonl")
OUT_FRAME = Path("data/outputs/features_frame.csv")
OUT_TRACK = Path("data/outputs/features_track.csv")

# 如果你只想统计“大船”，可以再加面积过滤（相对画面）。这里先不做，保持通用。


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def bbox_center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    if not TRACKS_JSONL.exists():
        raise FileNotFoundError(f"Missing {TRACKS_JSONL}. Please run detect_track.py with MODE='track' first.")

    rows = read_jsonl(TRACKS_JSONL)

    # per-frame
    frame_feats = []
    # per-track accum
    track_points = defaultdict(list)   # tid -> list of (frame_idx, cx, cy, area, conf)
    track_night = defaultdict(list)    # tid -> list of is_night flags if provided

    for fi, r in enumerate(rows):
        items = r.get("items", []) or r.get("tracks", [])  # 兼容你旧版 tracks 字段名
        is_night = r.get("is_night", None)

        n = 0
        areas = []
        confs = []
        for it in items:
            tid = it.get("track_id", None)
            bbox = it.get("bbox", None)
            conf = it.get("conf", None)
            if tid is None or bbox is None:
                continue
            n += 1
            a = bbox_area(bbox)
            c = bbox_center(bbox)
            areas.append(a)
            if conf is not None:
                confs.append(float(conf))
            track_points[int(tid)].append((fi, c[0], c[1], a, float(conf) if conf is not None else None))
            if is_night is not None:
                track_night[int(tid)].append(int(is_night))

        frame_feats.append({
            "frame_idx": fi,
            "frame": r.get("frame", f"{fi}"),
            "is_night": is_night if is_night is not None else "",
            "ship_count": n,
            "area_sum": sum(areas) if areas else 0.0,
            "area_mean": (sum(areas) / n) if n else 0.0,
            "conf_mean": (sum(confs) / len(confs)) if confs else "",
        })

    # per-track summary
    track_feats = []
    for tid, pts in track_points.items():
        pts = sorted(pts, key=lambda x: x[0])
        # duration in frames
        first_f = pts[0][0]
        last_f = pts[-1][0]
        duration_frames = last_f - first_f + 1

        # path length (sum of centroid displacements)
        path = 0.0
        for i in range(1, len(pts)):
            p = (pts[i-1][1], pts[i-1][2])
            q = (pts[i][1], pts[i][2])
            path += dist(p, q)

        area_mean = sum(p[3] for p in pts) / len(pts)
        conf_vals = [p[4] for p in pts if p[4] is not None]
        conf_mean = sum(conf_vals) / len(conf_vals) if conf_vals else ""

        night_ratio = ""
        if tid in track_night and track_night[tid]:
            night_ratio = sum(track_night[tid]) / len(track_night[tid])

        track_feats.append({
            "track_id": tid,
            "first_frame_idx": first_f,
            "last_frame_idx": last_f,
            "duration_frames": duration_frames,
            "path_len_px": path,
            "area_mean": area_mean,
            "conf_mean": conf_mean,
            "night_ratio": night_ratio,
        })

    df_f = pd.DataFrame(frame_feats)
    # === 时间平滑（去抖） ===
    df_f["ship_count_smooth"] = (
        df_f["ship_count"]
        .rolling(5, center=True, min_periods=1)
        .mean()
    )

    df_f["area_smooth"] = (
        df_f["area_sum"]
        .rolling(5, center=True, min_periods=1)
        .mean()
    )

    # ========================

    df_t = pd.DataFrame(track_feats).sort_values("duration_frames", ascending=False)



    OUT_FRAME.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(OUT_FRAME, index=False, encoding="utf-8")
    df_t.to_csv(OUT_TRACK, index=False, encoding="utf-8")

    print("[done] frame features:", OUT_FRAME)
    print("[done] track features:", OUT_TRACK)
    print("[info] tracks:", len(df_t), "frames:", len(df_f))

if __name__ == "__main__":
    main()
