import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

IN_JSONL  = Path("data/outputs/tracks.jsonl")
OUT_JSONL = Path("data/outputs/tracks_merged.jsonl")
OUT_SUM   = Path("data/outputs/track_summary.csv")

# ====== 合并超参数（先用这套，后面再根据效果微调）======
MAX_GAP_FRAMES = 3         # 允许断几帧还算同一艘船（FPS_TARGET=1时建议 2~5）
MAX_DIST_NORM  = 0.08      # 中心点距离阈值（归一化到图像对角线）
AREA_RATIO_MIN = 0.5       # 面积相似度：A_end / B_start 在 [0.5, 2.0] 之间
AREA_RATIO_MAX = 2.0

def center_xyxy(b):
    x1,y1,x2,y2 = b
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=float)

def area_xyxy(b):
    x1,y1,x2,y2 = b
    return max(0.0, (x2-x1)) * max(0.0, (y2-y1))

def load_frames(jsonl_path: Path):
    frames = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            frames.append(json.loads(line))
    return frames

def build_tracklets(frames):
    """
    tracklet: 一个原始 track_id 在整个视频中的连续出现段
    我们记录：
      - start_i / end_i：出现的帧索引范围
      - start_bbox / end_bbox：首尾 bbox（用于合并判定）
      - length：出现帧数
    """
    by_tid = defaultdict(list)  # tid -> list of (frame_idx, bbox)
    for fi, rec in enumerate(frames):
        for it in rec.get("items", []):
            if "track_id" not in it:
                continue
            tid = int(it["track_id"])
            by_tid[tid].append((fi, it["bbox"]))

    tracklets = []
    for tid, arr in by_tid.items():
        arr.sort(key=lambda x: x[0])
        fis  = [a[0] for a in arr]
        bbs  = [a[1] for a in arr]

        tracklets.append({
            "tid": tid,
            "start_i": fis[0],
            "end_i": fis[-1],
            "start_bbox": bbs[0],
            "end_bbox": bbs[-1],
            "length": len(fis),
        })

    # 按起始时间排序，方便后面串联
    tracklets.sort(key=lambda t: (t["start_i"], t["end_i"]))
    return tracklets

def merge_tracklets(tracklets, img_diag=1.0):
    """
    贪心合并：
      对每个 tracklet B，找一个最匹配的 A（A 在 B 之前结束，且 gap<=MAX_GAP_FRAMES）
    输出：
      map_tid_to_gid: 原 tid -> global_id
      gid_tracks: global_id -> list of tids（合并链）
    """
    # 先给每个 tid 一个 gid（初始=自己）
    next_gid = 1
    tid_to_gid = {}
    gid_tail = {}  # gid -> 当前链最后一个 tracklet（用于继续接）

    # 为了快速找“可接上的 tail”，维护一个列表
    for t in tracklets:
        tid_to_gid[t["tid"]] = None

    for t in tracklets:
        best_gid = None
        best_score = None

        c_start = center_xyxy(t["start_bbox"])
        a_start = area_xyxy(t["start_bbox"])

        # 找所有已存在 gid 的 tail，看能否接到它后面
        for gid, tail in gid_tail.items():
            gap = t["start_i"] - tail["end_i"]
            if gap <= 0 or gap > MAX_GAP_FRAMES:
                continue

            c_end = center_xyxy(tail["end_bbox"])
            a_end = area_xyxy(tail["end_bbox"])

            # 距离（归一化到对角线）
            dist = np.linalg.norm(c_start - c_end) / img_diag

            # 面积相似（防止把小船接到大船上）
            if a_start <= 0 or a_end <= 0:
                continue
            ar = a_end / a_start
            if not (AREA_RATIO_MIN <= ar <= AREA_RATIO_MAX):
                continue

            if dist > MAX_DIST_NORM:
                continue

            # score：越近越好（你也可以加更多项）
            score = dist
            if best_score is None or score < best_score:
                best_score = score
                best_gid = gid

        if best_gid is None:
            # 开一个新的 global_id
            gid = next_gid
            next_gid += 1
            tid_to_gid[t["tid"]] = gid
            gid_tail[gid] = t
        else:
            # 接到 best_gid 后面
            tid_to_gid[t["tid"]] = best_gid
            gid_tail[best_gid] = t

    # 汇总 gid -> tids
    gid_tracks = defaultdict(list)
    for tid, gid in tid_to_gid.items():
        gid_tracks[gid].append(tid)

    return tid_to_gid, gid_tracks

def main():
    frames = load_frames(IN_JSONL)
    if not frames:
        raise RuntimeError("empty tracks.jsonl")

    # 尝试从第一帧估计图像对角线（如果你把宽高写进meta就更准）
    # 这里用一个“像素尺度”估计：从 bbox 里猜不出来，只能用常量。
    # 所以我们用一个经验：把 dist 阈值当作相对像素比例来用。
    # 如果你知道 frame 宽高，替换 img_diag = sqrt(w*w+h*h)
    img_diag = 1500.0  # 经验值（1280宽左右时差不多）

    tracklets = build_tracklets(frames)
    tid_to_gid, gid_tracks = merge_tracklets(tracklets, img_diag=img_diag)

    # 写 merged jsonl：在每个 item 里加入 global_id
    with OUT_JSONL.open("w", encoding="utf-8") as fo:
        for rec in frames:
            new = dict(rec)
            new_items = []
            for it in rec.get("items", []):
                it2 = dict(it)
                if "track_id" in it2:
                    tid = int(it2["track_id"])
                    it2["global_id"] = int(tid_to_gid.get(tid, -1))
                new_items.append(it2)
            new["items"] = new_items
            fo.write(json.dumps(new, ensure_ascii=False) + "\n")

    # summary：每个 global_id 的出现帧数、首次/末次出现帧
    gid_stats = defaultdict(lambda: {"frames": 0, "first_i": None, "last_i": None})
    for fi, rec in enumerate(frames):
        gids = set()
        for it in rec.get("items", []):
            if "track_id" not in it:
                continue
            tid = int(it["track_id"])
            gid = int(tid_to_gid.get(tid, -1))
            if gid >= 0:
                gids.add(gid)

        for gid in gids:
            st = gid_stats[gid]
            st["frames"] += 1
            st["first_i"] = fi if st["first_i"] is None else min(st["first_i"], fi)
            st["last_i"]  = fi if st["last_i"]  is None else max(st["last_i"],  fi)

    rows = []
    for gid, st in sorted(gid_stats.items(), key=lambda x: x[0]):
        rows.append({
            "global_id": gid,
            "frames_present": st["frames"],
            "first_frame_idx": st["first_i"],
            "last_frame_idx": st["last_i"],
            "merged_tids": ",".join(map(str, sorted(gid_tracks[gid]))),
        })

    pd.DataFrame(rows).to_csv(OUT_SUM, index=False, encoding="utf-8")
    print("[done] merged jsonl:", OUT_JSONL)
    print("[done] summary csv :", OUT_SUM)

if __name__ == "__main__":
    main()
