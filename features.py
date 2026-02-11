import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd

# =========================
# 入出力パス設定
# =========================
TRACKS_JSONL = Path("data/outputs/tracks.jsonl")              # 追跡結果（1行=1フレームのJSONL）
OUT_FRAME    = Path("data/outputs/features_frame.csv")        # フレーム単位の特徴量
OUT_TRACK    = Path("data/outputs/features_track.csv")        # トラック（track_id）単位の特徴量


# =========================
# 幾何ユーティリティ
# =========================
def bbox_area(b):
    """bbox（x1,y1,x2,y2）の面積を返す。座標異常時は0にクリップ。"""
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def bbox_center(b):
    """bbox（x1,y1,x2,y2）の中心点（cx,cy）を返す。"""
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def dist(p, q):
    """2点間のユークリッド距離を返す（トラック移動量算出用）。"""
    return math.hypot(p[0] - q[0], p[1] - q[1])


# =========================
# I/O
# =========================
def read_jsonl(path: Path):
    """JSONL（1行=1JSON）を読み込み、dictのリストとして返す。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    # 入力ファイル存在チェック（trackモード出力前提）
    if not TRACKS_JSONL.exists():
        raise FileNotFoundError(
            f"Missing {TRACKS_JSONL}. Please run detect_track.py with MODE='track' first."
        )

    rows = read_jsonl(TRACKS_JSONL)

    # =========================
    # 特徴量格納（フレーム / トラック）
    # =========================
    frame_feats = []  # フレーム単位の特徴量（後でDataFrame化）

    # トラック単位の生データ蓄積：
    # tid -> [(src_frame_idx, cx, cy, area, conf), ...]
    track_points = defaultdict(list)

    # tid -> [is_night(0/1), ...] （夜間比率算出用）
    track_night = defaultdict(list)

    # =========================
    # フレーム走査：per-frame集計 ＋ per-track蓄積
    # =========================
    for i, r in enumerate(rows):
        sample_idx = i

        # 旧フォーマット互換（src_frame_idxが無い場合は連番を使用）
        src_val = r.get("src_frame_idx", None)
        src_idx = int(src_val) if src_val is not None else i

        # time_sec（欠損時はNone）
        ts = r.get("time_sec", None)
        time_sec = float(ts) if ts is not None and ts != "" else None

        # 互換性：フィールド名が items / tracks どちらでも読み取れるようにする
        items = r.get("items", []) or r.get("tracks", [])
        is_night = r.get("is_night", None)

        # フレーム単位集計用
        n = 0
        areas = []
        confs = []

        for it in items:
            tid  = it.get("track_id", None)
            bbox = it.get("bbox", None)
            conf = it.get("conf", None)

            # track_idやbboxが欠けているレコードは除外
            if tid is None or bbox is None:
                continue

            n += 1
            a = bbox_area(bbox)
            c = bbox_center(bbox)

            areas.append(a)
            if conf is not None:
                confs.append(float(conf))

            # トラック単位の時系列点を蓄積（後で移動距離などを算出）
            track_points[int(tid)].append(
                (src_idx, c[0], c[1], a, float(conf) if conf is not None else None)
            )

            # 夜間フラグもトラック単位で蓄積（夜間比率）
            if is_night is not None:
                track_night[int(tid)].append(int(is_night))

        # フレーム特徴量
        frame_feats.append({
            "time_sec": time_sec,
            "sample_idx": sample_idx,
            "frame_idx": src_idx,
            "frame": r.get("frame", f"{src_idx}"),
            "is_night": is_night if is_night is not None else "",
            "ship_count": n,
            "area_sum": sum(areas) if areas else 0.0,
            "area_mean": (sum(areas) / n) if n else 0.0,
            "conf_mean": (sum(confs) / len(confs)) if confs else "",
        })

    # =========================
    # トラック集計：per-track summary
    # =========================
    track_feats = []
    for tid, pts in track_points.items():
        # src_frame_idx 昇順にソート
        pts = sorted(pts, key=lambda x: x[0])

        first_src = pts[0][0]
        last_src  = pts[-1][0]

        # 原動画フレーム番号のスパン
        duration_src_frames = last_src - first_src + 1

        # 実際に観測された抽出フレーム数
        appear_samples = len(pts)

        # トラック移動量（中心点の移動距離の総和）
        path = 0.0
        for j in range(1, len(pts)):
            p = (pts[j-1][1], pts[j-1][2])
            q = (pts[j][1], pts[j][2])
            path += dist(p, q)

        # 平均面積・平均信頼度
        area_mean = sum(p[3] for p in pts) / len(pts)
        conf_vals = [p[4] for p in pts if p[4] is not None]
        conf_mean = sum(conf_vals) / len(conf_vals) if conf_vals else ""

        # 夜間比率（0〜1）
        night_ratio = ""
        if tid in track_night and track_night[tid]:
            night_ratio = sum(track_night[tid]) / len(track_night[tid])

        # 参考：平均速度（px / 抽出フレーム）
        avg_speed_px_per_sample = ""
        if appear_samples >= 2:
            avg_speed_px_per_sample = path / (appear_samples - 1)

        track_feats.append({
            "track_id": tid,
            "first_frame_idx": first_src,
            "last_frame_idx": last_src,
            "duration_src_frames": duration_src_frames,
            "appear_samples": appear_samples,
            "path_len_px": path,
            "avg_speed_px_per_sample": avg_speed_px_per_sample,
            "area_mean": area_mean,
            "conf_mean": conf_mean,
            "night_ratio": night_ratio,
        })

    # =========================
    # DataFrame化 + 平滑化
    # =========================
    df_f = pd.DataFrame(frame_feats)

    # 安全のため連番順に並べる
    df_f = df_f.sort_values("sample_idx").reset_index(drop=True)

    # 移動平均（中心5フレーム）で船数・面積を平滑化
    df_f["ship_count_smooth"] = df_f["ship_count"].rolling(5, center=True, min_periods=1).mean()
    df_f["area_smooth"] = df_f["area_sum"].rolling(5, center=True, min_periods=1).mean()

    # トラック特徴量（観測点数が多い順）
    df_t = pd.DataFrame(track_feats).sort_values("appear_samples", ascending=False)

    # =========================
    # CSV出力
    # =========================
    OUT_FRAME.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(OUT_FRAME, index=False, encoding="utf-8")
    df_t.to_csv(OUT_TRACK, index=False, encoding="utf-8")

    print("[done] frame features:", OUT_FRAME)
    print("[done] track features:", OUT_TRACK)
    print("[info] tracks:", len(df_t), "frames:", len(df_f))


if __name__ == "__main__":
    main()
