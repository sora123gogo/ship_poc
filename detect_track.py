from ultralytics import YOLO
import supervision as sv
import cv2, os, glob, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# =========================
# ===== 配置区 =====
# =========================
# detect: for detect
# track : track（get track_id）
MODE = "track"   # "detect" or "track"
VIDEO_NAME = "Busiest Day so far - 8 Ships in Port - LIVE Replay [u62fnabcShI].mp4"

FPS_TARGET = 3                  # fps（frames）for 1s
CLEAR_FRAMES_IF_EXISTS = False  # True: 每次运行都重抽；False: frames有就复用

# day and night para
DAY_CFG = {"imgsz": 1280, "conf": 0.25}
NIGHT_CFG = {"imgsz": 1280, "conf": 0.03}

# 夜晚判定阈值（gray < 阈值 => night）
NIGHT_THRESH = 60
NIGHT_CHECK_EVERY = 10  # 一定フレーム数ごとに昼夜判定を再実行し、判定の揺らぎ（チャタリング）を抑制

# 面積フィルタリング
USE_AREA_FILTER = True
AREA_RATIO = 0.01  # bbox面積 / 画面面積 > 1% hold

# 出力可視化動画のFPSは再生速度のみに影響し、解析処理自体には影響しない
OUT_FPS = 10

# ===== 可視化平滑処理は表示用動画のみに適用し、解析データ自体には影響しない=====
ENABLE_VIZ_SMOOTH = True  # True：可視化時に短時間のHOLDを許可 / False：完全に実測値を表示（ちらつきの可能性あり）
KEEP_ZERO = 2             # 最大HOLDフレーム数
prev_det_vis = None       # 可視化専用の直前検出結果
zero_keep = 0             #HOLD用カウンタ

# =========================
# path
# =========================
FRAMES_DIR = Path("data/frames")
RAW_VIDEO = Path("data/raw") / VIDEO_NAME

OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIZ = Path("data/viz")
OUT_VIZ.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / ("tracks.jsonl" if MODE == "track" else "dets.jsonl")
OUT_MP4   = OUT_VIZ / ("tracked.mp4" if MODE == "track" else "detected.mp4")
OUT_META  = OUT_DIR / "frame_meta.csv"


def enhance_night(frame):
    """夜画像強化"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def gray_mean(frame):
    """グレースケール平均値算出"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def decide_night(mean_val, thresh=NIGHT_THRESH):
    """ 昼夜判定"""
    return mean_val < thresh


def ensure_frames():
    """フレームが存在しない場合に動画から抽出"""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_FRAMES_IF_EXISTS:
        for p in FRAMES_DIR.glob("*.jpg"):
            p.unlink()

    existing = list(FRAMES_DIR.glob("*.jpg"))
    if existing:
        print(f"[frames] reuse existing frames: {len(existing)}")
        return

    if not RAW_VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {RAW_VIDEO}")

    cap = cv2.VideoCapture(str(RAW_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {RAW_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps / FPS_TARGET)))

    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(FRAMES_DIR / f"frame_{idx:07d}.jpg"), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[extract] saved frames: {saved}")


def apply_area_filter(det: sv.Detections, frame_shape, ratio=AREA_RATIO):
    """面積フィルタ処理"""
    if len(det) == 0:
        return det

    h, w = frame_shape[:2]
    frame_area = float(h * w)

    x1 = det.xyxy[:, 0]
    y1 = det.xyxy[:, 1]
    x2 = det.xyxy[:, 2]
    y2 = det.xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = areas > (ratio * frame_area)
    return det[keep]


def main():
    global prev_det_vis, zero_keep

    ensure_frames()
    frame_paths = sorted(glob.glob(str(FRAMES_DIR / "*.jpg")))
    if not frame_paths:
        raise RuntimeError("No frames found in data/frames")

    # YOLOモデル（COCO汎用）
    model = YOLO("yolov8s.pt")

    # トラッカー（trackモードのみ）
    tracker = sv.ByteTrack() if MODE == "track" else None

    # 出力動画初期化
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError("Failed to read first frame.")
    h, w = first.shape[:2]

    vw = cv2.VideoWriter(
        str(OUT_MP4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUT_FPS,
        (w, h),
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    meta_rows = []
    night_flag = False  # 昼夜状態キャッシュ（防抖）

    with OUT_JSONL.open("w", encoding="utf-8") as f_json:
        for i, fp in enumerate(tqdm(frame_paths)):
            frame = cv2.imread(fp)
            if frame is None:
                continue

            # ====== 昼夜判定（防抖処理）======
            if i % NIGHT_CHECK_EVERY == 0:
                m = gray_mean(frame)
                night_flag = decide_night(m)
            else:
                m = None

            meta_rows.append({
                "frame": os.path.basename(fp),
                "is_night": int(night_flag),
                "gray_mean": m if m is not None else ""
            })

            # ====== 推論設定切替======
            if night_flag:
                frame_infer = enhance_night(frame)
                cfg = NIGHT_CFG
            else:
                frame_infer = frame
                cfg = DAY_CFG

            # ====== YOLO推論 ======
            res = model.predict(
                frame_infer,
                imgsz=cfg["imgsz"],
                conf=cfg["conf"],
                verbose=False
            )[0]

            det = sv.Detections.from_ultralytics(res)

            # ====== 面積フィルタ======
            if USE_AREA_FILTER:
                det = apply_area_filter(det, frame.shape, ratio=AREA_RATIO)

            # ====== トラッキング======
            if MODE == "track":
                det = tracker.update_with_detections(det)

            # ====== データ用 / 可視化用 分離======
            # det_data：真实结果（用于 jsonl/统计）
            det_data = det

            # det_vis：只用于画框（可选平滑）
            hold_flag = False
            if ENABLE_VIZ_SMOOTH and len(det_data) == 0 and prev_det_vis is not None and zero_keep < KEEP_ZERO:
                det_vis = prev_det_vis
                zero_keep += 1
                hold_flag = True
            else:
                det_vis = det_data
                zero_keep = 0
                prev_det_vis = det_vis

            # ======  JSONL出力（常に det_data 使用）======
            rec = {"frame": os.path.basename(fp), "is_night": int(night_flag), "items": []}

            if MODE == "track":
                # tracker_id 可能为 None（没分配到轨迹），就跳过
                for xyxy, tid, conf in zip(det_data.xyxy, det_data.tracker_id, det_data.confidence):
                    if tid is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    rec["items"].append({
                        "track_id": int(tid),
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf)
                    })
            else:
                for xyxy, conf in zip(det_data.xyxy, det_data.confidence):
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    rec["items"].append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf)
                    })

            f_json.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ====== 可視化描画（常に det_vis 使用）======
            if MODE == "track":
                labels = [f"id:{int(tid)}" if tid is not None else "" for tid in det_vis.tracker_id]
            else:
                labels = [f"{c:.2f}" for c in det_vis.confidence]

            annotated = box_annotator.annotate(frame.copy(), det_vis)
            annotated = label_annotator.annotate(annotated, det_vis, labels=labels)

            # 表示タグ（DAY/NIGHT + RAW/HOLD）
            tag1 = "NIGHT" if night_flag else "DAY"
            tag2 = "HOLD" if hold_flag else "RAW"

            cv2.putText(annotated, tag1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, tag2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            vw.write(annotated)

    vw.release()

    # save meta csv
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False, encoding="utf-8")

    print(f"[done] saved jsonl: {OUT_JSONL}")
    print(f"[done] saved video: {OUT_MP4}")
    print(f"[done] saved meta:  {OUT_META}")


if __name__ == "__main__":
    main()
