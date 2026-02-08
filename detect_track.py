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
# detect: 只做检测（每帧独立）
# track : 检测 + 跟踪（分配 track_id）
MODE = "track"   # "detect" or "track"
VIDEO_NAME = "Busiest Day so far - 8 Ships in Port - LIVE Replay [u62fnabcShI].mp4"

FPS_TARGET = 3                  # 抽帧频率：1fps（frames已有则不抽）
CLEAR_FRAMES_IF_EXISTS = False  # True: 每次运行都重抽；False: frames有就复用

# 白天/夜晚推理参数（建议你后面把 DAY conf 降到 0.2~0.25 试试，会少漏检）
DAY_CFG = {"imgsz": 1280, "conf": 0.25}
NIGHT_CFG = {"imgsz": 1280, "conf": 0.03}

# 夜晚判定阈值（灰度均值 < 阈值 => night）
NIGHT_THRESH = 60
NIGHT_CHECK_EVERY = 10  # 每 N 帧重新判断一次昼夜（防抖）

# 面积过滤：只保留“大目标”（更像大船）
USE_AREA_FILTER = True
AREA_RATIO = 0.01  # bbox面积 / 画面面积 > 1% 才保留

# 输出可视化视频fps（只影响输出播放速度）
OUT_FPS = 10

# ===== 可视化平滑（仅影响视频，不影响数据）=====
ENABLE_VIZ_SMOOTH = True  # True: 视频中允许短暂 HOLD；False: 完全真实（可能闪）
KEEP_ZERO = 2             # 最多 HOLD 几帧
prev_det_vis = None       # 只给可视化用
zero_keep = 0

# =========================
# 路径区
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
    """夜间轻量增强：CLAHE 提亮+提对比（PoC友好）"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def gray_mean(frame):
    """灰度均值：用于判断昼夜"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def decide_night(mean_val, thresh=NIGHT_THRESH):
    """mean_val 越小越暗；小于阈值认为 night"""
    return mean_val < thresh


def ensure_frames():
    """保证 frames 目录里有抽帧 jpg；没有就从视频抽（按 FPS_TARGET）"""
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
    """按 bbox 面积过滤小目标（保留大船更稳）"""
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

    # 模型（COCO通用）
    model = YOLO("yolov8s.pt")

    # 跟踪器（仅 track 模式）
    tracker = sv.ByteTrack() if MODE == "track" else None

    # 初始化输出视频 writer
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
    night_flag = False  # 防抖后的昼夜状态缓存

    with OUT_JSONL.open("w", encoding="utf-8") as f_json:
        for i, fp in enumerate(tqdm(frame_paths)):
            frame = cv2.imread(fp)
            if frame is None:
                continue

            # ====== 昼夜判断（防抖）======
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

            # ====== 选择推理输入/参数 ======
            if night_flag:
                frame_infer = enhance_night(frame)
                cfg = NIGHT_CFG
            else:
                frame_infer = frame
                cfg = DAY_CFG

            # ====== YOLO 推理 ======
            res = model.predict(
                frame_infer,
                imgsz=cfg["imgsz"],
                conf=cfg["conf"],
                verbose=False
            )[0]

            det = sv.Detections.from_ultralytics(res)

            # ====== 面积过滤（可选）======
            if USE_AREA_FILTER:
                det = apply_area_filter(det, frame.shape, ratio=AREA_RATIO)

            # ====== 跟踪（可选）======
            if MODE == "track":
                det = tracker.update_with_detections(det)

            # ====== det_data / det_vis 分离 ======
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

            # ====== 写 jsonl（永远写 det_data）======
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

            # ====== 可视化（永远画 det_vis）======
            if MODE == "track":
                labels = [f"id:{int(tid)}" if tid is not None else "" for tid in det_vis.tracker_id]
            else:
                labels = [f"{c:.2f}" for c in det_vis.confidence]

            annotated = box_annotator.annotate(frame.copy(), det_vis)
            annotated = label_annotator.annotate(annotated, det_vis, labels=labels)

            # 角标：DAY/NIGHT + RAW/HOLD（HOLD=仅视频平滑）
            tag1 = "NIGHT" if night_flag else "DAY"
            tag2 = "HOLD" if hold_flag else "RAW"

            cv2.putText(annotated, tag1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, tag2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            vw.write(annotated)

    vw.release()

    # 保存 meta csv
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False, encoding="utf-8")

    print(f"[done] saved jsonl: {OUT_JSONL}")
    print(f"[done] saved video: {OUT_MP4}")
    print(f"[done] saved meta:  {OUT_META}")


if __name__ == "__main__":
    main()
