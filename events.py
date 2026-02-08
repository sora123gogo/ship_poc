import json
from pathlib import Path
import pandas as pd

IN_FEATURES = Path("data/outputs/features_frame.csv")
OUT_CSV = Path("data/outputs/events.csv")
OUT_JSON = Path("data/outputs/events.json")

WINDOW = 30        # rolling窗口：30帧（你是1fps的话就是30秒）
Z_TH = 2.5         # 阈值：越大越“保守”
MIN_GAP = 10       # 事件最小间隔（帧数），避免连续报警

def rolling_z(df: pd.Series, w: int):
    m = df.rolling(w, min_periods=max(5, w//3)).mean()
    s = df.rolling(w, min_periods=max(5, w//3)).std()
    z = (df - m) / (s + 1e-9)
    return z

def pick_events(df, score_col, kind):
    events = []
    last_idx = -10**9
    for _, r in df[df[score_col] >= Z_TH].iterrows():
        idx = int(r["frame_idx"])
        if idx - last_idx < MIN_GAP:
            continue
        last_idx = idx
        events.append({
            "kind": kind,
            "frame_idx": idx,
            "frame": r.get("frame", ""),
            "score": float(r[score_col]),
            "ship_count": int(r["ship_count"]),
            "area_sum": float(r["area_sum"]),
            "is_night": r.get("is_night", ""),
        })
    return events

def main():
    if not IN_FEATURES.exists():
        raise FileNotFoundError(f"Missing {IN_FEATURES}. Run features.py first.")

    df = pd.read_csv(IN_FEATURES)
    df["ship_count"] = df["ship_count"].fillna(0).astype(int)
    df["area_sum"] = df["area_sum"].fillna(0.0).astype(float)

    df["z_ship_count"] = rolling_z(df["ship_count"], WINDOW)
    df["z_area_sum"] = rolling_z(df["area_sum"], WINDOW)

    ev = []
    ev += pick_events(df, "z_ship_count", "traffic_spike")
    ev += pick_events(df, "z_area_sum", "big_ship_congestion")

    ev = sorted(ev, key=lambda x: x["frame_idx"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev).to_csv(OUT_CSV, index=False, encoding="utf-8")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"events": ev, "params": {"window": WINDOW, "z_th": Z_TH, "min_gap": MIN_GAP}}, f, ensure_ascii=False, indent=2)

    print("[done] events csv:", OUT_CSV)
    print("[done] events json:", OUT_JSON)
    print("[info] events:", len(ev))

if __name__ == "__main__":
    main()
