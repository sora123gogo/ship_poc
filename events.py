import json
from pathlib import Path
import pandas as pd

# =========================
# 入出力パス設定
# =========================
IN_FEATURES = Path("data/outputs/features_frame.csv")   # フレーム特徴量（features.pyの出力）
OUT_CSV     = Path("data/outputs/events.csv")           # 抽出イベント一覧（CSV）
OUT_JSON    = Path("data/outputs/events.json")          # 抽出イベント一覧＋パラメータ（JSON）

# =========================
# パラメータ（※ sample_idx 基準：抽出フレーム数）
# =========================
WINDOW  = 30   # rolling窓幅（抽出フレーム数）。例：FPS_TARGET=3なら30点=約10秒
Z_TH    = 2.5  # zスコア閾値。大きいほど検出が保守的（誤検知↓/見逃し↑）
MIN_GAP = 10   # 同種イベントの最小間隔（抽出フレーム数）。連続検出の抑制用

# =========================
# rolling z-score 計算
# =========================
def rolling_z(s: pd.Series, w: int):
    """
    rolling平均・rolling標準偏差からzスコアを計算する。
    - min_periods を小さくしすぎると初期区間で不安定になるため、
      max(5, w//3) を下限として設定。
    """
    m = s.rolling(w, min_periods=max(5, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(5, w // 3)).std()
    z = (s - m) / (sd + 1e-9)  # 0除算回避（std=0対策）
    return z

# =========================
# イベント抽出（閾値超えのピークを間引き）
#   - 判定/間引きは sample_idx 基準（連続）
#   - 出力の frame_idx は原動画フレーム番号（突合用）
# =========================
def pick_events(df: pd.DataFrame, score_col: str, kind: str):
    events = []
    last_sample = -10**9

    # 閾値超えの行のみ走査（sample_idxで時系列順に）
    for _, r in df[df[score_col] >= Z_TH].sort_values("sample_idx").iterrows():
        sample = int(r["sample_idx"])

        # 直前イベントから近すぎる場合は捨てる（sample_idx基準）
        if sample - last_sample < MIN_GAP:
            continue
        last_sample = sample

        # time_sec（あればfloat、なければNone）
        ts = r.get("time_sec", None)
        time_sec = float(ts) if ts is not None and pd.notna(ts) else None

        # 出力は frame_idx（原動画フレーム番号）を保持（回放/突合に便利）
        events.append({
            "kind": kind,
            "sample_idx": sample,                 # 連続軸（解析用）
            "frame_idx": int(r["frame_idx"]),     # 原動画フレーム番号（突合用）
            "time_sec": time_sec,                 # 実時間（秒）。存在しない場合はNone
            "frame": r.get("frame", ""),
            "score": float(r[score_col]),
            "ship_count": int(r["ship_count"]),
            "area_sum": float(r["area_sum"]),
            "is_night": r.get("is_night", ""),
        })

    return events

def main():
    # 入力ファイル存在チェック
    if not IN_FEATURES.exists():
        raise FileNotFoundError(f"Missing {IN_FEATURES}. Run features.py first.")

    # 特徴量CSVの読み込み
    df = pd.read_csv(IN_FEATURES)

    # 必須列チェック（features.py の新仕様）
    if "sample_idx" not in df.columns:
        raise ValueError("features_frame.csv に sample_idx 列がありません。features.py を更新して再生成してください。")
    if "frame_idx" not in df.columns:
        raise ValueError("features_frame.csv に frame_idx 列がありません。features.py を更新して再生成してください。")

    # 欠損対策（安全側に0埋め）
    df["ship_count"] = df["ship_count"].fillna(0).astype(int)
    df["area_sum"]   = df["area_sum"].fillna(0.0).astype(float)

    # 並び順を保証（rollingは行順依存）
    df = df.sort_values("sample_idx").reset_index(drop=True)

    # =========================
    # スコア算出：rolling z-score（sample_idx基準）
    # =========================
    df["z_ship_count"] = rolling_z(df["ship_count"], WINDOW)
    df["z_area_sum"]   = rolling_z(df["area_sum"], WINDOW)

    # =========================
    # イベント抽出
    # =========================
    ev = []
    ev += pick_events(df, "z_ship_count", "traffic_spike")
    ev += pick_events(df, "z_area_sum",   "big_ship_congestion")

    # 出力は原動画フレーム番号でソート（人間が追いやすい）
    ev = sorted(ev, key=lambda x: x["frame_idx"])

    # =========================
    # 出力（CSV / JSON）
    # =========================
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev).to_csv(OUT_CSV, index=False, encoding="utf-8")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"events": ev, "params": {"window": WINDOW, "z_th": Z_TH, "min_gap": MIN_GAP}},
            f,
            ensure_ascii=False,
            indent=2
        )

    print("[done] events csv:", OUT_CSV)
    print("[done] events json:", OUT_JSON)
    print("[info] events:", len(ev))

if __name__ == "__main__":
    main()
