from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 入出力パス設定
# =========================
FEATURES = Path("data/outputs/features_frame.csv")  # フレーム特徴量（船数・面積など）
EVENTS   = Path("data/outputs/events.csv")          # 検出イベント（任意：存在しない場合もある）
OUT_MD   = Path("data/outputs/report.md")           # 簡易レポート（Markdown）
OUT_P1   = Path("data/outputs/plot_ship_count.png") # 船数推移プロット
OUT_P2   = Path("data/outputs/plot_area_sum.png")   # 面積総和推移プロット（大型船/混雑の代理指標）

def main():
    # 入力ファイル存在チェック
    if not FEATURES.exists():
        raise FileNotFoundError("Missing features_frame.csv. Run features.py first.")

    # データ読み込み
    df = pd.read_csv(FEATURES)
    ev = pd.read_csv(EVENTS) if EVENTS.exists() else pd.DataFrame()

    # 必須列チェック（新仕様：sample_idx を前提）
    if "sample_idx" not in df.columns:
        raise ValueError("features_frame.csv に sample_idx 列がありません。features.py を更新して再生成してください。")

    # 並び順を保証（rolling/可視化の安定化）
    df = df.sort_values("sample_idx").reset_index(drop=True)

    # x軸は time_sec を優先（なければ sample_idx）
    use_time = ("time_sec" in df.columns) and df["time_sec"].notna().any()
    x = df["time_sec"] if use_time else df["sample_idx"]
    x_label = "time_sec" if use_time else "sample_idx"

    # 速度優先で index を用意（イベント点の突合用）
    df_by_sample = df.set_index("sample_idx", drop=False)
    df_by_frame  = df.set_index("frame_idx", drop=False) if "frame_idx" in df.columns else None
    df_by_time   = df.set_index("time_sec",  drop=False) if use_time else None

    # =========================
    # Plot 1: ship_count（船数推移）
    # =========================
    plt.figure()

    # 生値（薄く）
    plt.plot(x, df["ship_count"], alpha=0.3, label="raw")

    # 平滑化（主線）
    if "ship_count_smooth" in df.columns:
        plt.plot(x, df["ship_count_smooth"], label="smooth")
    else:
        plt.plot(x, df["ship_count"], label="raw_only")

    plt.xlabel(x_label)
    plt.ylabel("ship_count")
    plt.title("Ship Count Over Time")
    plt.legend()

    # イベント（traffic_spike）を点で重ねる
    if not ev.empty:
        ev_spike = ev[ev["kind"] == "traffic_spike"]

        if use_time and ("time_sec" in ev_spike.columns) and df_by_time is not None:
            xs = ev_spike["time_sec"].dropna().astype(float)
            if len(xs):
                # time_sec基準で突合（同一秒の重複がある場合は注意）
                ys = df_by_time.loc[xs, "ship_count"].values
                plt.scatter(xs, ys)
        elif "sample_idx" in ev_spike.columns:
            xs = ev_spike["sample_idx"].dropna().astype(int)
            if len(xs):
                ys = df_by_sample.loc[xs, "ship_count"].values
                plt.scatter(df_by_sample.loc[xs, x_label].values if use_time else xs, ys)
        elif ("frame_idx" in ev_spike.columns) and (df_by_frame is not None):
            xs = ev_spike["frame_idx"].dropna().astype(int)
            if len(xs):
                ys = df_by_frame.loc[xs, "ship_count"].values
                # x軸がtime_secならtime_secへ、そうでなければsample_idxへプロット
                plt.scatter(df_by_frame.loc[xs, x_label].values if use_time else df_by_frame.loc[xs, "sample_idx"].values, ys)

    plt.tight_layout()
    OUT_P1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_P1, dpi=150)
    plt.close()

    # =========================
    # Plot 2: area_sum（bbox面積総和推移）
    #  - 面積総和は大型船の出現/密集の代理指標として利用
    # =========================
    plt.figure()

    # 生値（薄く）
    plt.plot(x, df["area_sum"], alpha=0.3, label="raw")

    # 平滑化（主線）
    base_col = "area_smooth" if "area_smooth" in df.columns else "area_sum"
    plt.plot(x, df[base_col], label="smooth" if base_col == "area_smooth" else "raw_only")

    plt.xlabel(x_label)
    plt.ylabel("area_sum")
    plt.title("Total BBox Area Sum Over Time (proxy of big ship presence)")
    plt.legend()

    # イベント（big_ship_congestion）を点で重ねる
    if not ev.empty:
        ev_big = ev[ev["kind"] == "big_ship_congestion"]

        if use_time and ("time_sec" in ev_big.columns) and df_by_time is not None:
            xs = ev_big["time_sec"].dropna().astype(float)
            if len(xs):
                ys = df_by_time.loc[xs, base_col].values
                plt.scatter(xs, ys)
        elif "sample_idx" in ev_big.columns:
            xs = ev_big["sample_idx"].dropna().astype(int)
            if len(xs):
                ys = df_by_sample.loc[xs, base_col].values
                plt.scatter(df_by_sample.loc[xs, x_label].values if use_time else xs, ys)
        elif ("frame_idx" in ev_big.columns) and (df_by_frame is not None):
            xs = ev_big["frame_idx"].dropna().astype(int)
            if len(xs):
                ys = df_by_frame.loc[xs, base_col].values
                plt.scatter(df_by_frame.loc[xs, x_label].values if use_time else df_by_frame.loc[xs, "sample_idx"].values, ys)

    plt.tight_layout()
    plt.savefig(OUT_P2, dpi=150)
    plt.close()

    # =========================
    # Summary（簡易統計＋Markdown生成）
    # =========================
    n_frames = len(df)
    total_events = 0 if ev.empty else len(ev)
    max_count = int(df["ship_count"].max()) if n_frames else 0
    max_area  = float(df["area_sum"].max()) if n_frames else 0.0

    md = []
    md.append("# Ship PoC Report\n")

    md.append("## Outputs\n")
    md.append(f"- Frame features: `{FEATURES}`\n")
    md.append(f"- Events: `{EVENTS}`\n" if EVENTS.exists() else "- Events: (not generated)\n")
    md.append(f"- Plots: `{OUT_P1.name}`, `{OUT_P2.name}`\n")

    md.append("\n## Quick Stats\n")
    md.append(f"- frames: **{n_frames}**\n")
    md.append(f"- max ship_count: **{max_count}**\n")
    md.append(f"- max area_sum: **{max_area:.2f}**\n")
    md.append(f"- events detected: **{total_events}**\n")

    # イベント一覧（先頭20件のみ）
    if not ev.empty:
        md.append("\n## Events (top 20)\n")
        show_cols = ["kind"]
        if "time_sec" in ev.columns:
            show_cols += ["time_sec"]
        if "sample_idx" in ev.columns:
            show_cols += ["sample_idx"]
        show_cols += ["frame_idx", "score", "ship_count", "area_sum", "is_night"]
        show_cols = [c for c in show_cols if c in ev.columns]
        md.append(ev[show_cols].head(20).to_markdown(index=False))
        md.append("\n")

    # プロット画像をMarkdownに埋め込む
    md.append("\n## Plots\n")
    md.append(f"![ship_count]({OUT_P1.name})\n")
    md.append(f"![area_sum]({OUT_P2.name})\n")

    # Markdown出力
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("[done] report:", OUT_MD)
    print("[done] plots :", OUT_P1, OUT_P2)

if __name__ == "__main__":
    main()
