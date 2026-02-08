from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = Path("data/outputs/features_frame.csv")
EVENTS = Path("data/outputs/events.csv")
OUT_MD = Path("data/outputs/report.md")
OUT_P1 = Path("data/outputs/plot_ship_count.png")
OUT_P2 = Path("data/outputs/plot_area_sum.png")

def main():
    if not FEATURES.exists():
        raise FileNotFoundError("Missing features_frame.csv. Run features.py first.")

    df = pd.read_csv(FEATURES)
    ev = pd.read_csv(EVENTS) if EVENTS.exists() else pd.DataFrame()

    # Plot ship_count
    plt.figure()
    plt.plot(df["frame_idx"], df["ship_count"], alpha=0.3, label="raw")
    plt.plot(df["frame_idx"], df["ship_count_smooth"], label="smooth")
    plt.legend()

    plt.xlabel("frame_idx")
    plt.ylabel("ship_count")
    plt.title("Ship Count Over Time")
    if not ev.empty:
        xs = ev[ev["kind"] == "traffic_spike"]["frame_idx"]
        ys = df.set_index("frame_idx").loc[xs, "ship_count"].values if len(xs) else []
        plt.scatter(xs, ys)
    plt.tight_layout()
    OUT_P1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_P1, dpi=150)
    plt.close()

    # Plot area_sum
    plt.figure()

    # 原始（淡）
    plt.plot(df["frame_idx"], df["area_sum"], alpha=0.3, label="raw")

    # 平滑（主线）
    if "area_smooth" in df.columns:
        plt.plot(df["frame_idx"], df["area_smooth"], label="smooth")
    else:
        # 万一你忘了在features.py里生成area_smooth，至少不报错
        plt.plot(df["frame_idx"], df["area_sum"], label="raw_only")

    plt.xlabel("frame_idx")
    plt.ylabel("area_sum")
    plt.title("Total BBox Area Sum Over Time (proxy of big ship presence)")
    plt.legend()

    if not ev.empty:
        xs = ev[ev["kind"] == "big_ship_congestion"]["frame_idx"]
        if len(xs):
            # 点标在“平滑值”上更好看
            base_col = "area_smooth" if "area_smooth" in df.columns else "area_sum"
            ys = df.set_index("frame_idx").loc[xs, base_col].values
            plt.scatter(xs, ys)

    plt.tight_layout()
    plt.savefig(OUT_P2, dpi=150)
    plt.close()



    # Summary
    n_frames = len(df)
    total_events = 0 if ev.empty else len(ev)
    max_count = int(df["ship_count"].max()) if n_frames else 0
    max_area = float(df["area_sum"].max()) if n_frames else 0.0

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

    if not ev.empty:
        md.append("\n## Events (top 20)\n")
        md.append(ev.head(20).to_markdown(index=False))
        md.append("\n")

    md.append("\n## Plots\n")
    md.append(f"![ship_count]({OUT_P1.name})\n")
    md.append(f"![area_sum]({OUT_P2.name})\n")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("[done] report:", OUT_MD)
    print("[done] plots :", OUT_P1, OUT_P2)

if __name__ == "__main__":
    main()
