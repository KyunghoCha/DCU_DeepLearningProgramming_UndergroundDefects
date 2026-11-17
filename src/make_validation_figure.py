"""Draw the registered validation comparison from the bundled numeric table."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]


def main():
    font = next((path for path in (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
                 ROOT / "assets/fonts/nanum/NanumGothic.ttf") if path.is_file()), None)
    if font is not None:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False})
    with (ROOT / "results/validation_comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 90
    fig, ax = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    for checkpoints, marker, label in [(False, "o", "학습 중 선택 가중치"), (True, "^", "고정 간격 에폭 가중치")]:
        items = [r for r in rows if ("_epoch" in r["model"]) == checkpoints]
        scatter = ax.scatter([float(r["macro_f1"])*100 for r in items],
                             [float(r["mask_map50"])*100 for r in items],
                             c=[float(r["mask_map50_95"])*100 for r in items],
                             cmap="viridis", vmin=10, vmax=35, marker=marker,
                             s=65, alpha=.8, edgecolors="white", linewidths=.5, label=label)
    chosen = next(r for r in rows if r["selected"] == "True")
    x, y = float(chosen["macro_f1"])*100, float(chosen["mask_map50"])*100
    ax.scatter([x], [y], marker="*", s=270, facecolor="#e24a33", edgecolor="black", zorder=5, label="최종 선택 K · 15에폭 / 1280")
    ax.annotate(f"선택 모델\nF1 {x:.2f}% · Mask mAP50 {y:.2f}%", (x,y),
                xytext=(x-14,y+6), arrowprops={"arrowstyle":"->","color":"#26384a"}, fontsize=11)
    ax.axvline(89.6, color="#56616a", linestyle="--", linewidth=1.2)
    ax.text(89.6+.35, ax.get_ylim()[0]+1, "참고 F1 89.6%", rotation=90, va="bottom", color="#56616a")
    ax.set(xlabel="검증 영상 단위 macro F1 (%)", ylabel="검증 Mask mAP50 (%)", xlim=(58,94), ylim=(25,86))
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", fontsize=10)
    cbar = fig.colorbar(scatter, ax=ax, pad=.03)
    cbar.set_label("Mask mAP50–95 (%)")
    target = ROOT / "docs/images/validation_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
