"""Rebuild three experiment figures from the small, bundled evidence tables.

python src/make_experiment_figures.py
python src/make_experiment_figures.py --source-root /path/to/course-materials

The optional collection step reads week02/week03; it never changes those sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("A", "ft960_lr3e5_e15", "기초 모델 추가 학습", "기초"),
    ("B", "ft1280_lr3e5_e15", "입력 크기 1280", "기초"),
    ("C", "cap1_1280_lr3e5_e30", "원본 ID당 최대 1장", "기초"),
    ("D", "cap4_1280_lr3e5_e30", "원본 ID당 최대 4장", "기초"),
    ("E", "coco_cap4_1280_e100", "COCO 시작 · 최대 4장", "COCO"),
    ("F", "manualaug_1280_lr1e5_e10", "직접 주석의 새 증강 138장", "B"),
    ("G", "coco_full960_mr2_e100", "COCO 시작 · 전체 학습 자료", "COCO"),
    ("H", "cont960_lr15e6_e30", "균일 표본 추출로 추가 학습", "G"),
    ("I", "balanced960_lr15e6_e30", "클래스·원본별 균등 추출", "G"),
    ("J", "weakaug960_lr15e6_e30", "색상·이동·크기 증강 완화", "G"),
    ("K", "frozenbn960_lr15e6_e30", "BatchNorm 이동 통계 고정", "G"),
    ("L", "cls960_lr15e6_e30", "분류 손실 가중치 1.0", "G"),
    ("M", "coco_full1280_mr2_e100", "COCO 시작 · 입력 크기 1280", "COCO"),
]
PARAMS = ("imgsz", "batch", "epochs", "patience", "optimizer", "lr0", "lrf",
          "warmup_epochs", "warmup_bias_lr", "momentum", "weight_decay", "mask_ratio",
          "cls", "box", "dfl", "nbs", "hsv_h", "hsv_s", "hsv_v", "translate", "scale",
          "fliplr", "flipud", "degrees", "mosaic", "mixup", "seed", "workers", "amp")
METRIC_COLUMNS = {"train_seg_loss": "train/seg_loss", "val_seg_loss": "val/seg_loss",
                  "mask_map50": "metrics/mAP50(M)", "mask_map50_95": "metrics/mAP50-95(M)",
                  "box_map50_95": "metrics/mAP50-95(B)", "lr": "lr/pg0"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(source, out):
    """Collect explicit fields only; server paths and private launch records are omitted."""
    import yaml
    used = {}

    def track(relative):
        p = source / relative
        if not p.is_file():
            raise FileNotFoundError(p)
        used[relative] = digest(p)
        return p

    train_rows, epoch_rows = [], []
    base = "week02/results/lecture60"
    baseline_done = read_json(track(f"{base}/completion.json"))
    baseline_best = read_json(track(f"{base}/best_checkpoint_selection.json"))
    baseline_args = yaml.safe_load(track(f"{base}/args.yaml").read_text())
    native_records = [("기초", "lecture60", "2주차 기초 학습", "COCO", base,
                       {"baseline": {"expected_counts": {"train": 3087}}, "sampling": "uniform"},
                       baseline_args, baseline_done, None, baseline_best)]
    for tag, name, title, parent in RUNS:
        rel = f"week03/experiments/runs/{name}"
        cfg = read_json(track(f"{rel}/experiment_config.json"))
        args = read_json(track(f"{rel}/normalized_args.json"))
        done = read_json(track(f"{rel}/completed.json"))
        ledger = [json.loads(line) for line in track(f"{rel}/epochs.jsonl").read_text().splitlines()]
        assert done["status"] == "completed"
        assert [r["epoch"] for r in ledger] == list(range(1, done["epochs_completed"] + 1))
        selected = max(ledger, key=lambda r: (r["fitness"], r["epoch"]))
        assert selected["fitness"] == done["best_fitness"]
        native_records.append((tag, name, title, parent, rel, cfg, args, done, ledger, selected))

    for tag, name, title, parent, rel, cfg, args, done, ledger, selected in native_records:
        raw = read_csv(track(f"{rel}/results.csv"))
        assert [int(r["epoch"]) for r in raw] == list(range(1, len(raw) + 1))
        assert len(raw) == done["epochs_completed"]
        best = int(selected["epoch"])
        criterion = "box_plus_mask_map50_95" if tag in {"기초", "A", "B", "C", "D"} else "mask_map50_95"
        chosen = raw[best - 1]
        fitness = float(chosen["metrics/mAP50-95(M)"])
        if criterion == "box_plus_mask_map50_95":
            fitness += float(chosen["metrics/mAP50-95(B)"])
        assert abs(fitness - done["best_fitness"]) <= 2e-5
        final_val = done.get("final_best_validation", done.get("best_validation_metrics"))
        env_path = source / rel / "environment.json"
        env = read_json(track(f"{rel}/environment.json")) if env_path.exists() else {}
        row = {"id": tag, "run": name, "description": title, "initial_weights": parent,
               "train_images": cfg["baseline"]["expected_counts"]["train"],
               "actual_epochs": len(raw), "requested_epochs": done["epochs_requested"], "selected_epoch": best,
               "selection_metric": criterion, "native_selected_mask_map50": float(chosen["metrics/mAP50(M)"]),
               "native_selected_mask_map50_95": float(chosen["metrics/mAP50-95(M)"]),
               "native_peak_mask_map50_95": max(float(x["metrics/mAP50-95(M)"]) for x in raw),
               "native_last_mask_map50_95": float(raw[-1]["metrics/mAP50-95(M)"]),
               "final_revalidation_mask_map50": final_val["metrics/mAP50(M)"],
               "final_revalidation_mask_map50_95": final_val["metrics/mAP50-95(M)"],
               "elapsed_seconds": done["elapsed_seconds"],
               "sampling": cfg.get("sampling", "uniform"), "bn_stats": cfg.get("bn_stats", "update"),
               "validation_batch": env.get("actual_validation_batch", 2 * args["batch"]),
               "training_segments": "50+25+25" if tag == "M" else str(len(raw)),
               "source_results": f"{rel}/results.csv"}
        row.update({key: args[key] for key in PARAMS})
        train_rows.append(row)
        for i, r in enumerate(raw):
            epoch = i + 1
            e = {"id": tag, "run": name, "epoch": epoch, "selected": epoch == best,
                 "segment": (1 if epoch <= 50 else 2 if epoch <= 75 else 3) if tag == "M" else 1,
                 "elapsed_seconds": ledger[i]["elapsed_seconds"] if ledger else float(r["time"])}
            e.update({key: float(r[value]) for key, value in METRIC_COLUMNS.items()})
            assert all(math.isfinite(v) for v in e.values() if isinstance(v, float))
            epoch_rows.append(e)

    # Confirm the documented 50 + 25 + 25 handoff without exporting private paths.
    mrel = "week03/experiments/runs/coco_full1280_mr2_e100"
    handoff = read_json(track(f"{mrel}/resume_attempt1/handoff.json"))
    m_done = native_records[-1][7]
    assert handoff["start_epoch"] == 50 and handoff["end_epoch"] == 75
    assert m_done["epochs_before_resume"] == 75 and m_done["epochs_completed"] == 100
    resume_audits = []
    for stage, start in ((1, 50), (2, 75)):
        evidence = read_json(track(f"{mrel}/resume_attempt{stage}/start_audit.json"))
        assert evidence["status"] == "passed" and evidence["start_epoch_index"] == start
        assert evidence["target_epochs"] == 100 and evidence["scheduler_last_epoch"] == start - 1
        assert evidence["model_and_ema_match_source_ema"] and evidence["optimizer_and_scaler_restored"]
        resume_audits.append({key: evidence[key] for key in
                             ("status", "start_epoch_index", "first_display_epoch", "target_epochs",
                              "scheduler_last_epoch", "ema_updates", "best_fitness",
                              "model_and_ema_match_source_ema", "optimizer_and_scaler_restored")})

    manifest = read_csv(track("week02/results/data_audit/split_manifest.csv"))
    audit = read_json(track("week02/results/data_audit/summary.json"))
    assert all(r["status"] == "included" for r in manifest)
    classes = ["crack", "efflorescence", "spall_detach"]
    split_rows, split_sets = [], {}
    for split in ("train", "val", "test"):
        rows = [r for r in manifest if r["split"] == split]
        origins = {r["origin_id"] for r in rows}
        groups = {r["split_group"] for r in rows}
        split_sets[split] = (origins, groups)
        counts = Counter(r["kind"] for r in rows)
        split_rows.append({"split": split, "images": len(rows), "origin_ids": len(origins),
                           "capture_groups": len(groups), "plain": counts["plain"],
                           "rf_export": counts["rf_export"], "augmented": counts["augmented"],
                           "manual_images": sum(r["manual"] == "True" for r in rows)})
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (split_sets[a][0] & split_sets[b][0] or split_sets[a][1] & split_sets[b][1])
    assert all(r["kind"] == "plain" for r in manifest if r["split"] != "train")
    train = [r for r in manifest if r["split"] == "train"]
    class_rows = []
    for i, name in enumerate(classes):
        matching = [r for r in train if json.loads(r["class_counts"])[str(i)] > 0]
        origins = {r["origin_id"] for r in matching}
        class_rows.append({"class": name, "origin_ids": len(origins), "images": len(matching),
                           "instances": sum(json.loads(r["class_counts"])[str(i)] for r in matching),
                           "images_per_origin": len(matching) / len(origins)})
    views = []
    for view in ("origin_cap1", "origin_cap4", "manual_augmented"):
        info = read_json(track(f"week03/experiments/data/{view}/summary.json"))
        assert info["counts"]["val"] == 14 and info["counts"]["test"] == 16
        views.append({"view": view, "train_images": info["counts"]["train"],
                      "held_out_files_sha256": info["held_out_files_sha256"]})
    assert len({r["held_out_files_sha256"] for r in views}) == 1
    data_summary = {"seed": 42, "splits": split_rows, "train_classes": class_rows, "train_views": views,
                    "source_images": audit["source_images"], "included_images": len(manifest),
                    "excluded_by_reason": {k: v for k, v in audit["status_counts"].items() if k != "included"},
                    "original_group_definition": "촬영 접두부와 동일 픽셀 중복을 묶은 split_group",
                    "origin_definition": "원본 프레임 식별자(origin_id); 모든 ID에 plain 원본이 있는 것은 아님",
                    "train_origin_representatives": {"plain": 126, "rf_export_only": 37, "augmented_only": 1},
                    "assertions": {"group_overlap": 0, "origin_overlap": 0, "heldout_nonplain": 0,
                                   "heldout_files_same_across_training_views": True}}
    cap1 = read_json(source / "week03/experiments/data/origin_cap1/summary.json")
    assert cap1["train"]["kinds"] == {"rf_export": 37, "plain": 126, "augmented": 1}
    summary = {"runs": train_rows, "training_count": 14, "additional_training_count": 13,
               "curve_scope": "각 학습의 native 검증: 학습 imgsz, 검증 batch 8, 해당 mask_ratio 사용",
               "selection_scope": "기초·A–D는 Box+Mask mAP50–95 합; E–M은 Mask mAP50–95",
               "comparison_scope": "이 표는 학습 중 기록이다. 90개 공통 validation 조건의 재평가 결과와 분리한다.",
               "time_scope": "완료 기록의 elapsed_seconds; M은 세 실행 구간의 누적 시간이며 대기 시간 제외",
               "M_execution": {"segments": [50, 25, 25], "checkpoints_for_resume": [49, 74],
                               "target_epochs": 100, "optimizer_scaler_ema_restored": True,
                               "rng_state_restored": False,
                               "resume_start_audits": resume_audits,
                               "description": "50에폭 후 중단하여 25에폭씩 두 번 재개; 전체 100에폭 학습률 일정 유지"},
               "figures": [{"path": f"docs/images/{name}.png"} for name in
                           ("training_curves", "experiment_lineage", "data_groups")]}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "training_runs.csv", train_rows)
    write_csv(out / "training_epochs.csv", epoch_rows)
    write_csv(out / "data_groups.csv", split_rows)
    write_json(out / "training_summary.json", summary)
    write_json(out / "data_summary.json", data_summary)
    assert all(digest(source / rel) == sha for rel, sha in used.items()), "A source changed during collection."
    write_json(out / "source_hashes.json", {"algorithm": "SHA256", "source_path_base": "course_materials",
                                           "files": used})


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic") if f in fonts), None)
    if family is None:
        for font in sorted((ROOT / "assets/fonts/nanum").glob("*.ttf")):
            font_manager.fontManager.addfont(str(font))
        fonts = {f.name for f in font_manager.fontManager.ttflist}
        family = next((f for f in ("NanumGothic", "NanumBarunGothic", "NanumGothicCoding") if f in fonts), None)
    if family is None:
        raise RuntimeError("한국어 글꼴을 설치하거나 assets/fonts/nanum에 나눔 TTF 글꼴을 두세요.")
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "normal"})
    return plt


def plot_training(plt, summary, epochs, path):
    from matplotlib.lines import Line2D
    groups = defaultdict(list)
    for row in epochs:
        groups[row["id"]].append(row)
    fig, axes = plt.subplots(7, 4, figsize=(17.8, 24.5))
    fig.subplots_adjust(left=.055, right=.985, bottom=.065, top=.947, hspace=.57, wspace=.38)
    for i, run in enumerate(summary["runs"]):
        r, col = divmod(i, 2)
        loss_ax, metric_ax = axes[r, 2 * col:2 * col + 2]
        rows = groups[run["id"]]
        x = [int(v["epoch"]) for v in rows]
        for key, color in (("train_seg_loss", "#2874A6"), ("val_seg_loss", "#CF6D1D")):
            loss_ax.plot(x, [float(v[key]) for v in rows], color=color, lw=1.35)
        for key, color in (("mask_map50", "#8B739B"), ("mask_map50_95", "#147D64")):
            metric_ax.plot(x, [100 * float(v[key]) for v in rows], color=color, lw=1.35)
        for ax in (loss_ax, metric_ax):
            ax.axvline(run["selected_epoch"], color="#222222", lw=1, ls="--")
            ax.set_xlim(1, run["actual_epochs"])
            ax.set_xlabel("에폭", labelpad=2)
            ax.grid(axis="y", color="#E6E6E6", lw=.6)
            ax.tick_params(labelsize=9)
        loss_ax.set_ylim(bottom=0)
        metric_ax.set_ylim(0, 100)
        loss_ax.set_ylabel("마스크 손실")
        metric_ax.set_ylabel("Mask mAP (%)")
        loss_ax.set_title(f"{run['id']}  {run['imgsz']} px · {run['actual_epochs']}/{run['requested_epochs']}에폭", loc="left", fontsize=11)
        metric_ax.set_title(f"선택 {run['selected_epoch']}에폭 · AP50–95 {run['native_selected_mask_map50_95'] * 100:.2f}%", loc="left", fontsize=10)
        if run["id"] == "M":
            for ax in (loss_ax, metric_ax):
                for boundary in (50.5, 75.5):
                    ax.axvline(boundary, color="#777777", ls=":", lw=1)
            loss_ax.text(.97, .95, "50 + 25 + 25에폭", transform=loss_ax.transAxes, ha="right", va="top", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
    fig.suptitle("기초 학습과 13개 추가 학습의 에폭별 변화", y=.984, fontsize=19)
    handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in
               (("#2874A6", "학습 마스크 손실"), ("#CF6D1D", "검증 마스크 손실"),
                ("#8B739B", "Mask mAP50"), ("#147D64", "Mask mAP50–95"))]
    handles.append(Line2D([], [], color="#222222", ls="--", label="실제 best.pt 선택 에폭"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .973), ncol=5, frameon=False, fontsize=10)
    fig.text(.055, .012, "각 학습의 입력 크기·mask_ratio와 검증 batch 8을 사용한 학습 중 평가 기록(native validation).\n기초·A–D: Box+Mask mAP50–95 합으로 선택 / E–M: Mask mAP50–95로 선택.\nM의 점선은 재개 경계이다. 공통 조건으로 재평가한 90개 검증 결과는 별도 결과표에 제시한다.", fontsize=11, va="bottom")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lineage(plt, summary, path):
    from matplotlib.patches import Rectangle
    by_id = {r["id"]: r for r in summary["runs"]}
    lanes = [("COCO 사전학습", ["기초", "E", "G", "M"]),
             ("기초 best.pt · 26에폭", ["A", "B", "C", "D"]),
             ("B best.pt · 7에폭", ["F"]),
             ("G best.pt · 97에폭", ["H", "I", "J", "K", "L"])]
    fig, ax = plt.subplots(figsize=(17.8, 9.2))
    ax.set_xlim(0, 18); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(.15, 8.75, "초기 가중치에 따른 학습 실험 계보", fontsize=20, va="top")
    ax.text(.15, 8.25, "각 행의 왼쪽 가중치에서 오른쪽 실험을 시작했다. 같은 YOLOv8s-Seg 구조에서 초기 가중치와 학습 조건을 달리한 14회 학습이다.", fontsize=11)
    for row_index, (parent, children) in enumerate(lanes):
        y = 6.5 - row_index * 1.75
        ax.text(.15, y + .62, parent, fontsize=12, va="center")
        ax.annotate("", xy=(3.30, y + .62), xytext=(2.8, y + .62), arrowprops={"arrowstyle": "->", "color": "#555555"})
        width = 2.72
        for j, key in enumerate(children):
            run = by_id[key]; x = 3.45 + j * 2.85
            ax.add_patch(Rectangle((x, y), width, 1.45, facecolor="white", edgecolor="#A8B0B7", lw=.9))
            ax.text(x + .12, y + 1.25, f"{key}  {run['description']}", fontsize=10 if key != "F" else 9.5, va="top", wrap=True)
            detail = f"{run['imgsz']} px · {run['train_images']:,}장\n{run['actual_epochs']}/{run['requested_epochs']}에폭 · lr {run['lr0']:g}\n선택 {run['selected_epoch']}에폭 · mask_ratio {run['mask_ratio']}"
            ax.text(x + .12, y + .78, detail, fontsize=10, va="top", linespacing=1.55)
    ax.text(.15, .05, "M: 50에폭 후 중단 → 25에폭 재개 → 25에폭 재개. optimizer·scaler·EMA 상태와 100에폭 학습률 일정을 이어 사용했다.", fontsize=11, va="bottom")
    fig.subplots_adjust(left=.025, right=.99, top=.98, bottom=.035)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_data(plt, data, path):
    fig = plt.figure(figsize=(15.2, 10.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], left=.065, right=.97, bottom=.12, top=.86, hspace=.45, wspace=.35)
    ax = fig.add_subplot(grid[0, :]); ax.axis("off")
    labels = {"train": "학습", "val": "검증", "test": "시험"}
    cols = ["집합", "촬영 그룹", "원본 프레임 ID", "전체 영상", "plain / rf / 증강", "직접 주석"]
    cells = [[labels[r['split']], r['capture_groups'], r['origin_ids'], f"{r['images']:,}",
              f"{r['plain']} / {r['rf_export']} / {r['augmented']:,}", r['manual_images']] for r in data['splits']]
    table = ax.table(cellText=cells, colLabels=cols, cellLoc="center", colWidths=[.12,.16,.19,.14,.25,.14], bbox=[0,.15,1,.78])
    table.auto_set_font_size(False); table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D2D7DC"); cell.set_facecolor("#F2F5F7" if r == 0 else "white")
    ax.text(0, 1.03, "촬영 그룹과 원본 ID가 겹치지 않는 분할", transform=ax.transAxes, fontsize=15)
    ax.text(0, -.04, "집합 간 촬영 그룹·원본 ID 교집합 0개 · 검증·시험에는 plain 영상만 사용", transform=ax.transAxes, fontsize=11)
    left = fig.add_subplot(grid[1, 0])
    pos = left.get_position()
    left.set_position([pos.x0 + .085, pos.y0, pos.width - .085, pos.height])
    view_names = ["전체 학습 자료", "원본 ID당 최대 1장", "원본 ID당 최대 4장", "직접 주석 증강 추가"]
    values = [data['splits'][0]['images']] + [r['train_images'] for r in data['train_views']]
    bars = left.barh(view_names, values, color=["#355E7B", "#B2C6D2", "#83A8BE", "#5B8F9A"], height=.6)
    left.invert_yaxis(); left.set_xlim(0, 3600); left.set_xlabel("학습에 사용한 영상 수 (장)")
    left.set_title("원본 ID 164개와 검증·시험 자료는 유지", loc="left", fontsize=13)
    left.grid(axis="x", color="#E6E6E6"); left.set_axisbelow(True)
    for b, n in zip(bars, values): left.text(n + 45, b.get_y() + b.get_height()/2, f"{n:,}", va="center", fontsize=11)
    right = fig.add_subplot(grid[1, 1]); right.axis("off")
    class_names = {"crack": "균열", "efflorescence": "백태", "spall_detach": "박리·박락"}
    cells = [[class_names[r['class']], r['origin_ids'], f"{r['images']:,}", f"{r['images_per_origin']:.1f}"] for r in data['train_classes']]
    table = right.table(cellText=cells, colLabels=["클래스", "원본 ID", "학습 영상", "ID당 평균"], cellLoc="center", bbox=[0,.18,1,.65])
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (r,c), cell in table.get_celld().items(): cell.set_edgecolor("#D2D7DC"); cell.set_facecolor("#F2F5F7" if r==0 else "white")
    right.set_title("전체 학습 자료의 클래스별 노출 수", loc="left", fontsize=13)
    right.text(0, .02, "영상 수는 증강본을 포함한다.\nID당 평균은 학습 영상 수 ÷ 원본 ID 수이다.", fontsize=10, transform=right.transAxes)
    fig.suptitle("분할 단위와 학습 자료 구성", y=.965, fontsize=20)
    fig.text(.065, .91, "제공 영상 4,116장 → 실제 분할에 사용한 영상 3,117장. 검증 14장·시험 16장은 모든 학습 자료 구성에서 동일하다.", fontsize=12)
    fig.text(.065, .035, "학습 원본 ID 164개: plain 영상이 있는 126개 + rf_export만 있는 37개 + 증강 영상만 있는 1개.\n1장/4장 제한은 기존 영상 선택이며, 직접 주석 증강은 학습용 6장에 새 변환을 적용한 138장을 추가한 구성이다.", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional course directory containing the original week02 and week03 results")
    args = parser.parse_args()
    tables, images = ROOT / "results/experiments", ROOT / "docs/images"
    if args.source_root:
        collect(args.source_root.resolve(), tables)
    summary, data = read_json(tables / "training_summary.json"), read_json(tables / "data_summary.json")
    epochs = read_csv(tables / "training_epochs.csv")
    assert len(summary["runs"]) == 14 and len({r["id"] for r in summary["runs"]}) == 14
    plt = setup_plotting(); images.mkdir(parents=True, exist_ok=True)
    plot_training(plt, summary, epochs, images / "training_curves.png")
    plot_lineage(plt, summary, images / "experiment_lineage.png")
    plot_data(plt, data, images / "data_groups.png")
    print("Created training_curves.png, experiment_lineage.png, data_groups.png from bundled evidence.")


if __name__ == "__main__":
    main()
