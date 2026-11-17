"""Validation threshold selection followed by frozen test evaluation."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crack", "efflorescence", "spall_detach"]
METRIC_KEYS = ["precision", "recall", "map50", "map50_95"]
GRID = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Publish a complete JSON file atomically, without replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def root_path(value, root=ROOT):
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def parse_labels(text):
    counts = [0] * len(CLASSES)
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        require(len(values) >= 7 and len(values) % 2 == 1
                and all(math.isfinite(v) for v in values), "Invalid segmentation label.")
        cid = int(values[0])
        require(values[0] == cid and 0 <= cid < len(CLASSES)
                and all(0 <= v <= 1 for v in values[1:]), "Invalid class or normalized coordinate.")
        counts[cid] += 1
    return counts


def preflight(config, root=ROOT):
    """Verify the recorded data and labels without importing torch or loading a model."""
    import yaml
    version = importlib.metadata.version("ultralytics")
    require(version == config["ultralytics_version"], f"Ultralytics mismatch: {version}")
    paths = {key: root_path(value, root) for key, value in config["inputs"].items()}
    for key in ("data_yaml", "weights", "split_manifest"):
        require(paths[key].is_file(), f"Missing input: {key}: {paths[key]}")
    for key in ("source_root", "annotations_root"):
        require(paths[key].is_dir(), f"Missing input directory: {key}")
    weights_sha = sha256(paths["weights"])
    split_sha = sha256(paths["split_manifest"])
    require(weights_sha == config["baseline"]["weights_sha256"], "Model SHA-256 changed.")
    require(split_sha == config["baseline"]["split_manifest_sha256"], "Split manifest SHA-256 changed.")
    data = yaml.safe_load(paths["data_yaml"].read_text(encoding="utf-8"))
    require(isinstance(data, dict) and "download" not in data, "Dataset YAML must not contain a download action.")
    names = data["names"]
    require(len(names) == len(CLASSES), "data.yaml must define exactly three classes.")
    names = [names[k] for k in range(3)] if isinstance(names, dict) and 0 in names else names
    names = [names[str(k)] for k in range(3)] if isinstance(names, dict) else names
    require(names == config["classes"], "data.yaml class mapping differs.")
    dataset = Path(data["path"])
    require(dataset.is_absolute(), "data.yaml path must be an explicit absolute dataset path.")
    dataset = dataset.resolve()
    require(dataset == paths["data_yaml"].parent, "data.yaml must describe its own derived dataset directory.")
    with paths["split_manifest"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(all(row["status"] == "included" for row in rows), "Split manifest contains excluded rows.")
    require({row["split"] for row in rows} == {"train", "val", "test"}, "Unexpected split names.")
    split_rows = {split: sorted([r for r in rows if r["split"] == split], key=lambda r: r["output_image"])
                  for split in ("train", "val", "test")}
    counts = {split: len(items) for split, items in split_rows.items()}
    require(counts == config["baseline"]["expected_counts"], f"Split counts changed: {counts}")
    for key in ("split_group", "sequence_group", "origin_id", "file_sha256", "pixel_sha256"):
        groups = {s: {r[key] for r in items} for s, items in split_rows.items()}
        require(all("" not in values for values in groups.values()), f"Empty {key}.")
        require(not (groups["train"] & groups["val"] or groups["train"] & groups["test"]
                     or groups["val"] & groups["test"]), f"Cross-split leakage: {key}")
    records, distributions, fingerprint_rows = {}, {}, []
    for split, items in split_rows.items():
        require(data[split] == f"images/{split}", f"Unexpected YAML split directory: {split}")
        expected = {r["output_image"] for r in items}
        actual = {p.relative_to(dataset).as_posix() for p in (dataset / data[split]).rglob("*") if p.is_file()}
        require(actual == expected and len(expected) == len(items), f"Image membership differs: {split}")
        require(all(Path(r["output_label"]).parent.as_posix() == f"labels/{split}"
                    and Path(r["output_image"]).stem == Path(r["output_label"]).stem for r in items),
                "Output label mapping differs.")
        records[split] = []
        positives, instances, all_negative = [0] * 3, [0] * 3, 0
        for row in items:
            image = dataset / row["output_image"]
            label = dataset / row["output_label"]
            require(label.resolve().is_relative_to(dataset), "Label escapes derived dataset.")
            require(image.resolve().is_relative_to(paths["source_root"]), "Image escapes the supplied source data.")
            require(sha256(image) == row["file_sha256"], f"Image SHA changed: {row['output_image']}")
            if image.suffix.lower() in {".jpg", ".jpeg"}:
                with image.open("rb") as stream:
                    stream.seek(-2, 2)
                    require(stream.read() == b"\xff\xd9", "JPEG needs repair; input repair is not allowed here.")
            if row["manual"].lower() == "true":
                source_label = paths["annotations_root"] / "labels" / f"{row['origin_id']}.txt"
            else:
                source_label = paths["source_root"] / row["source_label"]
            require(source_label.resolve().is_relative_to(paths["source_root"])
                    or source_label.resolve().is_relative_to(paths["annotations_root"]), "Source label escapes inputs.")
            require(sha256(source_label) == row["label_sha256"], f"Input label SHA changed: {row['origin_id']}")
            normalized = source_label.read_text(encoding="utf-8-sig").rstrip() + "\n"
            require(label.read_bytes() == normalized.encode("utf-8"), f"Derived label differs: {row['output_label']}")
            label_counts = parse_labels(normalized)
            expected_counts = json.loads(row["class_counts"])
            require(label_counts == [expected_counts[str(i)] for i in range(3)], "Manifest class counts differ.")
            gt = [i for i, n in enumerate(label_counts) if n]
            for cid in range(3):
                positives[cid] += int(cid in gt)
                instances[cid] += label_counts[cid]
            all_negative += int(not gt)
            record = {"image": row["output_image"], "label": row["output_label"],
                      "gt_classes": gt, "image_sha256": row["file_sha256"],
                      "label_sha256": sha256(label)}
            records[split].append(record)
            fingerprint_rows.append(record)
        distributions[split] = {"images": len(items), "all_classes_negative": all_negative,
            "classes": {name: {"positive_images": positives[i], "negative_images": len(items) - positives[i],
                               "instances": instances[i]} for i, name in enumerate(CLASSES)}}
    freeze = {"config_sha256": object_sha(config), "weights_sha256": weights_sha,
              "split_manifest_sha256": split_sha,
              "dataset_fingerprint": object_sha({"yaml_sha256": sha256(paths["data_yaml"]),
                                                 "split_manifest_sha256": split_sha, "held_out": fingerprint_rows})}
    report = {"status": "passed", "ultralytics_version": version, "counts": counts,
              "groups_disjoint": True, "held_out_files_verified": counts["val"] + counts["test"],
              "training_files_verified": counts["train"],
              "distribution": distributions, "freeze": freeze, "model_loaded": False}
    return report, {"paths": paths, "dataset": dataset, "records": records}


def score(records, cid, threshold):
    counts = dict.fromkeys(("TP", "FP", "FN", "TN"), 0)
    for row in records:
        gt = cid in row["gt_classes"]
        conf = row["max_confidence"][cid]
        predicted = conf is not None and conf >= threshold
        counts[("TP" if gt else "FP") if predicted else ("FN" if gt else "TN")] += 1
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def select_thresholds(records, thresholds):
    grid, chosen = [], {}
    for cid, name in enumerate(CLASSES):
        rows = [{"class": name, "threshold": t, **score(records, cid, t)} for t in thresholds]
        grid.extend(rows)
        chosen[name] = min(rows, key=lambda r: (-r["f1"], r["threshold"]))["threshold"]
    return chosen, grid


def verify_selection(selected, freeze, cache):
    require(selected["freeze"] == freeze, "Frozen model, dataset, split or configuration changed.")
    require(sha256(cache) == selected["validation_cache_sha256"], "Validation cache changed.")
    cached = json.loads(cache.read_text(encoding="utf-8"))
    require(cached["freeze"] == freeze and cached["split"] == "val", "Cache provenance differs.")
    chosen, _ = select_thresholds(cached["images"], selected["threshold_grid"])
    require(chosen == selected["thresholds"], "Selected thresholds differ from the frozen validation result.")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_curve(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name in CLASSES:
        rows = [r for r in grid if r["class"] == name]
        axis.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], marker="o", label=name)
    axis.set(xscale="log", xlabel="Confidence threshold (log scale)", ylabel="Image-level F1",
             title="Validation F1 by threshold", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def memory_label_cache(prefix, path, cache, version):
    # get_labels() expects the version inserted by the normal disk-cache writer.
    cache["version"] = version


def segmentation_metrics(model, guard, config, state, destination, split="test"):
    from ultralytics.models.yolo.segment import SegmentationValidator

    class BoundedValidator(SegmentationValidator):
        def postprocess(self, preds):
            processed = super().postprocess(preds)
            require(all(len(row["cls"]) < self.args.max_det for row in processed),
                    "max_det saturation during mask evaluation.")
            return processed

    guard()
    # cache=False controls image caching; the separate label cache needs its own write guard.
    with patch("ultralytics.data.dataset.save_dataset_cache_file", memory_label_cache), \
            patch("ultralytics.data.dataset.load_dataset_cache_file", side_effect=FileNotFoundError), \
            patch("ultralytics.data.dataset.NUM_THREADS", 2):
        metrics = model.val(validator=BoundedValidator, data=str(state["paths"]["data_yaml"]),
                            split=split, imgsz=config["inference"]["imgsz"], device=config["inference"]["device"], half=False,
                            rect=True, **config["mask_validation"], cache=False,
                            project=str(destination), name="mask_validation", exist_ok=False,
                            augment=False, seed=config["seed"], deterministic=True, verbose=False)
    guard()
    result = {}
    for name, metric in (("mask", metrics.seg), ("box", metrics.box)):
        by_class = {CLASSES[int(cid)]: dict(zip(METRIC_KEYS, map(float, metric.class_result(i))))
                    for i, cid in enumerate(metric.ap_class_index)}
        # All test classes have GT instances; an absent AP row means no matched predictions.
        result[name] = {"overall": dict(zip(METRIC_KEYS, map(float, metric.mean_results()))),
                        "per_class": {name: by_class.get(name, dict.fromkeys(METRIC_KEYS, 0.0)) for name in CLASSES}}
    result["speed_ms_per_image"] = {k: float(v) for k, v in metrics.speed.items()}
    return result


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    require(config["schema_version"] == 1 and config["classes"] == CLASSES,
                 "Unsupported schema or class mapping.")
    require(config["ultralytics_version"] == "8.4.12" and config["seed"] == 42,
                 "Use Ultralytics 8.4.12 and seed 42.")
    require(config["thresholds"] == GRID and config["tie_break"] == "lowest_threshold",
                 "Use the recorded threshold grid and tie rule.")
    infer, mask = config["inference"], config["mask_validation"]
    require(set(infer) == {"imgsz", "batch", "device", "conf", "iou", "max_det", "half", "rect"}
                 and set(mask) == {"conf", "iou", "max_det", "batch", "workers", "plots"},
                 "Unexpected inference or mask-validation option.")
    require(infer["imgsz"] in (640, 800, 960, 1152, 1280, 1536) and infer["batch"] == 1
                 and str(infer["device"]) == "0" and infer["half"] is False
                 and infer["rect"] is True, "Invalid experiment inference settings.")
    require(infer["conf"] == 0.0001 and mask["conf"] == 0.001,
                 "Keep prediction and mask confidence floors fixed.")
    require(mask["batch"] == 1 and mask["workers"] == 0 and mask["plots"] is True,
                 "Use mask batch 1, workers 0 and plots=True.")
    for settings in (infer, mask):
        require(settings["iou"] == 0.7 and settings["max_det"] == 300,
                     "Keep NMS IoU and max_det fixed.")
    for key in ("weights_sha256", "split_manifest_sha256"):
        digest = config["baseline"][key]
        require(isinstance(digest, str) and len(digest) == 64
                     and all(c in "0123456789abcdef" for c in digest), "Invalid baseline SHA-256.")
    return config


def bind_inputs(config, dataset, weights, device):
    """Resolve machine-specific paths at execution time, not in tracked configuration."""
    dataset, weights = Path(dataset).resolve(), Path(weights).resolve()
    source = json.loads((dataset / "source.json").read_text(encoding="utf-8"))
    require(source["split_manifest_sha256"] == config["baseline"]["split_manifest_sha256"],
            "Prepared dataset uses a different manifest.")
    config["inputs"] = {"data_yaml": str(dataset / "data.yaml"), "weights": str(weights),
                        "source_root": source["source_root"], "annotations_root": str(ROOT / "annotations"),
                        "split_manifest": str(ROOT / "data/split_manifest.csv")}
    if "inference" in config:
        config["inference"]["device"] = device
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "tune", "test"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--data", type=Path, required=True, help="Directory created by prepare_data.py")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--weights-sha256", help="Explicit SHA for newly reproduced model weights")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.weights_sha256:
        require(len(args.weights_sha256) == 64 and all(c in "0123456789abcdef" for c in args.weights_sha256),
                "Invalid input SHA-256.")
        config["reproduced_weights"] = {"recorded_sha256": config["baseline"]["weights_sha256"],
                                        "provided_sha256": args.weights_sha256}
        config["baseline"]["weights_sha256"] = args.weights_sha256
    config = bind_inputs(config, args.data, args.weights, args.device)
    report, state = preflight(config)
    output = args.output.resolve()
    require(not output.is_relative_to(state["dataset"])
            and not output.is_relative_to(state["paths"]["source_root"]), "Output cannot be inside input data.")
    if args.command == "preflight":
        write_json(output / "preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    selected = None
    if args.command == "test":
        completed = json.loads((output / "tune/completed.json").read_text(encoding="utf-8"))
        require(completed["status"] == "completed" and completed["command"] == "tune"
                and completed["freeze"] == report["freeze"], "Validation tuning did not finish with these inputs.")
        selected = json.loads((output / "tune/selected_thresholds.json").read_text(encoding="utf-8"))
        require(selected["threshold_grid"] == config["thresholds"], "Threshold grid changed.")
        verify_selection(selected, report["freeze"], output / "tune/validation_predictions.json")
    destination = output / args.command
    destination.mkdir(parents=True, exist_ok=False)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from runtime import configure
        guard, environment = configure(destination / ".runtime", args.device)
        from ultralytics import YOLO
        model = YOLO(str(state["paths"]["weights"]))
        require(model.task == "segment" and len(model.names) == 3
                and [model.names[i] for i in range(3)] == CLASSES, "Model task or class mapping differs.")
        for event in ("on_predict_batch_start", "on_predict_batch_end", "on_val_batch_start", "on_val_batch_end"):
            model.add_callback(event, guard)
        split = "val" if args.command == "tune" else "test"
        with patch("ultralytics.utils.downloads.safe_download", side_effect=ConnectionError("Downloads are disabled.")), \
                patch("ultralytics.utils.callbacks.add_integration_callbacks", return_value=None):
            images = predict_split(model, guard, config, state, split, destination,
                                   selected["thresholds"] if selected else None)
            cache = destination / ("validation_predictions.json" if split == "val" else "test_predictions.json")
            write_json(cache, {"freeze": report["freeze"], "split": split, "images": images})
            mask = segmentation_metrics(model, guard, config, state, destination, split)
        write_json(destination / "segmentation_metrics.json", mask)
        if args.command == "tune":
            thresholds, grid = select_thresholds(images, config["thresholds"])
            write_csv(destination / "validation_thresholds.csv", grid)
            threshold_curve(grid, destination / "val_threshold_f1.png")
            write_json(destination / "selected_thresholds.json", {
                "freeze": report["freeze"], "validation_cache_sha256": sha256(cache),
                "threshold_grid": config["thresholds"], "tie_break": config["tie_break"],
                "thresholds": thresholds, "selection_split": "val", "environment": environment})
        else:
            rows = [{"class": name, "threshold": selected["thresholds"][name],
                     **score(images, cid, selected["thresholds"][name])} for cid, name in enumerate(CLASSES)]
            write_csv(destination / "test_onoff.csv", rows)
            errors = {name: {"FP": [], "FN": []} for name in CLASSES}
            for row in images:
                for cid, name in enumerate(CLASSES):
                    conf = row["max_confidence"][cid]
                    predicted = conf is not None and conf >= selected["thresholds"][name]
                    gt = cid in row["gt_classes"]
                    if predicted != gt:
                        errors[name]["FN" if gt else "FP"].append(row["image"])
            write_json(destination / "error_cases.json", errors)
            write_json(destination / "test_summary.json", {"freeze": report["freeze"], "classes": rows,
                "macro_f1": sum(row["f1"] for row in rows) / len(rows),
                "selected_thresholds_sha256": sha256(output / "tune/selected_thresholds.json"),
                "environment": environment})
        require(sha256(state["paths"]["weights"]) == report["freeze"]["weights_sha256"], "Weights changed during evaluation.")
        write_json(destination / "completed.json", {"status": "completed", "command": args.command,
                   "completed_at": datetime.now(timezone.utc).isoformat(), "freeze": report["freeze"]})
    except BaseException as error:
        write_json(destination / "failed.json", {"status": "failed", "error": str(error)})
        raise


def prediction_limit_metadata(scores, limit, threshold_floor):
    """Check a native YOLOv8 NMS prefix without changing its returned predictions."""
    require(len(scores) <= limit, "Prediction count exceeds max_det.")
    if len(scores) < limit:
        return None
    require(all(math.isfinite(score) and 0 <= score <= 1 for score in scores),
                 "Invalid confidence at max_det saturation.")
    require(all(left >= right for left, right in zip(scores, scores[1:])),
                 "Saturated predictions are not in native NMS descending score order.")
    # v8.4.12 native NMS returns descending scores before i[:max_det]. Thus any
    # omitted NMS survivor is below every search threshold only under strict <.
    require(scores[-1] < threshold_floor,
                 "max_det saturation may truncate a prediction at a search threshold.")
    return {"limit": limit, "reached": True, "lowest_kept_confidence": scores[-1],
            "threshold_floor": threshold_floor, "threshold_decisions_complete": True}


def predict_split(model, guard, config, state, split, destination, thresholds=None):
    images = []
    if thresholds is not None:
        (destination / "predictions").mkdir()
    for row in state["records"][split]:
        guard()
        result = model.predict(source=str(state["dataset"] / row["image"]),
                               **config["inference"], verbose=False, save=False,
                               project=str(destination), name="prediction",
                               augment=False, agnostic_nms=False, stream=False)[0]
        scores = [float(box.conf.item()) for box in result.boxes]
        limit = prediction_limit_metadata(scores, config["inference"]["max_det"], min(config["thresholds"]))
        require(not len(result.boxes) or result.masks is not None, "Segmentation masks are missing.")
        instances, maxima = [], [None] * 3
        polygons = result.masks.xyn if result.masks is not None else []
        for i, box in enumerate(result.boxes):
            cid, confidence = int(box.cls.item()), scores[i]
            require(0 <= cid < 3 and math.isfinite(confidence), "Invalid prediction class/confidence.")
            maxima[cid] = confidence if maxima[cid] is None else max(maxima[cid], confidence)
            instances.append({"class_id": cid, "confidence": confidence,
                              "box_xyxy_pixels": box.xyxy[0].cpu().tolist(),
                              "polygon_xyn": polygons[i].tolist()})
        image = {**row, "shape_hw": list(result.orig_shape),
                 "max_confidence": maxima, "instances": instances}
        if limit is not None:
            image["prediction_limit"] = limit
        images.append(image)
        if thresholds is not None:
            keep = [i for i, obj in enumerate(instances)
                    if obj["confidence"] >= thresholds[CLASSES[obj["class_id"]]]]
            result[keep].save(filename=str(destination / "predictions" / (Path(row["image"]).stem + ".jpg")))
        guard()
    return images


if __name__ == "__main__":
    main()
