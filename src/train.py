"""Train the recorded COCO or continuation configuration on the prepared split."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as evaluation

require = evaluation.require
write_json = evaluation.write_json
sha256 = evaluation.sha256


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    require(config["ultralytics_version"] == "8.4.12" and config["seed"] == 42
            and config["classes"] == evaluation.CLASSES, "Use the recorded environment and classes.")
    require(config["initialization"] in ("coco_pretrained", "previous_best")
            and config["checkpoint_metric"] == "mask_map50_95"
            and config["sampling"] == "uniform" and config["bn_stats"] in ("update", "frozen"),
            "Unsupported training method.")
    params = config["params"]
    require(params["resume"] is False and params["workers"] == 0 and params["cache"] is False
            and params["seed"] == 42 and params["save_period"] == 1 and params["batch"] == 4
            and params["mask_ratio"] == 2, "Training input or resource settings differ.")
    require(not (set(params) & {"data", "model", "project", "name", "exist_ok", "save_dir", "trainer"}),
            "Training paths belong to the command line.")
    return config


def run(config, state, report, output, device):
    from runtime import configure
    guard, environment = configure(output / ".runtime", device)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import SegmentMetrics
    model = YOLO(str(state["paths"]["weights"]))
    expected = 80 if config["initialization"] == "coco_pretrained" else 3
    require(model.task == "segment" and len(model.names) == expected, "Initial class count differs.")
    if expected == 3:
        require([model.names[i] for i in range(3)] == evaluation.CLASSES, "Initial class mapping differs.")
    architecture = model.model.yaml
    require(architecture.get("scale") == "s" or
            (architecture.get("depth_multiple") == 0.33 and architecture.get("width_multiple") == 0.5),
            "Use the YOLOv8s segmentation architecture.")
    options = {}
    if config["bn_stats"] == "frozen":
        from frozen_batchnorm import make_frozen_bn_trainer
        from ultralytics.models.yolo.segment import SegmentationTrainer
        options["trainer"] = make_frozen_bn_trainer(SegmentationTrainer)
    started, epochs = time.monotonic(), []

    def record_start(trainer):
        require(trainer.save_dir.resolve() == output and trainer.start_epoch == 0 and not trainer.resume,
                "Expected a new run and optimizer in the requested output directory.")
        require([trainer.model.names[i] for i in range(3)] == evaluation.CLASSES, "Training classes differ.")
        write_json(output / "normalized_args.json", vars(trainer.args))
        write_json(output / "environment.json", {**environment, "amp_enabled": bool(trainer.amp),
            "training_batch": trainer.batch_size, "validation_batch": trainer.test_loader.batch_size,
            "initialization": config["initialization"], "bn_stats": config["bn_stats"]})

    def record_epoch(trainer):
        guard()
        epoch = int(trainer.epoch) + 1
        require(epoch == len(epochs) + 1, "Repeated or missing epoch record.")
        checkpoint = trainer.wdir / f"epoch{trainer.epoch}.pt"
        require(checkpoint.is_file(), "Per-epoch checkpoint is missing.")
        row = {"epoch": epoch, "checkpoint": checkpoint.relative_to(output).as_posix(),
               "checkpoint_sha256": sha256(checkpoint), "fitness": float(trainer.fitness),
               "validation": {key: float(value) for key, value in trainer.metrics.items()},
               "elapsed_seconds": round(time.monotonic() - started, 3)}
        if config["bn_stats"] == "frozen":
            row["frozen_bn"] = trainer.frozen_bn_epoch_summary()
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        epochs.append(epoch)

    for event in ("on_train_batch_start", "on_train_batch_end", "on_val_batch_start", "on_val_batch_end"):
        model.add_callback(event, guard)
    model.add_callback("on_train_start", record_start)
    model.add_callback("on_model_save", record_epoch)
    params = {**config["params"], "device": device}
    with patch.object(SegmentMetrics, "fitness", property(lambda metrics: metrics.seg.fitness())), \
            patch("ultralytics.data.dataset.save_dataset_cache_file", evaluation.memory_label_cache), \
            patch("ultralytics.data.dataset.load_dataset_cache_file", side_effect=FileNotFoundError), \
            patch("ultralytics.data.dataset.NUM_THREADS", 2), \
            patch("ultralytics.utils.callbacks.add_integration_callbacks", return_value=None), \
            patch("ultralytics.utils.downloads.safe_download", side_effect=ConnectionError("Downloads are disabled.")):
        model.train(**params, data=str(state["paths"]["data_yaml"]),
                    project=str(output.parent), name=output.name, exist_ok=True, **options)
    guard()
    require(sha256(state["paths"]["weights"]) == report["freeze"]["weights_sha256"], "Initial weights changed.")
    write_json(output / "completed.json", {"status": "completed", "epochs_completed": len(epochs),
        "epochs_requested": config["params"]["epochs"], "checkpoint_metric": config["checkpoint_metric"],
        "elapsed_seconds": round(time.monotonic() - started, 3), "freeze": report["freeze"],
        "best_checkpoint": "weights/best.pt", "best_checkpoint_sha256": sha256(output / "weights/best.pt"),
        "best_fitness": float(model.trainer.best_fitness),
        "final_best_validation": {key: float(value) for key, value in model.trainer.metrics.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--weights-sha256", help="Explicit input SHA for a newly reproduced parent checkpoint")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    original_sha = config["baseline"]["weights_sha256"]
    if args.weights_sha256:
        require(len(args.weights_sha256) == 64 and all(c in "0123456789abcdef" for c in args.weights_sha256),
                "Invalid input SHA-256.")
        config["baseline"]["weights_sha256"] = args.weights_sha256
    config = evaluation.bind_inputs(config, args.data, args.weights, args.device)
    report, state = evaluation.preflight(config)
    output = args.output.resolve()
    require(not output.is_relative_to(state["dataset"])
            and not output.is_relative_to(state["paths"]["source_root"]), "Output cannot be inside input data.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", report)
    write_json(output / "configuration.json", config)
    if args.weights_sha256:
        write_json(output / "parent_checkpoint.json", {"recorded_sha256": original_sha,
                    "provided_sha256": args.weights_sha256, "checkpoint": args.weights.name})
    if args.check_only:
        print(json.dumps({"status": "passed", "counts": report["counts"], "model_loaded": False}))
        return
    try:
        run(config, state, report, output, args.device)
    except BaseException as error:
        write_json(output / "failed.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
