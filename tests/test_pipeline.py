"""CPU tests for source preservation, split validation and threshold freezing."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate as evaluation
import prepare_data


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self):
        from PIL import Image
        source, annotations = self.root / "원본", self.root / "annotations"
        source.mkdir()
        (annotations / "labels").mkdir(parents=True)
        rows = []
        for i, split in enumerate(("train", "val", "test")):
            image, label = source / f"image{i}.jpg", source / f"image{i}.txt"
            Image.new("RGB", (16, 12), (i * 60, 20, 30)).save(image)
            label.write_text(f"{i} 0.1 0.1 0.9 0.1 0.9 0.9\n", encoding="utf-8-sig")
            manual = i == 1
            if manual:
                target = annotations / "labels/image1.txt"
                target.write_bytes(label.read_bytes())
            rows.append({"status": "included", "split": split, "origin_id": f"image{i}",
                "source_image": image.name, "source_label": label.name, "manual": str(manual),
                "output_image": f"images/{split}/image{i}.jpg", "output_label": f"labels/{split}/image{i}.txt",
                "file_sha256": evaluation.sha256(image), "label_sha256": evaluation.sha256(label),
                "pixel_sha256": str(i), "split_group": str(i), "sequence_group": str(i),
                "class_counts": json.dumps({str(cid): int(cid == i) for cid in range(3)})})
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        weights = self.root / "weights.pt"
        weights.write_bytes(b"CPU preflight does not deserialize this file")
        output = self.root / "dataset"
        config = {"ultralytics_version": "8.4.12", "classes": evaluation.CLASSES,
            "baseline": {"weights_sha256": evaluation.sha256(weights),
                         "split_manifest_sha256": evaluation.sha256(manifest),
                         "expected_counts": {"train": 1, "val": 1, "test": 1}},
            "inputs": {"data_yaml": str(output / "data.yaml"), "weights": str(weights),
                       "split_manifest": str(manifest), "source_root": str(source), "annotations_root": str(annotations)}}
        return source, output, manifest, annotations, config

    def test_prepare_and_preflight_preserve_bom_sources(self):
        source, output, manifest, annotations, config = self.fixture()
        original = {p: p.read_bytes() for p in source.iterdir()}
        self.assertEqual(prepare_data.prepare(source, output, manifest, annotations), {"train": 1, "val": 1, "test": 1})
        report, state = evaluation.preflight(config)
        self.assertEqual(report["held_out_files_verified"], 2)
        self.assertEqual(report["training_files_verified"], 1)
        self.assertEqual(state["records"]["val"][0]["gt_classes"], [1])
        self.assertFalse((output / "labels/val/image1.txt").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(original, {p: p.read_bytes() for p in source.iterdir()})
        with self.assertRaises(FileExistsError):
            prepare_data.prepare(source, output, manifest, annotations)
        (output / "labels/train/image0.txt").write_text("0 0 0 1 0 1 1\n")
        with self.assertRaisesRegex(ValueError, "Derived label differs"):
            evaluation.preflight(config)

    def test_failed_prepare_leaves_no_partial_output(self):
        source, output, manifest, annotations, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Output cannot be inside"):
            prepare_data.prepare(source, source / "derived", manifest, annotations)
        self.assertFalse((source / "derived").exists())
        (source / "image2.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            prepare_data.prepare(source, output, manifest, annotations)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".prepare-*")))
        with self.assertRaises(ValueError):
            prepare_data.within(source, "../outside.txt")

    def test_validation_freeze_and_lowest_threshold_tie(self):
        records = [{"gt_classes": [0], "max_confidence": [0.9, None, None]},
                   {"gt_classes": [], "max_confidence": [0.005, None, None]}]
        thresholds, _ = evaluation.select_thresholds(records, [0.001, 0.01, 0.1])
        self.assertEqual(thresholds, {"crack": 0.01, "efflorescence": 0.001, "spall_detach": 0.001})
        freeze = {"weights_sha256": "one"}
        cache = self.root / "validation.json"
        evaluation.write_json(cache, {"freeze": freeze, "split": "val", "images": records})
        selected = {"freeze": freeze, "validation_cache_sha256": evaluation.sha256(cache),
                    "threshold_grid": [0.001, 0.01, 0.1], "thresholds": thresholds}
        evaluation.verify_selection(selected, freeze, cache)
        with self.assertRaisesRegex(ValueError, "Frozen model"):
            evaluation.verify_selection(selected, {"weights_sha256": "two"}, cache)
        with self.assertRaises(FileExistsError):
            evaluation.write_json(cache, {})
        cache.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Validation cache changed"):
            evaluation.verify_selection(selected, freeze, cache)

    def test_nms_prefix_and_label_validation(self):
        self.assertIsNone(evaluation.prediction_limit_metadata([0.5], 2, 0.001))
        self.assertTrue(evaluation.prediction_limit_metadata([0.5, 0.0005], 2, 0.001)["threshold_decisions_complete"])
        with self.assertRaisesRegex(ValueError, "truncate"):
            evaluation.prediction_limit_metadata([0.5, 0.001], 2, 0.001)
        for text in ("3 0 0 1 0 1 1", "0 0 0 2 0 1 1", "0 nan 0 1 0 1 1"):
            with self.assertRaises(ValueError):
                evaluation.parse_labels(text)

    def test_weights_export_changes_only_metadata_and_keeps_source(self):
        import torch
        import export_weights
        model = torch.nn.Linear(2, 1)
        model.names = {0: "crack"}
        model.yaml = {"nc": 1, "yaml_file": "/private/model.yaml"}
        model.args = {"data": "/private/data.yaml"}
        source, destination = self.root / "source.pt", self.root / "export.pt"
        torch.save({"model": model, "ema": None, "train_args": model.args}, source)
        digest = evaluation.sha256(source)
        result = export_weights.export(source, destination, digest)
        self.assertTrue(result["state_tensors_identical"])
        self.assertEqual(evaluation.sha256(source), digest)
        restored = torch.load(destination, map_location="cpu", weights_only=False)
        self.assertEqual(restored["model"].args["data"], "work/dataset/data.yaml")
        self.assertTrue(torch.equal(model.weight, restored["model"].weight))
        with self.assertRaises(FileExistsError):
            export_weights.export(source, destination, digest)


if __name__ == "__main__":
    unittest.main()
