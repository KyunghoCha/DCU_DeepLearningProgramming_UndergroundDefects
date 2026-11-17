"""Export inference weights with relative metadata and identical model tensors."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickletools
import tempfile
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative_metadata(value):
    if isinstance(value, dict):
        return {key: relative_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(relative_metadata(item) for item in value)
    if isinstance(value, (str, Path)) and str(value).startswith("/"):
        path = Path(value)
        if path.name == "data.yaml":
            return "work/dataset/data.yaml"
        if path.name == "best.pt":
            return "work/train_coco/weights/best.pt"
        if path.name == "runs":
            return "work"
        if path.name == "frozenbn960_lr15e6_e30":
            return "work/train_frozenbn"
        return path.name
    return value


def export(source, destination, expected_sha):
    import torch
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if source == destination or destination.exists() or destination.is_symlink():
        raise FileExistsError("Export requires a new destination file.")
    if sha256(source) != expected_sha:
        raise ValueError("Source checkpoint SHA-256 differs.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    original = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model = copy.deepcopy(original)
    for module in model.modules():
        for key, value in list(vars(module).items()):
            if key in ("args", "yaml", "yaml_file", "pt_path") or isinstance(value, (str, Path)):
                setattr(module, key, relative_metadata(value))
    result = {key: relative_metadata(value) for key, value in checkpoint.items()
              if key not in ("model", "ema", "optimizer", "scaler")}
    result.update(model=model, ema=None, optimizer=None, scaler=None, updates=None, epoch=-1, best_fitness=None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".weights-", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(result, stream)
    try:
        restored = torch.load(temporary, map_location="cpu", weights_only=False)["model"]
        expected, actual = original.state_dict(), restored.state_dict()
        if expected.keys() != actual.keys() or any(
                expected[key].dtype != actual[key].dtype or not torch.equal(expected[key], actual[key]) for key in expected):
            raise ValueError("Export changed a model tensor.")
        if original.names != restored.names or relative_metadata(original.yaml) != restored.yaml:
            raise ValueError("Export changed the model classes or architecture.")
        with zipfile.ZipFile(temporary) as archive:
            metadata = archive.read(next(name for name in archive.namelist() if name.endswith("/data.pkl")))
            if any(isinstance(value, str) and value.startswith("/") for _, value, _ in pickletools.genops(metadata)):
                raise ValueError("Absolute path remains in checkpoint metadata.")
        if sha256(source) != expected_sha:
            raise ValueError("Source checkpoint changed during export.")
        os.link(temporary, destination)
        return {"source_sha256": expected_sha, "distribution_sha256": sha256(destination),
                "distribution_file": destination.name, "state_tensor_count": len(actual),
                "state_tensors_identical": True, "class_mapping_identical": True,
                "architecture_identical": True, "metadata_paths": "relative"}
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    # Keep Ultralytics import settings outside the user's home directory.
    with tempfile.TemporaryDirectory(prefix="export-runtime-") as runtime:
        os.environ.update(YOLO_CONFIG_DIR=runtime, YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", YOLO_VERBOSE="false")
        print(json.dumps(export(args.source, args.output, args.expected_sha256), indent=2))
