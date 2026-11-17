"""Phase10: freeze checkpoint BN statistics while retaining trainable affine weights."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bn_modules(model):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    require(len(modules) == 66 and all(isinstance(m, nn.BatchNorm2d) and m.track_running_stats and m.affine
                                     for m in modules.values()), "Expected 66 affine BatchNorm2d modules with running statistics.")
    return modules


def snapshot(model):
    values = {}
    for name, module in bn_modules(model).items():
        for key in BUFFERS:
            value = getattr(module, key)
            require(value is not None and torch.isfinite(value).all().item(), "Missing or non-finite BN buffer.")
            values[f"{name}.{key}"] = value.detach().to(device="cpu", dtype=torch.float32 if value.is_floating_point()
                                                        else torch.int64).contiguous().clone()
    return values


def fingerprint(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def equal_buffers(actual, reference):
    return actual.keys() == reference.keys() and all(torch.equal(actual[k], v) for k, v in reference.items())


def write_new(path, value):
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def make_frozen_bn_trainer(base_trainer):
    """Keep native optimizer/validation behavior; replace only BN statistics updates."""
    from ultralytics import __version__
    require(__version__ == "8.4.12", "Use the pinned Ultralytics 8.4.12 trainer.")

    class FrozenBNTrainer(base_trainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            require(isinstance(weights, nn.Module) and not hasattr(self, "_bn_reference"),
                    "Frozen BN requires one loaded checkpoint model for a fresh run.")
            path = Path(weights.pt_path)
            checkpoint_sha = file_sha(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            original = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
            require(isinstance(original, nn.Module), "The source checkpoint has no model or EMA.")
            reference = snapshot(original)
            require(all(not value.is_floating_point() or torch.equal(value, value.half().float())
                        for value in reference.values()), "Source BN statistics must survive native FP16 validation exactly.")
            require(equal_buffers(snapshot(weights), reference), "Loaded checkpoint BN statistics differ from the source file.")
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            require(equal_buffers(snapshot(model), reference) and file_sha(path) == checkpoint_sha,
                    "Training BN initialization differs from the unchanged source checkpoint.")
            self._bn_reference, self._bn_source_sha = reference, checkpoint_sha
            self._bn_reference_sha = fingerprint(reference)
            self._bn_summary = None
            return model

        def _setup_train(self):
            require(not self.args.resume and not self.args.compile and self.world_size == 1 and self.args.workers == 0,
                    "Frozen BN requires a fresh uncompiled single-GPU run with workers=0.")
            require(hasattr(self, "_bn_reference"), "Capture the source BN statistics before setup.")
            super()._setup_train()
            require(equal_buffers(snapshot(self.model), self._bn_reference)
                    and equal_buffers(snapshot(self.ema.ema), self._bn_reference), "Setup changed the initial BN statistics.")
            # Native _model_train() runs after on_train_epoch_start, so freeze in
            # that method rather than a callback which native model.train() undoes.
            self._model_train()
            self._check_bn()
            write_new(Path(self.save_dir) / "frozen_bn/initial.json", {
                "module_count": 66, "source_checkpoint_sha256": self._bn_source_sha,
                "reference_sha256": self._bn_reference_sha, "canonical_float_dtype": "float32",
                "canonical_counter_dtype": "int64", "source_loaded_training_buffers_equal": True,
                "fp16_roundtrip_equal": True, "affine_trainable": True,
                "buffers": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": fingerprint({name: value})} for name, value in self._bn_reference.items()}})
            self.add_callback("on_train_epoch_end", lambda trainer: trainer._finish_bn_epoch())

        def _model_train(self):
            super()._model_train()
            for module in bn_modules(self.model).values():
                module.eval()

        def optimizer_step(self):
            super().optimizer_step()
            # Native EMA averages buffers as well as parameters. Copy only BN
            # buffers to remove repeated floating-point averaging roundoff.
            # Native validation creates inference tensors during half()->float();
            # no_grad alone cannot modify those buffers on the following epoch.
            with torch.inference_mode():
                for name, module in bn_modules(self.ema.ema).items():
                    for key in BUFFERS:
                        value = getattr(module, key)
                        value.copy_(self._bn_reference[f"{name}.{key}"].to(device=value.device, dtype=value.dtype))

        def _check_bn(self):
            training, ema = snapshot(self.model), snapshot(self.ema.ema)
            require(equal_buffers(training, self._bn_reference), "Training BN buffers changed.")
            require(equal_buffers(ema, self._bn_reference), "EMA BN buffers changed.")
            modules, ema_modules = bn_modules(self.model), bn_modules(self.ema.ema)
            require(self.model.training and all(not m.training for m in modules.values()), "Training BN mode is not frozen.")
            require(all(not m.training for m in ema_modules.values()), "EMA BN mode is not evaluation.")
            require(all(m.weight.requires_grad and m.bias.requires_grad for m in modules.values()),
                    "BN affine weights must remain trainable.")
            return {"module_count": len(modules), "training_buffers_equal": True, "ema_buffers_equal": True,
                    "training_bn_eval": True, "ema_bn_eval": True, "affine_trainable": True,
                    "reference_sha256": self._bn_reference_sha, "training_buffers_sha256": fingerprint(training),
                    "ema_buffers_sha256": fingerprint(ema)}

        def _finish_bn_epoch(self):
            epoch = int(self.epoch) + 1
            require(self._bn_summary is None or self._bn_summary["epoch"] + 1 == epoch,
                    "Frozen BN audit epoch was repeated or skipped.")
            summary = {"epoch": epoch, "completed": True, "verified": True, **self._check_bn()}
            relative = f"frozen_bn/epoch_{epoch:03d}.json"
            self._bn_summary = {**summary, "trace_file": relative,
                                "trace_sha256": write_new(Path(self.save_dir) / relative, summary)}

        def frozen_bn_epoch_summary(self):
            require(self._bn_summary is not None and self._bn_summary["epoch"] == int(self.epoch) + 1,
                    "Frozen BN summary requires the completed current epoch.")
            # on_model_save occurs after native validation's half()->float().
            current = self._check_bn()
            require(current == {key: self._bn_summary[key] for key in current},
                    "BN buffers or modes changed during validation.")
            return copy.deepcopy(self._bn_summary)

    return FrozenBNTrainer
