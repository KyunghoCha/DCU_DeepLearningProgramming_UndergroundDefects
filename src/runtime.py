"""Keep runtime files local and bound CPU/GPU use before importing Ultralytics."""
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess


def gpu_processes():
    output = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True, timeout=10)
    return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}


def configure(directory, device):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", YOLO_VERBOSE="false",
                      YOLO_CONFIG_DIR=str(directory), MPLCONFIGDIR=str(directory / "matplotlib"),
                      TORCH_HOME=str(directory / "torch"), XDG_CACHE_HOME=str(directory / "cache"),
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      WANDB_MODE="disabled", COMET_MODE="DISABLED",
                      CUDA_VISIBLE_DEVICES="0" if device == "0" else "", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    font = importlib.metadata.distribution("matplotlib").locate_file("matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")
    (directory / "Ultralytics").mkdir()
    shutil.copyfile(font, directory / "Ultralytics/Arial.ttf")
    if device == "0":
        usage = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used",
                                         "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
        utilization, memory = map(int, usage.split(","))
        if gpu_processes() or utilization != 0 or memory >= 128:
            raise RuntimeError("GPU 0 is busy; no model was loaded.")
    import torch
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
    from ultralytics.utils.torch_utils import init_seeds
    if USER_CONFIG_DIR.resolve() != directory / "Ultralytics":
        raise RuntimeError("Ultralytics was imported before its runtime directory was configured.")
    integrations = ("sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb")
    SETTINGS.update(**{key: False for key in integrations if key in SETTINGS},
                    runs_dir=str(directory / "runs"), weights_dir=str(directory / "weights"),
                    datasets_dir=str(directory / "datasets"))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    def guard(_=None):
        if device == "0" and gpu_processes() - {os.getpid()}:
            raise RuntimeError("Another GPU compute process appeared; stopping this run.")

    if device == "0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the CUDA requirements or use --device cpu.")
        guard()
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
        probe = torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()
        guard()
        del probe
    init_seeds(42, deterministic=True)
    return guard, {"torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
                   "cuda": torch.version.cuda, "device": device,
                   "gpu": torch.cuda.get_device_name(0) if device == "0" else None,
                   "allocator_fraction": 0.5 if device == "0" else None, "cpu_threads": 2}
