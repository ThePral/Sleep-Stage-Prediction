"""Exercise the actual supervised and SSL kernels before a training campaign.

This check uses synthetic signals only. Device visibility alone does not prove
that a wheel supplies usable kernels for an older GPU's compute capability.
"""

import argparse
from dataclasses import asdict, dataclass
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil


@dataclass(frozen=True)
class PreflightConfig:
    supervised_batch_size: int = 4
    contrastive_batch_size: int = 32
    warmup_steps: int = 2
    measured_steps: int = 5
    require_gpu: bool = True
    temporal_head: str = "legacy"
    temporal_units: int = 64

    def __post_init__(self):
        for name in ["supervised_batch_size", "contrastive_batch_size", "warmup_steps", "measured_steps", "temporal_units"]:
            value = getattr(self, name)
            minimum = 2 if name == "contrastive_batch_size" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.require_gpu, bool):
            raise ValueError("require_gpu must be boolean.")
        if self.temporal_head not in {"legacy", "bilstm"}:
            raise ValueError("temporal_head must be legacy or bilstm.")


def hardware_inventory():
    result = {"platform": platform.platform(), "python": platform.python_version(), "python_executable": sys.executable,
              "logical_cpus": psutil.cpu_count(), "physical_memory_bytes": psutil.virtual_memory().total,
              "workspace_free_bytes": shutil.disk_usage(Path.cwd()).free}
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable:
        try:
            process = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30)
            result["nvidia_smi"] = {"returncode": process.returncode, "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["nvidia_smi"] = {"error": str(exc)}
    else:
        result["nvidia_smi"] = None
    return result


def run_preflight(output_path, config=None):
    config = config or PreflightConfig()
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("A preflight report cannot replace a previous measurement.")
    report = {"schema_version": 1, "config": asdict(config), "hardware": hardware_inventory(),
              "packages": {}, "passed": False, "checks": {}, "data_access": "synthetic_only_no_recordings_or_labels_loaded"}
    try:
        import tensorflow as tf
        from src.models.multiview import ModelConfig, build_multiview_model, compile_multiview_model
        from src.models.contrastive import augment_waveform, build_contrastive_model, nt_xent
        report["packages"] = {p: importlib.metadata.version(p) for p in ["tensorflow", "keras", "numpy", "pandas", "scipy", "mne", "psutil"]}
        report["tensorflow_build"] = tf.sysconfig.get_build_info()
        gpus = tf.config.list_physical_devices("GPU")
        report["tensorflow_devices"] = [d.name for d in tf.config.list_physical_devices()]
        if config.require_gpu and not gpus:
            raise RuntimeError("TensorFlow exposes no GPU. GPU training readiness is unverified.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = "/GPU:0" if gpus else "/CPU:0"
        report["requested_device"] = device
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()

        with tf.device(device):
            architecture = ModelConfig(temporal_head=config.temporal_head, temporal_units=config.temporal_units)
            report["supervised_model_config"] = architecture.to_dict()
            model = compile_multiview_model(build_multiview_model(architecture))
            inputs = {"waveform": tf.random.stateless_normal((config.supervised_batch_size, 10, 3000, 1), [42, 1]),
                      "spectral": tf.random.stateless_normal((config.supervised_batch_size, 10, 10), [42, 2])}
            labels = tf.range(config.supervised_batch_size) % 5
            for _ in range(config.warmup_steps):
                model.train_on_batch(inputs, labels)
            times = []
            for _ in range(config.measured_steps):
                start = time.perf_counter()
                loss = model.train_on_batch(inputs, labels, return_dict=True)["loss"]
                times.append(time.perf_counter() - start)
                if not np.isfinite(loss):
                    raise ValueError("Supervised synthetic update produced a nonfinite loss.")
            probabilities = model(inputs, training=False)
            probabilities.numpy()
            if gpus and "GPU" not in probabilities.device:
                raise RuntimeError("Supervised output did not execute on the requested GPU.")
            report["checks"]["gated_supervised_updates"] = {"passed": True, "parameters": model.count_params(),
                "output_device": probabilities.device, "median_seconds": float(np.median(times)),
                "measured_steps": len(times), "last_loss": float(loss)}
        del model, inputs, labels, probabilities
        tf.keras.backend.clear_session()
        gc.collect()

        with tf.device(device):
            model = build_contrastive_model(ModelConfig(variant="raw", sequence_length=1))
            optimizer = tf.keras.optimizers.Adam(.001, clipnorm=1.)
            raw = tf.random.stateless_normal((config.contrastive_batch_size, 3000, 1), [42, 3])

            @tf.function
            def update(seed):
                seeds = tf.random.experimental.stateless_split(seed, 2)
                first, second = augment_waveform(raw, seeds[0]), augment_waveform(raw, seeds[1])
                with tf.GradientTape() as tape:
                    values = model(tf.concat([first, second], axis=0), training=True)
                    loss, _ = nt_xent(values[:config.contrastive_batch_size], values[config.contrastive_batch_size:])
                gradients = tape.gradient(loss, model.trainable_variables)
                for gradient in gradients:
                    tf.debugging.assert_all_finite(gradient, "Nonfinite SSL gradient")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            for index in range(config.warmup_steps):
                update(tf.constant([42, index])).numpy()
            times = []
            for index in range(config.measured_steps):
                start = time.perf_counter()
                value = update(tf.constant([43, index]))
                loss = float(value.numpy())
                times.append(time.perf_counter() - start)
                if not np.isfinite(loss):
                    raise ValueError("Contrastive synthetic update produced a nonfinite loss.")
            if gpus and "GPU" not in value.device:
                raise RuntimeError("Contrastive loss did not execute on the requested GPU.")
            report["checks"]["contrastive_updates"] = {"passed": True, "parameters": model.count_params(),
                "loss_device": value.device, "original_epochs_per_batch": config.contrastive_batch_size,
                "maximum_negatives_per_anchor": 2 * (config.contrastive_batch_size - 1),
                "median_seconds": float(np.median(times)), "measured_steps": len(times), "last_loss": loss}
        report["tensorflow_gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0") if gpus else None
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["limitations"] = ["Synthetic kernel checks establish neither data-loader throughput nor convergence.",
                              "A passed check applies only to the recorded framework, hardware and batch sizes.",
                              "TF allocator bytes exclude driver reservations; host memory is a different resource."]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--supervised-batch-size", type=int, default=4)
    parser.add_argument("--contrastive-batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--temporal-head", choices=["legacy", "bilstm"], default="legacy")
    parser.add_argument("--temporal-units", type=int, default=64, help="Units per direction for BiLSTM.")
    args = parser.parse_args(argv)
    result = run_preflight(args.report, PreflightConfig(args.supervised_batch_size, args.contrastive_batch_size,
                                                      args.warmup_steps, args.measured_steps, not args.allow_cpu,
                                                      args.temporal_head, args.temporal_units))
    print(json.dumps({"passed": result["passed"], "devices": result.get("tensorflow_devices"),
                      "checks": result["checks"], "error": result.get("error")}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
