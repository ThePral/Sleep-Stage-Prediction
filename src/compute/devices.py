"""Device visibility guard; this performs no model construction or updates."""


def require_gpu_device():
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError(
            "--require-gpu was requested, but TensorFlow exposes no GPU. "
            "Use the WSL environment and complete the GPU preflight before training."
        )
    return [device.name for device in devices]
