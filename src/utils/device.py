"""Device setup and telemetry shared by training, generation and scorers."""

import torch


def set_device(kind: str, index: int, cpu_threads: int):
    torch.set_num_threads(cpu_threads)
    if kind == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)
    elif kind == "cuda":
        torch.cuda.set_device(index)
    elif kind != "cpu":
        raise ValueError(f"unsupported device {kind}")
    return torch.device(kind if kind == "cpu" else f"{kind}:{index}")


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def peak_memory(device):
    return (
        0
        if device.type == "cpu"
        else getattr(torch, device.type).max_memory_allocated(device)
    )


def device_count(kind: str):
    """Use the job's visible allocation; reward workers live on another node."""
    if kind == "cpu":
        return 1
    if kind == "npu":
        import torch_npu  # noqa: F401
    count = getattr(torch, kind).device_count()
    if count < 1:
        raise RuntimeError(f"no visible {kind} devices")
    return count
