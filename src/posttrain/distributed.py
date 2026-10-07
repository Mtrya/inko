"""Learner-only collectives and explicit global sample-mean reduction."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def initialize(rank, world, device, rendezvous, timeout):
    dist.init_process_group(
        "hccl" if device.type == "npu" else "nccl" if device.type == "cuda" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout),
    )


def wrap(module, device):
    return DistributedDataParallel(
        module,
        device_ids=None if device.type == "cpu" else [device.index],
        broadcast_buffers=False,
    )


def prime_buckets(module, *args):
    """Build the steady DDP reduction layout without an optimizer/RNG update.

    DDP rebuilds buckets after its first backward. A fresh resumed wrapper would
    otherwise reduce its next gradients with a different layout from the live
    wrapper. Floating-point HCCL reductions can differ across those layouts.
    The next real forward applies the rebuild requested by this zero backward.
    """
    (module(*args).float().sum() * 0).backward()
    module.zero_grad(set_to_none=True)


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_loss(local_losses, global_samples):
    return local_losses.float().sum() * (world_size() / global_samples)


def unanimous_finite(valid, device):
    failed = torch.tensor(int(not valid), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise FloatingPointError(
            "nonfinite role loss/gradient/parameter/optimizer state; recover last committed round"
        )


def barrier():
    if dist.is_initialized():
        dist.barrier()
