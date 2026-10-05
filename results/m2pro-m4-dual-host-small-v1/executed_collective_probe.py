"""Standalone strict two-rank MLX ring communication smoke probe."""
import argparse
import importlib.metadata
import json

import mlx.core as mx


def collective():
    mx.set_default_device(mx.gpu)
    group = mx.distributed.init(strict=True, backend="ring")
    rank, size = group.rank(), group.size()
    if size != 2 or rank not in (0, 1):
        raise ValueError("This smoke probe requires exactly two ranks")
    total = mx.distributed.all_sum(mx.array(rank + 1), group=group)
    gathered = mx.distributed.all_gather(mx.array([rank]), group=group)
    mx.eval(total, gathered)
    if total.item() != 3 or gathered.tolist() != [0, 1]:
        raise RuntimeError("Collective numerical check failed")
    if rank == 0:
        mx.eval(mx.distributed.send(mx.array([11.0, 12.0]), 1, group=group))
        received = mx.distributed.recv((2,), mx.float32, 1, group=group)
    else:
        received = mx.distributed.recv((2,), mx.float32, 0, group=group)
        mx.eval(received)
        mx.eval(mx.distributed.send(mx.array([21.0, 22.0]), 0, group=group))
    mx.eval(received)
    return {
        "mode": "collective", "rank": rank, "world_size": size,
        "all_sum": total.item(), "all_gather": gathered.tolist(),
        "received": received.tolist(), "mlx_version": mx.__version__,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device": mx.device_info(), "peak_mlx_bytes": mx.get_peak_memory(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["collective"], required=True)
    parser.parse_args()
    print(json.dumps(collective(), allow_nan=False))


if __name__ == "__main__":
    main()
