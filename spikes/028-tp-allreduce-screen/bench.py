"""Two-rank MLX ring all-sum latency screen with dependent GPU-produced activations.

Each step: y = quantized_matmul(h, W_rank) (GPU, depends on previous step's reduced
output) -> optional all_sum(y) -> h = rms_norm(.). Warm-up precedes every timed phase.
Phases (per hidden width):
  percall_comm / percall_nocomm : mx.eval after every step, wall time per step.
  token_comm / token_nocomm     : 128 dependent steps built lazily, one eval per token.
Communication overhead = comm - nocomm on the same dependent chain.
No model weights are loaded; W is a random 4-bit [D, D] matrix (~D*D/2 bytes).
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import threading
import time


def watchdog(seconds):
    def fire():
        time.sleep(seconds)
        os._exit(124)

    threading.Thread(target=fire, daemon=True).start()


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary_ms(xs):
    xs_ms = [x * 1e3 for x in xs]
    return {
        "n": len(xs_ms),
        "mean": statistics.fmean(xs_ms),
        "p10": pct(xs_ms, 0.10),
        "p50": pct(xs_ms, 0.50),
        "p90": pct(xs_ms, 0.90),
        "p99": pct(xs_ms, 0.99),
        "min": min(xs_ms),
        "max": max(xs_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--multipliers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--stream", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--percall", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--calls-per-token", type=int, default=128)
    ap.add_argument("--deadline", type=float, default=240)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    watchdog(args.deadline)

    import mlx.core as mx

    t0 = time.perf_counter()
    group = mx.distributed.init(strict=True, backend="ring")
    init_s = time.perf_counter() - t0
    rank, size = group.rank(), group.size()
    if size != 2:
        raise SystemExit("need exactly two ranks")
    comm_stream = mx.cpu if args.stream == "cpu" else None

    def barrier():
        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))

    def allsum(y):
        if comm_stream is None:
            return mx.distributed.all_sum(y, group=group)
        return mx.distributed.all_sum(y, group=group, stream=comm_stream)

    out = {
        "rank": rank,
        "host": platform.node(),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "init_s": init_s,
        "stream": args.stream,
        "dtype": "bfloat16",
        "results": {},
    }
    for mult in args.multipliers:
        d = args.hidden * mult
        mx.random.seed(1234 + rank)
        w_f = mx.random.normal((d, d), dtype=mx.float32) * (1.0 / d**0.5)
        wq, sc, bi = mx.quantize(w_f.astype(mx.bfloat16), group_size=64, bits=4)
        del w_f
        norm_w = mx.ones((d,), dtype=mx.bfloat16)
        h0 = (mx.random.normal((1, d)) * 1.0).astype(mx.bfloat16)
        mx.eval(wq, sc, bi, norm_w, h0)

        def step(h, comm):
            y = mx.quantized_matmul(h, wq, sc, bi, transpose=True, group_size=64, bits=4)
            if comm:
                y = allsum(y)
            return mx.fast.rms_norm(y, norm_w, 1e-6)

        res = {"width": d, "bytes_per_call": d * 2}
        for comm in (True, False):
            tag = "comm" if comm else "nocomm"
            h = h0
            barrier()
            for _ in range(args.warmup):
                h = step(h, comm)
                mx.eval(h)
            barrier()
            times = []
            for _ in range(args.percall):
                s = time.perf_counter()
                h = step(h, comm)
                mx.eval(h)
                times.append(time.perf_counter() - s)
            res[f"percall_{tag}_ms"] = summary_ms(times)
            # per-token lazy chain: 128 dependent steps, one eval
            for _ in range(3):  # token-shape warmup
                for _ in range(args.calls_per_token):
                    h = step(h, comm)
                mx.eval(h)
            barrier()
            ttimes = []
            for _ in range(args.tokens):
                s = time.perf_counter()
                for _ in range(args.calls_per_token):
                    h = step(h, comm)
                mx.eval(h)
                ttimes.append(time.perf_counter() - s)
            res[f"token_{tag}_ms"] = summary_ms(ttimes)
            hv = h.astype(mx.float32)
            mx.eval(hv)
            res[f"final_{tag}_finite"] = bool(mx.all(mx.isfinite(hv)).item())
            res[f"final_{tag}_sha256"] = hashlib.sha256(bytes(memoryview(hv))).hexdigest()
        out["results"][str(mult)] = res
        del wq, sc, bi
    out["peak_memory_bytes"] = mx.get_peak_memory()
    barrier()
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"rank": rank, "done": True, "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
