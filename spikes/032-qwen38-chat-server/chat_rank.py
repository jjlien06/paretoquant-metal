"""032 interactive serving rank: load once, admit once, then serve chat turns.

Reuses the 031 runtime unchanged (weight streaming, small+full strict width gates,
MTPBackend/Rank1Target, prefix rollback, adaptive depth 1/2). Rank0 (M2) exposes a
loopback-only HTTP API; rank1 (mini) follows bounded per-request rank commands.
One request at a time. Bounds: <=6144 new tokens/turn, prompt+new <= 16384 tokens.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from probe import phase_reporter

MAX_NEW = 6144
CONTEXT = 16384
NOISY = {
    "layer_before", "layer_done", "block_before", "block_done", "embedding_before",
    "embedding_done", "handoff_receive_before", "handoff_receive_done", "handoff_send_before",
    "handoff_send_done", "head_before", "head_done", "token_done", "token_receive_before",
    "token_send_before", "rank_action", "adaptive_round", "prefill_before", "prefill_last_before",
}
GATE_PROMPT = "Explain in two short sentences what a queue is in computer science."


def stream_generate(backend, prompt, *, max_tokens, eos_ids, tau, on_tokens):
    """Round-streaming copies of 031 speculative.generate / generate_adaptive.

    Identical backend op sequence; adds an on_tokens(list)->bool callback after each
    committed round (False = cancel at the round boundary, caches stay consistent).
    """
    eos = set(eos_ids)
    y, hidden = backend.fresh(list(prompt))
    tokens = [y]
    stats = dict(rounds=0, accepted=0, width3_rounds=0, cancelled=False)
    if not on_tokens([y]):
        stats["cancelled"] = True
    while not stats["cancelled"] and len(tokens) < max_tokens and tokens[-1] not in eos:
        if tau is None:  # depth-1 (021 algorithm)
            d = backend.draft(hidden, y)
            backend.snapshot()
            predictions, hiddens = backend.verify([y, d])
            stats["rounds"] += 1
            correction, bonus = predictions
            if correction != d:
                replay_predictions, replay_hidden = backend.commit_prefix(1)
                y, hidden = replay_predictions[0], replay_hidden[0]
                out = [y]
            else:
                stats["accepted"] += 1
                if len(tokens) + 1 == max_tokens or d in eos:
                    _, replay_hidden = backend.commit_prefix(1)
                    hidden = replay_hidden[0]
                    out = [d]
                else:
                    backend.advance_mtp(hiddens[0], d)
                    out = [d, bonus]
                    y, hidden = bonus, hiddens[1]
        else:  # adaptive depth 1/2 (030/031 algorithm)
            remaining = max_tokens - len(tokens)
            d1, p1, d2, p2 = backend.draft_adaptive(
                hidden, y, allow_second=remaining >= 2, min_p1=tau
            )
            drafts = [d1] + ([d2] if d2 is not None and p2 >= tau else [])
            inputs = [y] + drafts
            backend.snapshot()
            predictions, hiddens = backend.verify(inputs)
            stats["rounds"] += 1
            n = 0
            while n < len(drafts) and predictions[n] == drafts[n]:
                n += 1
            out = []
            for t in drafts[:n] + [predictions[n]]:
                out.append(t)
                if t in eos or len(tokens) + len(out) == max_tokens:
                    break
            keep = len(out)
            stats["accepted"] += min(n, keep)
            stats["width3_rounds"] += len(inputs) == 3
            if keep < len(inputs):
                backend.commit_prefix(keep)
            for j in range(keep - 1):
                backend.advance_mtp(hiddens[j], inputs[j + 1])
            y, hidden = out[-1], hiddens[keep - 1]
        tokens.extend(out)
        if not on_tokens(out):
            stats["cancelled"] = True
    stats["finish_reason"] = (
        "cancelled" if stats["cancelled"] else "stop" if tokens[-1] in eos else "length"
    )
    return tokens, stats


def make_sampling_backend(MTPBackend):
    import mlx.core as mx

    class SamplingBackend(MTPBackend):
        """temperature>0: verifier predictions are samples from target softmax(logits/T).

        Accept-draft-iff-sample-equals-draft keeps each emitted token a sample from the
        target distribution given the accepted prefix. NOT covered by the greedy
        exactness admission; untested in 032's smoke test.
        """

        temperature = 0.0

        def _pick(self, scores):
            return mx.random.categorical(scores.astype(mx.float32) / self.temperature, axis=-1)

        def fresh(self, prompt):
            if self.temperature <= 0:
                return super().fresh(prompt)
            self.request = uuid.uuid4().hex
            self.sequence = 0
            self._command("reset")
            from mlx_lm.models.cache import KVCache

            self.mtp_cache = KVCache()
            for start in range(0, len(prompt) - 1, self.prefill_step_size):
                end = min(start + self.prefill_step_size, len(prompt) - 1)
                _, hidden = self._command("forward", prompt[start:end])
                shifted = mx.array([prompt[start + 1 : end + 1]], mx.int32)
                self.head.forward(hidden, shifted, self.mtp_cache, position=start, logits=False)
            scores, hidden = self._command("forward", [prompt[-1]])
            token = int(self._pick(scores[:, -1, :]).item())
            self.first_token_at = time.perf_counter()
            return token, hidden

        def verify(self, tokens):
            if self.temperature <= 0:
                return super().verify(tokens)
            scores, hidden = self._command(
                "verify" if len(tokens) in (2, 3) else "forward", tokens
            )
            self.verified_output = (
                self._pick(scores)[0].tolist(),
                [hidden[:, i : i + 1, :] for i in range(len(tokens))],
            )
            return self.verified_output

    return SamplingBackend


class State:
    def __init__(self):
        self.jobs = queue.Queue(maxsize=1)
        self.busy = threading.Lock()
        self.cancel = threading.Event()
        self.shutdown = threading.Event()
        self.ready = False
        self.info = {}


def http_server(state, tokenizer, port, ui_path):
    ui = Path(ui_path).read_bytes() if ui_path else b"<p>UI not configured</p>"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, value):
            raw = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _local_origin(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            ok_hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
            return host in ok_hosts and (origin is None or origin in {f"http://{h}" for h in ok_hosts})

        def do_GET(self):
            if not self._local_origin():
                return self._json(403, {"error": "loopback host/origin only"})
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(ui)))
                self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(ui)
            elif self.path == "/health":
                self._json(200, {"ready": state.ready, "busy": state.busy.locked(), **state.info})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._local_origin():
                return self._json(403, {"error": "loopback host/origin only"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                return self._json(413, {"error": "body too large"})
            body = self.rfile.read(n) if n else b"{}"
            if self.path == "/shutdown":
                state.cancel.set()
                state.shutdown.set()
                return self._json(200, {"shutdown": True})
            if self.path == "/v1/cancel":
                state.cancel.set()
                return self._json(200, {"cancelled": True})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            if not state.ready or state.shutdown.is_set():
                return self._json(503, {"error": "not ready"})
            try:
                req = json.loads(body)
                messages = req["messages"]
                if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str) for m in messages
                ):
                    raise ValueError("messages must be [{role, content}]")
                max_new = int(req.get("max_tokens", 1024))
                temperature = float(req.get("temperature", 0.0))
                if not 1 <= max_new <= MAX_NEW or not 0.0 <= temperature <= 2.0:
                    raise ValueError(f"max_tokens 1..{MAX_NEW}, temperature 0..2")
                thinking = bool(req.get("enable_thinking", False))
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception as exc:
                return self._json(400, {"error": f"bad request: {exc}"})
            if len(ids) + max_new > CONTEXT:
                return self._json(400, {
                    "error": "context_length_exceeded",
                    "message": f"prompt {len(ids)} + max_tokens {max_new} > {CONTEXT}; start a new chat",
                    "prompt_tokens": len(ids),
                })
            if not state.busy.acquire(blocking=False):
                return self._json(429, {"error": "busy: one request at a time"})
            out = queue.Queue()
            try:
                state.cancel.clear()
                state.jobs.put_nowait(dict(ids=ids, max_new=max_new, temperature=temperature, out=out))
                stream = bool(req.get("stream", False))
                cid = "chatcmpl-" + uuid.uuid4().hex[:16]
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                parts = []
                while True:
                    ev = out.get()
                    if "delta" in ev:
                        parts.append(ev["delta"])
                        if stream:
                            chunk = {"id": cid, "object": "chat.completion.chunk",
                                     "choices": [{"index": 0, "delta": {"content": ev["delta"]}}]}
                            try:
                                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                                self.wfile.flush()
                            except OSError:
                                state.cancel.set()  # client went away
                                stream = False
                                self.close_connection = True
                        continue
                    final = ev
                    break
                if "error" in final:
                    if stream:
                        self.wfile.write(f"data: {json.dumps({'error': final['error']})}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.close_connection = True
                        return
                    return self._json(500, {"error": final["error"]})
                stats = final["stats"]
                if stream:
                    chunk = {"id": cid, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {}, "finish_reason": stats["finish_reason"]}],
                             "usage": stats["usage"], "x_stats": stats}
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
                        self.wfile.flush()
                    except OSError:
                        pass
                    self.close_connection = True
                else:
                    self._json(200, {"id": cid, "object": "chat.completion",
                                     "choices": [{"index": 0, "finish_reason": stats["finish_reason"],
                                                  "message": {"role": "assistant", "content": "".join(parts)}}],
                                     "usage": stats["usage"], "x_stats": stats})
            finally:
                state.busy.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", type=int, nargs=2, default=[28, 36])
    p.add_argument("--eval-group-size", type=int, default=32)
    p.add_argument("--verifier-mode", default="native")
    p.add_argument("--adaptive-tau", type=float, default=None)
    p.add_argument("--target-revision", required=True)
    p.add_argument("--target-config-sha256", required=True)
    p.add_argument("--target-index-sha256", required=True)
    p.add_argument("--small-model", required=True)
    p.add_argument("--small-config-sha256", required=True)
    p.add_argument("--small-index-sha256", required=True)
    p.add_argument("--sidecar")
    streaming = p.add_mutually_exclusive_group(required=True)
    streaming.add_argument("--stream-local", action="store_true")
    streaming.add_argument("--weight-server")
    p.add_argument("--weight-client-bind")
    p.add_argument("--http-port", type=int, default=0)
    p.add_argument("--ui")
    args = p.parse_args()
    # Attributes the reused 031 loaders/gates read.
    args.prefix_rollback = True
    args.verification_gate_only = False
    args.phased = True
    args.single_host = False
    args.parameter_budget_bytes = 0
    args.max_tokens = 96
    args.repeats = 1
    args.eval_group_schedule = None
    args.benchmark_schedule = False
    args.benchmark_schedule_sha256 = None
    args.prompt = GATE_PROMPT
    from remote_actor import source_hashes

    rank_env = int(os.environ.get("MLX_RANK", "0"))
    phase_reporter(rank=rank_env)({"phase": "process_start", "source_sha256": source_hashes(Path(__file__).parent)})
    import mlx.core as mx

    previous = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    try:
        serve(args, mx)
    finally:
        mx.synchronize()
        mx.set_wired_limit(previous)


def load(args, mx, report):
    """Copy of 031 probe.generation loading (weight stream / local stream)."""
    import hashlib
    import socket

    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_tokenizer
    from shard_plan import select_pipeline_rank, text_config
    from stream_weights import checkpoint_inventory, load_from_tensors, receive_json, receive_records, tensor_records

    model_path = Path(args.model).resolve(strict=True)
    config = json.loads((model_path / "config.json").read_text())
    text = text_config(config)
    group = mx.distributed.init(strict=True, backend="ring")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise ValueError("two ranks required")
    from mtp_runtime import small_gate

    small = small_gate(args, [args.eval_group_size], report)
    mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    budget = int(mx.device_info()["max_recommended_working_set_size"] * 0.90)
    split = list(args.split)
    if args.stream_local:
        tensors, provenance = checkpoint_inventory(model_path)
        plan = select_pipeline_rank(
            tensors, model_type=config["model_type"], num_hidden_layers=text["num_hidden_layers"],
            split=split, rank=rank, tie_word_embeddings=text.get("tie_word_embeddings", False),
        )
        selected = {k: tensors[k] for k in plan["selected_keys"]}
        model, loading = load_from_tensors(config, group, split, selected, tensor_records(selected, provenance),
                                           budget_bytes=budget, progress=report)
    else:
        address, port = args.weight_server.rsplit(":", 1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(45)
            connection.bind((args.weight_client_bind, 0))
            connection.connect((address, int(port)))
            token = os.environ["PARETOQUANT_WEIGHT_TOKEN"].encode("ascii")
            connection.sendall(token)
            metadata = receive_json(connection)
            config_bytes = (model_path / "config.json").read_bytes()
            if (metadata["config_text"].encode() != config_bytes
                    or metadata["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
                    or metadata["rank"] != rank or metadata["split"] != split):
                raise ValueError("Server metadata differs from admitted checkpoint/rank/split")
            selected = metadata["tensors"]
            model, loading = load_from_tensors(config, group, split, selected,
                                               receive_records(connection, selected),
                                               budget_bytes=budget, progress=report)
            connection.sendall(b"K")
    tokenizer = load_tokenizer(model_path, {"trust_remote_code": False, "local_files_only": True},
                               eos_token_ids=config.get("eos_token_id"))
    parameter_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    if parameter_bytes > budget:
        raise ValueError("parameter bytes exceed budget")
    mx.eval(model.parameters())
    mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    report({"phase": "loading_done", "parameter_bytes": parameter_bytes,
            "tensor_count": loading.get("tensor_count"),
            "all_retained_tensors_loaded": loading.get("all_retained_tensors_loaded")})
    return model, tokenizer, config, group, small, loading


def serve(args, mx):
    from distributed_control import Channel, serve_request
    from mlx_backend import MTPBackend, Rank1Target
    from mtp_head import TARGET_REVISION, load_trusted_head
    from mtp_runtime import adaptive_enabled, summary
    from verification import Admission, bind_metadata, width_gate

    if args.target_revision != TARGET_REVISION:
        raise ValueError("pinned target revision required")
    bind_metadata(Path(args.model), args.target_config_sha256, args.target_index_sha256)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    serving = {"on": False, "dropped": 0}
    base = phase_reporter(rank=int(os.environ.get("MLX_RANK", "0")), memory=lambda: {
        "active_mlx_bytes": mx.get_active_memory(), "peak_mlx_bytes": mx.get_peak_memory(),
        "cache_mlx_bytes": mx.get_cache_memory()})

    def report(event):
        if serving["on"] and event.get("phase") in NOISY:
            serving["dropped"] += 1
            return
        base(event)

    model, tokenizer, config, group, small, loading = load(args, mx, report)
    rank = group.rank()
    channel = Channel(group)
    size = args.eval_group_size
    from probe import prepare_prompt

    gate_ids = tokenizer.encode(prepare_prompt(tokenizer, GATE_PROMPT), add_special_tokens=False)
    # Same-process strict small+full gates on both GPUs (031 run_loaded admission, unchanged).
    full = width_gate(model, gate_ids, [6, 7, 8, 9, 10, 11], mode=args.verifier_mode, group=group,
                      eval_group_size=size, report=report, prefix_rollback=True,
                      extra_widths=(3,) if adaptive_enabled(args) else ())
    local = {"small": summary(small[size], args.verifier_mode), "full": summary(full, args.verifier_mode)}
    if rank == 0:
        channel.send(local)
        peer = channel.receive()
    else:
        peer = channel.receive()
        channel.send(local)
    admission = Admission(model, args.verifier_mode, small[size], full, peer,
                          prefix_rollback=True, adaptive=adaptive_enabled(args))
    report({"phase": "numerical_admission_done", "verifier": args.verifier_mode,
            "adaptive_tau": args.adaptive_tau, "small_passed": small[size]["passed"],
            "full_passed": full["passed"], "peer": peer})
    serving["on"] = True
    if rank == 1:
        n = 0
        while True:
            msg = channel.receive()
            if msg.get("serve") == "shutdown":
                break
            if msg.get("serve") != "request":
                raise ValueError("unexpected serve frame")
            target = Rank1Target(model, group, eval_group_size=size, admission=admission,
                                 verifier_mode=args.verifier_mode, prefix_rollback=True, progress=report)
            serve_request(channel, target, report)
            del target
            n += 1
            report({"phase": "request_done", "served": n})
        report({"phase": "serve_shutdown", "served": n, "dropped_events": serving["dropped"]})
        print(json.dumps({"rank": 1, "served": n, "status": "clean_shutdown"}), flush=True)
        return
    head = load_trusted_head(args.sidecar, model, target_config=config, target_revision=args.target_revision)
    Backend = make_sampling_backend(MTPBackend)
    state = State()
    state.info = {"mode": "adaptive" if args.adaptive_tau is not None else "depth1",
                  "adaptive_tau": args.adaptive_tau, "rank1_verifier_admitted": peer["full"]["mode"],
                  "rank0_verifier": args.verifier_mode, "max_new_tokens": MAX_NEW, "context_tokens": CONTEXT,
                  "split": args.split, "eval_group_size": size}
    server = http_server(state, tokenizer, args.http_port, args.ui)
    state.ready = True
    report({"phase": "http_ready", "address": ["127.0.0.1", args.http_port]})
    print(json.dumps({"ready": True, "http": ["127.0.0.1", args.http_port]}), flush=True)
    served = 0
    try:
        while not state.shutdown.is_set():
            try:
                job = state.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            out = job["out"]
            try:
                channel.send({"serve": "request"})
                backend = Backend(model, head, group=group, peer=channel, eval_group_size=size,
                                  admission=admission, verifier_mode=args.verifier_mode,
                                  prefix_rollback=True, progress=report)
                backend.temperature = job["temperature"]
                mx.random.seed(int(time.time() * 1000) % (2**31))
                emitted, shown = [], ""
                began = time.perf_counter()
                first = [None]

                def on_tokens(new):
                    if first[0] is None:
                        first[0] = time.perf_counter()
                    emitted.extend(t for t in new if t not in tokenizer.eos_token_ids)
                    nonlocal shown
                    text = tokenizer.decode(emitted, skip_special_tokens=True)
                    if not text.endswith("\ufffd") and len(text) > len(shown) and text.startswith(shown):
                        out.put({"delta": text[len(shown):]})
                        shown = text
                    return not (state.cancel.is_set() or state.shutdown.is_set())

                tokens, stats = stream_generate(
                    backend, job["ids"], max_tokens=job["max_new"], eos_ids=tokenizer.eos_token_ids,
                    tau=args.adaptive_tau, on_tokens=on_tokens)
                backend.finish()
                mx.synchronize()
                end = time.perf_counter()
                text = tokenizer.decode(emitted, skip_special_tokens=True)
                if len(text) > len(shown) and text.startswith(shown):
                    out.put({"delta": text[len(shown):]})
                n = len(tokens)
                stats.update(
                    usage={"prompt_tokens": len(job["ids"]), "completion_tokens": n,
                           "total_tokens": len(job["ids"]) + n},
                    wall_seconds=end - began, first_token_seconds=first[0] - began,
                    post_first_tokens_per_second=(n - 1) / (end - first[0]) if n > 1 and end > first[0] else None,
                    wall_tokens_per_second=n / (end - began),
                    temperature=job["temperature"], mode=state.info["mode"],
                    exactness=("greedy: MTP admitted bit-exact vs native" if job["temperature"] == 0
                               else "sampled: not covered by greedy exactness admission"),
                    timing_convention="EOS included in completion_tokens; prefill included in wall; post-first excludes prefill",
                )
                served += 1
                report({"phase": "request_done", "served": served, "prompt_tokens": len(job["ids"]),
                        "completion_tokens": n, "finish_reason": stats["finish_reason"],
                        "post_first_tps": stats["post_first_tokens_per_second"],
                        "wall_seconds": stats["wall_seconds"]})
                out.put({"stats": stats})
                del backend
            except BaseException as exc:
                out.put({"error": type(exc).__name__})
                raise  # rank state unknown: fail closed, supervisor cleans up
    finally:
        state.ready = False
        server.shutdown()
        server.server_close()
    channel.send({"serve": "shutdown"})
    report({"phase": "serve_shutdown", "served": served, "dropped_events": serving["dropped"]})
    print(json.dumps({"rank": 0, "served": served, "status": "clean_shutdown"}), flush=True)


if __name__ == "__main__":
    main()
