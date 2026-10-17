"""Reference MLX adapter: width2 verify, full hybrid snapshots, width1 replay.

The standalone runtime supplies the bounded CPU-stream rank control transport.
Admission is bound to this live model and the explicitly selected numerical mode.
Inputs to native standalone head are NORMALIZED final target hidden states.
"""

import time
import uuid
from contextlib import nullcontext

import mlx.core as mx
from cache_state import restore, snapshot
from mlx_lm.models.cache import KVCache
from overlap_prefill import run_rank0_pipeline, run_rank1_pipeline
from phased_decode import ownership
from prefix_rollback import Capture
from rank_control import Command
from verification import Admission
from verifier import forward as phased_forward


class MTPBackend:
    def __init__(
        self,
        model,
        head,
        *,
        group=None,
        peer=None,
        prefill_step_size=256,
        eval_group_size=1,
        admission=None,
        verifier_mode="native",
        prefix_rollback=False,
        hidden_only_prefill=False,
        prefill_mode="serial",
        progress=None,
    ):
        rank, size = ownership(model, group)
        if not isinstance(admission, Admission):
            raise ValueError("Live same-process numerical admission required")
        admission.require(
            model,
            verifier_mode,
            prefix_rollback=prefix_rollback,
            hidden_only_prefill=hidden_only_prefill,
        )
        self.prefix_rollback = prefix_rollback
        if type(hidden_only_prefill) is not bool:
            raise ValueError("hidden_only_prefill must be a bool")
        self.hidden_only_prefill = hidden_only_prefill
        self.capture = None
        if rank != 0 or (size == 2 and peer is None):
            raise ValueError("Require final rank0 and an explicit peer for two ranks")
        if type(prefill_step_size) is not int or not 1 <= prefill_step_size <= 256:
            raise ValueError("Bounded prefill size required")
        self.model, self.head, self.group, self.peer = model, head, group, peer
        self.prefill_step_size, self.eval_group_size = prefill_step_size, eval_group_size
        self.cache = None
        self.mtp_cache = None
        self.sequence = 0
        self.verifier_mode = verifier_mode
        self.progress = progress or (lambda event: None)
        self._set_prefill_mode(prefill_mode, admission, model, prefill_step_size)

    def _set_prefill_mode(self, prefill_mode, admission, model, width):
        if prefill_mode not in ("serial", "overlap"):
            raise ValueError("Unknown prefill mode")
        self.prefill_mode = prefill_mode
        self.overlapped_prefill = prefill_mode == "overlap"
        if self.overlapped_prefill:
            if not self.hidden_only_prefill or width != 256:
                raise ValueError("Overlap requires hidden-only native256 prefill")
            admission.require_overlap(model, self.eval_group_size)

    def _pipeline(self, command):
        def forward(tokens, *, receive_stage, progress):
            return phased_forward(
                self.model,
                mx.array([tokens], mx.int32),
                self.cache,
                group=self.group,
                eval_group_size=self.eval_group_size,
                all_logits=True,
                return_hidden=True,
                logits=False,
                verifier_mode="native",
                receive_stage=receive_stage,
                progress=progress,
            )

        def head(hidden, shifted, position):
            self.head.forward(
                hidden,
                mx.array([shifted], mx.int32),
                self.mtp_cache,
                position=position,
                logits=False,
            )

        run_rank0_pipeline(command, self.peer, forward, head, self.progress)

    def _command(self, op, tokens=()):
        if getattr(self, "broken", False):
            raise ValueError("Rank raw stream is broken")
        if op == "prefill_pipeline" and (not self.overlapped_prefill or self.peer is None):
            raise ValueError("Compound requires admitted two-rank overlap")
        command = Command(self.request, self.sequence, op, list(tokens))
        command.validate()
        self.sequence += 1
        if self.peer is not None:
            self.peer.begin(command)
        if op == "reset":
            self.cache = self.model.make_cache()
            self.capture = None
        elif op == "prefill_pipeline":
            try:
                self._pipeline(command)
            except BaseException:
                self.broken = True
                raise
        elif op == "snapshot":
            self.saved = snapshot(self.cache)
        elif op == "restore":
            restore(self.cache, self.saved)
            self.capture = None
        elif op == "commit_prefix":
            if not self.prefix_rollback or self.capture is None:
                raise ValueError("No admitted prefix capture")
            self.capture.commit(tokens[0])
            self.capture = None
        elif op in ("prefill_hidden", "forward", "verify"):
            if op == "prefill_hidden" and not self.hidden_only_prefill:
                raise ValueError("Hidden prefill requires explicit opt-in")
            self.capture = (
                Capture(self.model, self.cache) if self.prefix_rollback and op == "verify" else None
            )
            with self.capture if self.capture is not None else nullcontext():
                result = phased_forward(
                    self.model,
                    mx.array([tokens], mx.int32),
                    self.cache,
                    group=self.group,
                    eval_group_size=self.eval_group_size,
                    all_logits=True,
                    return_hidden=True,
                    logits=op != "prefill_hidden",
                    verifier_mode=self.verifier_mode if op == "verify" else "native",
                    progress=self.progress,
                )
        if self.peer is not None:
            ack = self.peer.finish()
            expected = {
                "request": command.request,
                "sequence": command.sequence,
                "op": op,
                "rank": 1,
                "ok": True,
            }
            if ack != expected or any(type(ack[k]) is not type(v) for k, v in expected.items()):
                raise ValueError("Rank1 acknowledgment mismatch")
            if op == "prefill_pipeline":
                self.progress(
                    {
                        "phase": "pipeline_drained",
                        "request_nonce": command.request,
                        "command_sequence": command.sequence,
                        "prefill_mode": "overlap",
                        "pending_handoffs": 0,
                        "rank": 0,
                    }
                )
        self.progress(
            {
                "phase": "rank_action",
                "request": command.request,
                "sequence": command.sequence,
                "op": op,
                "tokens": list(tokens),
                "rank": 0,
                "ok": True,
            }
        )
        if getattr(self, "_gate_observe", False) and op in ("forward", "verify"):
            self._gate_last_output = result
        return result if op in ("prefill_hidden", "forward", "verify") else None

    def fresh(self, prompt):
        if self.overlapped_prefill:
            Command("prompt", 1, "prefill_pipeline", prompt).validate()
        if any(t >= self.model.language_model.args.vocab_size for t in prompt):
            raise ValueError("Prompt outside target vocabulary")
        self.request = uuid.uuid4().hex
        self.sequence = 0
        self._command("reset")
        self.mtp_cache = KVCache()
        if self.overlapped_prefill and len(prompt) > 1:
            self._command("prefill_pipeline", prompt)
        for start in (
            [] if self.overlapped_prefill else range(0, len(prompt) - 1, self.prefill_step_size)
        ):
            end = min(start + self.prefill_step_size, len(prompt) - 1)
            _, hidden = self._command(
                "prefill_hidden" if self.hidden_only_prefill else "forward", prompt[start:end]
            )
            shifted = mx.array([prompt[start + 1 : end + 1]], mx.int32)
            self.head.forward(hidden, shifted, self.mtp_cache, position=start, logits=False)
        scores, hidden = self._command("forward", [prompt[-1]])
        token = int(mx.argmax(scores[:, -1, :], -1).item())
        self.first_token_at = time.perf_counter()
        return token, hidden

    def draft(self, hidden, y):
        scores = self.head.forward(
            hidden, mx.array([[y]], mx.int32), self.mtp_cache, position=self.mtp_cache.offset
        )
        return int(mx.argmax(scores[:, -1, :], -1).item())

    def snapshot(self):
        self._command("snapshot")
        return snapshot([self.mtp_cache])

    def restore(self, saved):
        self._command("restore")
        restore([self.mtp_cache], saved)

    def verify(self, tokens):
        scores, hidden = self._command("verify" if len(tokens) == 2 else "forward", tokens)
        self.verified_output = (
            mx.argmax(scores, -1)[0].tolist(),
            [hidden[:, i : i + 1, :] for i in range(len(tokens))],
        )
        return self.verified_output

    def commit_prefix(self, keep):
        self._command("commit_prefix", [keep])
        predictions, hiddens = self.verified_output
        return predictions[:keep], hiddens[:keep]

    def replay(self, y):
        return self.verify([y])

    def advance_mtp(self, hidden, token):
        self.head.forward(
            hidden,
            mx.array([[token]], mx.int32),
            self.mtp_cache,
            position=self.mtp_cache.offset,
            logits=False,
        )

    def finish(self):
        self._command("finish")


class Rank1Target:
    """Rank1 target-only dispatcher; no MTP head or vocabulary projection."""

    def __init__(
        self,
        model,
        group,
        *,
        eval_group_size=1,
        admission=None,
        verifier_mode="native",
        prefix_rollback=False,
        hidden_only_prefill=False,
        prefill_mode="serial",
        progress=None,
    ):
        if ownership(model, group) != (1, 2):
            raise ValueError("Require first-stage rank1")
        if not isinstance(admission, Admission):
            raise ValueError("Live same-process numerical admission required")
        admission.require(
            model,
            verifier_mode,
            prefix_rollback=prefix_rollback,
            hidden_only_prefill=hidden_only_prefill,
        )
        self.prefix_rollback = prefix_rollback
        if type(hidden_only_prefill) is not bool:
            raise ValueError("hidden_only_prefill must be a bool")
        self.hidden_only_prefill = hidden_only_prefill
        self.capture = None
        self.model, self.group, self.eval_group_size = model, group, eval_group_size
        self.cache = None
        self.verifier_mode = verifier_mode
        self.progress = progress
        MTPBackend._set_prefill_mode(self, prefill_mode, admission, model, 256)

    def dispatch_pipeline(self, command):
        if not self.overlapped_prefill:
            raise ValueError("Compound requires explicit overlap mode")
        if any(t >= self.model.language_model.args.vocab_size for t in command.tokens):
            raise ValueError("Prompt outside target vocabulary")

        def forward(tokens, *, progress):
            phased_forward(
                self.model,
                mx.array([tokens], mx.int32),
                self.cache,
                group=self.group,
                eval_group_size=self.eval_group_size,
                logits=False,
                all_logits=True,
                verifier_mode="native",
                progress=progress,
            )

        run_rank1_pipeline(command, self.channel, forward, self.progress or (lambda event: None))

    def dispatch(self, op, tokens):
        if op == "reset":
            self.cache = self.model.make_cache()
            self.capture = None
        elif op == "snapshot":
            self.saved = snapshot(self.cache)
        elif op == "restore":
            restore(self.cache, self.saved)
            self.capture = None
        elif op == "commit_prefix":
            if not self.prefix_rollback or self.capture is None:
                raise ValueError("No admitted prefix capture")
            self.capture.commit(tokens[0])
            self.capture = None
        elif op in ("prefill_hidden", "forward", "verify"):
            if op == "prefill_hidden" and not self.hidden_only_prefill:
                raise ValueError("Hidden prefill requires explicit opt-in")
            self.capture = (
                Capture(self.model, self.cache) if self.prefix_rollback and op == "verify" else None
            )
            with self.capture if self.capture is not None else nullcontext():
                phased_forward(
                    self.model,
                    mx.array([tokens], mx.int32),
                    self.cache,
                    group=self.group,
                    eval_group_size=self.eval_group_size,
                    logits=False,
                    all_logits=True,
                    verifier_mode=self.verifier_mode if op == "verify" else "native",
                    progress=self.progress,
                )
        elif op == "finish":
            self.saved = None
