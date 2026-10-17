"""Opt-in actual tiny packed-GPU integration; no performance extrapolation."""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def run_cases():
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import phased_decode
    import speculative_decode
    from mlx_lm.models import cache as cache_module
    from mlx_lm.models.qwen2 import Model, ModelArgs

    class FixtureTokenizer:
        # Synthetic IDs for actual randomly initialized NN weights, not checkpoint tokens.
        bos_token = None
        bos_token_id = None

        def __init__(self, eos=()):
            self.eos_token_ids = set(eos)

        def get_vocab(self):
            return {f"fixture-{i}": i for i in range(128)}

        def decode(self, ids, skip_special_tokens=True):
            return ",".join(map(str, ids))

    def model(seed, tied):
        mx.random.seed(seed)
        result = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=64,
                num_hidden_layers=3,
                intermediate_size=128,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=128,
                num_key_value_heads=2,
                tie_word_embeddings=tied,
            )
        )
        result.set_dtype(mx.float16)
        nn.quantize(result, group_size=32, bits=4)
        mx.eval(result.parameters())
        return result

    rows, created = [], []
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    old_cache_limit = mx.set_cache_limit(256 * 1024 * 1024)
    mx.reset_peak_memory()
    original_make_cache = cache_module.make_prompt_cache

    def tracked_cache(owner):
        cache = original_make_cache(owner)
        created.append(cache)
        return cache

    try:
        for tied in (False, True):
            teacher, different_draft = model(49, tied), model(117, tied)
            prompt = [1, 2, 3, 4]
            untimed = phased_decode.phased_generate(
                teacher,
                FixtureTokenizer(),
                prompt,
                max_tokens=11,
                eval_group_size=8,
            )
            for group in (1, 8):
                for k in (1, 2, 4, 8):
                    for mode in ("matched", "different", "eos-first", "eos-third", "last-budget"):
                        eos = ()
                        if mode == "eos-first":
                            eos = (untimed["token_ids"][0],)
                        elif mode == "eos-third":
                            eos = (untimed["token_ids"][2],)
                        tokenizer = FixtureTokenizer(eos)
                        budget = 1 if mode == "last-budget" else 11
                        draft = different_draft if mode == "different" else teacher
                        created.clear()
                        with patch.object(cache_module, "make_prompt_cache", tracked_cache):
                            control = phased_decode.phased_generate(
                                teacher,
                                tokenizer,
                                prompt,
                                max_tokens=budget,
                                eval_group_size=group,
                                prefill_step_size=2,
                            )
                            events = []
                            trial = speculative_decode.speculative_generate(
                                teacher,
                                tokenizer,
                                prompt,
                                draft_model=draft,
                                draft_tokenizer=tokenizer,
                                max_tokens=budget,
                                draft_tokens=k,
                                eval_group_size=group,
                                prefill_step_size=2,
                                progress=events.append,
                            )
                        assert len(created) == 3
                        control_cache, target_cache, draft_cache = created
                        offset = len(prompt) + len(control["token_ids"]) - 1
                        equal_ids = trial["token_ids"] == control["token_ids"]
                        equal_stop = trial["finish_reason"] == control["finish_reason"]
                        offsets_equal = all(
                            state.offset == offset for cache in created for state in cache
                        )
                        kv_close = all(
                            np.allclose(
                                np.array(left[..., :offset, :]),
                                np.array(right[..., :offset, :]),
                                rtol=0.002,
                                atol=0.002,
                            )
                            for a, b in zip(control_cache, target_cache)
                            for left, right in zip(a.state[:2], b.state[:2])
                        )
                        draft_kv_close = None
                        if draft is teacher:
                            draft_kv_close = all(
                                np.allclose(
                                    np.array(left[..., :offset, :]),
                                    np.array(right[..., :offset, :]),
                                    rtol=0.002,
                                    atol=0.002,
                                )
                                for a, b in zip(control_cache, draft_cache)
                                for left, right in zip(a.state[:2], b.state[:2])
                            )
                        rounds = [e for e in events if e["phase"] == "speculative_round_done"]
                        token_events = [e["token_id"] for e in events if e["phase"] == "token_done"]
                        stats = trial["acceptance_stats"]
                        accounting = (
                            len(rounds) == stats["rounds"] == trial["target_verification_calls"]
                            and sum(e["accepted"] for e in rounds) == stats["accepted"]
                            and sum(e["proposed"] for e in rounds) == stats["proposed"]
                            and sum(e["emitted"] for e in rounds) == len(trial["token_ids"])
                            and token_events == trial["token_ids"]
                        )
                        row = {
                            "tied": tied,
                            "eval_group": group,
                            "k": k,
                            "mode": mode,
                            "control_ids": control["token_ids"],
                            "speculative_ids": trial["token_ids"],
                            "token_ids_equal": equal_ids,
                            "stop_equal": equal_stop,
                            "all_layer_offsets_equal": offsets_equal,
                            "teacher_live_kv_close": kv_close,
                            "matched_draft_live_kv_close": draft_kv_close,
                            "accounting": accounting,
                            "acceptance_stats": stats,
                            "rounds": rounds,
                            "finish_reason": trial["finish_reason"],
                        }
                        row["passed"] = all(
                            (equal_ids, equal_stop, offsets_equal, kv_close, accounting)
                        )
                        if draft_kv_close is not None:
                            row["passed"] &= draft_kv_close
                        rows.append(row)
        return {
            "passed": all(r["passed"] for r in rows),
            "cases": rows,
            "case_count": len(rows),
            "scope": (
                "Actual random-weight tiny 4bit FP16 Qwen2 GPU; "
                "not real-checkpoint or distributed parity"
            ),
            "weights": "Three layers, hidden64, vocab128, group32; no model downloads",
            "tolerance": {"rtol": 0.002, "atol": 0.002},
            "sources": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("speculative_decode.py", "phased_decode.py", Path(__file__).name)
            },
            "peak_mlx_bytes": mx.get_peak_memory(),
            "device": mx.device_info(),
        }
    finally:
        mx.set_cache_limit(old_cache_limit)
        mx.set_default_device(previous)


@unittest.skipUnless(os.environ.get("PARETOQUANT_SPEC_GPU") == "1", "Parent opt-in actual GPU gate")
class ActualGpuTests(unittest.TestCase):
    def test_real_packed_teacher_greedy_tokens_and_live_caches(self):
        self.assertTrue(run_cases()["passed"])


if __name__ == "__main__":
    # Parent provides an exclusive evidence path inside a bounded remote actor.
    report = run_cases()
    with Path(sys.argv[1]).open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "cases": report["case_count"]}))
    raise SystemExit(0 if report["passed"] else 1)
