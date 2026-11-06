"""Small-model (0.8B control) width-W rollback tests + negative controls.

Device: env DEPTH_TEST_DEVICE=gpu|cpu (default gpu). Loads ONLY the local 0.8B.
"""

import os

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu if os.environ.get("DEPTH_TEST_DEVICE") == "cpu" else mx.gpu)

import small_harness as h  # noqa: E402
from cache_snap import restore, snapshot  # noqa: E402
from prefix_rollback_k import CaptureK  # noqa: E402

MAX_TOKENS = int(os.environ.get("DEPTH_TEST_MAX_TOKENS", "32"))


@pytest.fixture(scope="module")
def loaded():
    model, tok = h.load_small(h.MODEL_DIR)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "List three prime numbers and explain why."}],
        add_generation_prompt=True, enable_thinking=False)
    ref = h.sequential_greedy(model, ids, MAX_TOKENS, set(tok.eos_token_ids))
    return model, tok, ids, ref


@pytest.mark.parametrize("width", [2, 3, 4])
def test_capture_gate_strict(loaded, width):
    model, _, ids, ref = loaded
    g = h.capture_gate(model, ids, ref[0][1:], width, "singleton-qmm")
    assert g["strict_pass"] and g["argmax_pass"], g


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("rate", [0.0, 0.4, 1.0])
def test_e2e_equivalence(loaded, depth, rate):
    model, tok, ids, ref = loaded
    r = h.e2e(model, ids, depth=depth, rate=rate, mode="singleton-qmm", max_tokens=MAX_TOKENS,
              eos_ids=set(tok.eos_token_ids), reference=ref)
    assert r["tokens_equal"], r
    assert r["final_cache_vs_sequential"]["strict_pass"], r


def _prefilled(model, ids):
    c = model.make_cache()
    h.prefill(model, c, ids)
    return snapshot(c)


def test_negative_wrong_keep_is_detected(loaded):
    model, _, ids, ref = loaded
    saved = _prefilled(model, ids)
    forced = ref[0][1:4]
    left, right = model.make_cache(), model.make_cache()
    restore(left, saved), restore(right, saved)
    for t in forced[:1]:  # sequential keep=1
        mx.eval(model(mx.array([[t]]), cache=left), [c.state for c in left])
    cap = CaptureK(model, right, 3)
    with cap:
        mx.eval(model(mx.array([forced]), cache=right), [c.state for c in right])
    cap.commit(2)  # wrong keep
    assert not h.cache_metrics(left, right)["strict_pass"]


def test_negative_skipped_conv_tail_is_detected(loaded, monkeypatch):
    model, _, ids, ref = loaded
    saved = _prefilled(model, ids)
    forced = ref[0][1:4]
    left, right = model.make_cache(), model.make_cache()
    restore(left, saved), restore(right, saved)
    for t in forced[:2]:
        mx.eval(model(mx.array([[t]]), cache=left), [c.state for c in left])
    cap = CaptureK(model, right, 3)
    with cap:
        mx.eval(model(mx.array([forced]), cache=right), [c.state for c in right])
    for r in cap.records:  # sabotage: keep conv tail from the full width
        r["conv_input"] = mx.concatenate(
            [r["conv_input"][:, 1:], r["conv_input"][:, -1:]], axis=1)
    cap.commit(2)
    assert not h.cache_metrics(left, right)["strict_pass"]


def test_negative_no_rollback_is_detected(loaded):
    model, tok, ids, ref = loaded

    class NoRollback(h.SmallBackend):
        def commit_prefix(self, keep):
            self.capture = None  # leaves rejected drafts in every cache

    from speculative_k import generate

    drafter = h.ReplayDrafter(ref[0], model.language_model.args.vocab_size, 0.4, 0)
    b = NoRollback(model, "singleton-qmm")
    r = generate(b, drafter, ids, depth=2, max_tokens=MAX_TOKENS, eos_ids=set(tok.eos_token_ids))
    assert r["token_ids"] != ref[0] or not h.cache_metrics(ref[1], b.cache)["strict_pass"]


def test_capture_rejects_width_mismatch(loaded):
    model, _, ids, _ = loaded
    c = model.make_cache()
    h.prefill(model, c, ids)
    with pytest.raises(ValueError):
        with CaptureK(model, c, 3):
            model(mx.array([[1, 2]]), cache=c)
    with pytest.raises(ValueError):
        CaptureK(model, c, 6)
