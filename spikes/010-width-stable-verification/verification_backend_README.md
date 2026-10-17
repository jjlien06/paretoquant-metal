# Verification-forward safety boundary

`verification_backend.py` exposes:

```python
forward = make_verification_forward(
    original_forward, mx=mx, qwen2=qwen2, qmm=qmm, attention=attention
)
result = forward(model, inputs, cache, **kwargs)
```

All dependencies are supplied by the caller. The implementation imports only
stdlib `contextlib.ExitStack` and `unittest.mock.patch`; it does not import MLX,
NumPy, models, kernels, decoder logic, or supervision/probe modules. This is a
forwarding boundary, **not decoder integration or a general global plugin**.

## Selection and metadata rejection

| Call metadata | Behavior |
| --- | --- |
| `all_logits` absent or exactly `False` | Call original unchanged; never inspect `inputs.shape`. |
| Present non-boolean `all_logits` | Raise `TypeError` before input shape access, patch writes, or original invocation. |
| Exactly `True`, shape `(1, 1)` | Call original unchanged, without installing patches. |
| Exactly `True`, shape `(1, L)`, `2 <= L <= 9` | Call original inside scoped component patches. |
| Exactly `True`, missing/invalid shape or unsupported width/batch | Raise `ValueError` before patches or original invocation. |

Active-path shapes must be exact built-in tuples or lists with two exact
built-in integer dimensions. Booleans, floats, arbitrary metadata wrappers,
container subclasses, non-unit batches, empty widths, and widths above nine
are rejected. Only shape metadata is read: no array evaluation,
materialization, operand-shape inspection, or tensor comparison is performed.
An exception raised by a shape property propagates without installing patches.
Caller-supplied shape accessors themselves must be metadata-only.

Metadata rejection does not invoke either injected component, change patch
targets, or modify the cache. The original callable receives the same model,
input, cache, keyword values, and keyword presence (including `all_logits`).
Its returned object and raised exception are preserved by identity. The boundary
does not copy or roll back cache mutations performed by the original callable.

## Scoped components

For an eligible verification call, `mx.quantized_matmul` is temporarily replaced
by an adapter, and `qwen2.scaled_dot_product_attention` by the **injected attention
callable itself**, not an additional attention wrapper. The injected attention
component owns its internal native-SDPA delegation.

The QMM adapter accepts `x, weights, scales, biases=None`, with keyword-only
`transpose=False, group_size=64, bits=4, mode="affine", stream=None`. Before
calling the injected QMM it requires:

- `transpose is True` (not a truthy equivalent);
- exact built-in string `mode == "affine"`;
- `stream is None`;
- `biases is not None` (no truth evaluation of biases).

An unsupported contract raises `ValueError`, not `AssertionError`; unknown
keywords raise `TypeError` through the explicit adapter signature. Supported
calls become exactly:

```python
qmm(x, weights, scales, biases, bits=bits, group_size=group_size)
```

Operands and quantization keyword values retain their identities. Numerical,
layout, and quantization-metadata validation belong to the injected component.
There is no fallback to native QMM after contract rejection.

Both bindings are restored on normal return, original/component exceptions
(including `BaseException`), or second-patch setup failure. Nested scopes restore
the enclosing bindings first, then the preexisting bindings. Bypass means no
new patches: it does not temporarily undo an already-active enclosing scope.

## Execution isolation — mandatory

Scoped patches are **process-global and NOT concurrency-safe**. Use only in a
single-threaded isolated actor, with one worker per GPU. The intended integration
target is two physical ranks, each owning its separate worker process and GPU;
this module neither starts ranks nor manages GPUs. Do not run concurrent model
calls, thread pools, interleaved async tasks, or unrelated consumers of the patched
module functions in one process while a scope is active.

Intended phase routing: teacher verification alone uses explicit
`all_logits=True`; prefill/control use absent/False. Direct draft-model calls
remain outside this adapter. No phase, core, speculative-decoding, or kernel
source files are changed here.

## CPU evidence and reproduction

From `/Users/jeremylien/Projects/paretoquant-metal`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest spikes/010-width-stable-verification/test_verification_backend.py -q --noconftest -p no:cacheprovider
.venv/bin/ruff check --no-cache spikes/010-width-stable-verification/verification_backend.py spikes/010-width-stable-verification/test_verification_backend.py
.venv/bin/ruff format --check --no-cache spikes/010-width-stable-verification/verification_backend.py spikes/010-width-stable-verification/test_verification_backend.py
```

The suite uses only stdlib test doubles and `unittest`; pytest is the existing
CPU test runner. Bytecode and pytest/Ruff caches are disabled to avoid writes
outside the four assigned files. Expected final coverage: **20 tests and 61
subtests**. These are structural safety tests, not numerical comparisons,
model/distributed admission, full-model correctness, or speed measurements.

`verification-backend-TDD.json` records literal command output and exit status
for eight incremental RED→GREEN production slices, subsequent regression checks,
formatting/lint checks, and the explicit final CPU run. The initial missing-module
failure is the first API RED. One intermediate shape-test run failed because of
an invalid test-fixture class assignment; it is recorded and was corrected before
the genuine shape-admission RED. Additional scoped-cleanup and metadata tests
exercise behavior already provided by the earlier GREEN implementation and are
honestly labeled regression checks, not fabricated RED runs.
