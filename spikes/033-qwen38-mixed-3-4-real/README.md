# Spike 033: Qwen3.8-27B real-activation mixed precision

All CPU code and data in this directory are new. Original `stream.py` and its
failed receipts are retained; executed continuation candidate is `stream_v2.py`.
Use project `.venv/bin/python` locally; never import/evaluate MLX locally while
the existing chat occupies the M2. The mini is the only experimental GPU owner.

## Physical mini entry points

Working directory:
`/Users/jeremylien/Projects/paretoquant-dual-host/spikes/033-qwen38-mixed-3-4-real`

Interpreter:
`/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python`

Set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=$PWD/dependency`.
The dependency allocator is a byte-identical copy of the project's unchanged
`src/paretoquant/allocator.py`; native model math comes from installed MLX-LM.

Every inference/build step is wrapped as:

    $PY guard.py --receipt UNIQUE-guard.jsonl --seconds 1200 -- $PY SCRIPT [ARGS]

Do not rerun in-place: receipt/checkpoint outputs are exclusive. Create fresh
output filenames/directories, preserve failures, and recheck both live chat/resource
conditions. `guard.py` checks normal/warning pressure, RSS<3GiB, observation gaps,
child deadline and reaping. It only signals its own direct unreaped child.

Executed continuation commands inside that wrapper:

    stream_v2.py small \
      --model /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462 \
      --output small-parity-v2.json

    stream_v2.py calibrate \
      --model /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
      --q3 "$PWD/q3-source" --data "$PWD/data/train.json" \
      --output sensitivity-v2.json

    audit_inputs.py

Local exact allocation (CPU only):

    .venv/bin/python spikes/033-qwen38-mixed-3-4-real/allocate_real.py \
      --inventory results/qwen38-mixed-3-4-real-v1/actual-inventory-v2.json \
      --sensitivity results/qwen38-mixed-3-4-real-v1/sensitivity-v2.json \
      --output results/qwen38-mixed-3-4-real-v1/allocator-plans-v2.json

Plan bytes are copied to the mini with a verified deployment hash. Mini:

    build_mixed.py \
      --q4 /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
      --q3 "$PWD/q3-source" --plan "$PWD/allocator-plans-v2.json" \
      --plan-index 0 --output "$PWD/checkpoint-mixed05"

Repeat at plan-index2 to `checkpoint-mixed15`. The serial driver
`finish_stages.py` contains the actual guarded commands, limits, prerequisites and
exclusive log names; it was run after the separately guarded uniform-q4 evaluation.
It builds/evaluates mixed05, builds/evaluates mixed15, then evaluates uniformq3.

For each of q4 source / mixed05 / mixed15 / q3 source:

    stream_v2.py evaluate --model CHECKPOINT_PATH \
      --data "$PWD/data/test.json" --output UNIQUE-evaluation.json

No 16-row head chunking: every sequence's 256 rows are projected together.
Perplexity uses 24 windows and 6,120 targets. The original 16-row `stream.py`
is historical and must not be substituted for this candidate.

## Evidence verification and collection

`assure_v2.py` is a CPU-only post-hoc checker. It recomputes allocation, verifies
actual train/test IDs and coverage, independently reads every source and mixed
tensor byte range against loaded hashes, binds actual headers/config overrides,
checks pinned source LFS shard hashes, and checks guard exits/samples. The earlier
unexecuted `assure.py` is retained as a reviewed but superseded candidate.

`remote.py` / `deploy_extra.py`: pinned strict-host-key SSH, exclusive source
uploads and readback hashes. `collect.py` preserves initial evidence;
`collect_v2.py` transfers continuation receipts/configs/manifests, not weights,
with independent remote/local SHA-256 equality.

`prepare_data.py` used pinned Salesforce WikiText parquet artifacts and the
project tokenizer on CPU. Its optional pyarrow dependency was installed only into
this new directory's `deps/`; no existing venv package was modified.

CPU utility tests:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
      -s spikes/033-qwen38-mixed-3-4-real -p test_allocation_build.py -v

`cleanup_check_v2.py` records final pgrep/process, pressure, disk and pre-existing
chat/listener readbacks. `cleanup_check.py` is retained but has an overescaped
relative-name regex. Always use v2 and exclude only the checker invocation/ancestors.

Final human report:
`results/qwen38-mixed-3-4-real-v1/README.md`.
