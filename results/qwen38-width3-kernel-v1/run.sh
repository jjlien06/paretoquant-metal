#!/bin/bash
# Usage: run.sh SUFFIX TAU   (runs spikes/031-qwen38-width3-kernel-SUFFIX; native/mtp/adaptive triples)
set -euo pipefail
name=031-qwen38-width3-kernel-$1; tau=$2
repo=/Users/jeremylien/Projects/paretoquant-metal
src=$repo/spikes/$name
sched=$(shasum -a 256 "$src/benchmark_schedule.json" | cut -d' ' -f1)
eval "$($repo/.venv/bin/python -c "import json,sys;r=json.load(open(sys.argv[1]))['requests'];print(f'N={len(r)} CAP={max(x[\"max_tokens\"] for x in r)}')" "$src/benchmark_schedule.json")"
out=$repo/results/qwen38-width3-kernel-v1/run-$1
sysctl kern.memorystatus_vm_pressure_level vm.swapusage
exec $repo/.venv/bin/python -u "$src/supervise.py" \
  --remote-root /Users/jeremylien/Projects/paretoquant-dual-host/spikes/$name \
  --remote-source /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --local-metadata $repo/models/qwen3.8-27b-4bit-tokenizer-only \
  --output "$out" --ssh-key /Users/jeremylien/.ssh/paretoquant_mini_ed25519 \
  --split 28 36 --max-tokens "$CAP" --repeats "$N" --deadline 900 --reserve-bytes 8589934592 \
  --execution phased --prefix-rollback --benchmark-schedule --adaptive-tau "$tau" \
  --local-address 169.254.201.199 --remote-address 169.254.130.250 \
  --ssh-target 'jeremylien@fe80::1456:cf50:88f2:90d9%bridge0' \
  --ssh-host-key-alias 'fe80::56:c507:c02a:a630%bridge0' \
  --eval-group-size 32 --rank0-verifier native --rank1-verifier "${RANK1_VERIFIER:-slot-qmv-tuned-m4}" \
  --target-revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 \
  --target-config-sha256 14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85 \
  --target-index-sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8 \
  --sidecar $repo/models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575 \
  --small-model-local $repo/models/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462 \
  --small-model-remote /Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462 \
  --small-config-sha256 ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772 \
  --small-index-sha256 6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650 \
  --prompt "Explain in two short sentences what a queue is in computer science." \
  --benchmark-schedule-sha256 "$sched"
