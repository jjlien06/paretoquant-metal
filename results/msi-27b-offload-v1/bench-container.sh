#!/usr/bin/env bash
# Run llama-bench for Qwen3.8-27B inside a one-off model-lab container (WSL RAM, same engine as the lab).
set -u
cd "$(dirname "$0")"; TAG=${1:-c1}; NGLS=${2:-20 24 28}; mkdir -p $TAG
S="ssh -o BatchMode=yes msi-vpn"
for n in $NGLS; do
  echo "ngl $n start $(date)"
  $S "docker exec lab-manager python3 -m manager.cli acquire paretoquant 75 --purpose msi-27b-offload-$TAG" >/dev/null
  $S "nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader" > $TAG/gpu-before-ngl$n.txt
  $S "docker run --rm --gpus all -m 14g --memory-swap 14g -v model-lab-data:/lab:ro -v pq-27b:/models:ro --entrypoint /lab/engines/llama.cpp@b11968/build/bin/llama-bench model-lab:latest -m /models/Qwen3.8-27B-UD-Q4_K_M.gguf -ngl $n -fa 1 -t 8 -b 2048 -ub 512 -p 512 -n 128 -r 3 -d 0,4096 -o json" > $TAG/ngl$n.json 2> $TAG/ngl$n.err
  echo "ngl $n exit $? $(date)"
done
echo done
