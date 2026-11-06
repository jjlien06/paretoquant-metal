#!/usr/bin/env bash
cd "$(dirname "$0")"; S="ssh -o BatchMode=yes msi-vpn"
$S "docker exec model-lab python3 -m lab.cli lease paretoquant 120"
for j in $(seq 1 40); do u=$($S "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d '\r '); echo "vram $u"; [ "$u" -lt 1200 ] && break; sleep 15; done
$S "docker run --rm -i --gpus all -m 14g --memory-swap 14g --entrypoint sh -v model-lab-data:/lab:ro -v pq-27b:/models:ro model-lab:latest -s" < runs.sh > raw.txt 2>&1
echo "bench done $(date)"
$S "docker exec model-lab python3 -m lab.cli release paretoquant"; $S "docker exec lab-manager python3 -m manager.cli release paretoquant"
$S 'cmd /c "del C:\Users\jerem\labs-campaign\labs24-20261028\STOP"'; $S 'cmd /c "dir C:\Users\jerem\labs-campaign\labs24-20261028"' | grep -c STOP
echo "cleanup done $(date)"
