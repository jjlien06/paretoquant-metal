#!/usr/bin/env bash
cd "$(dirname "$0")"; S="ssh -o BatchMode=yes msi-vpn"
for i in $(seq 1 120); do o=$($S "docker exec lab-manager python3 -m manager.cli acquire paretoquant 90 --purpose msi-27b-tune3"); echo "$o" | grep -v "not granted" | grep -qi granted && break; sleep 15; done
echo "grant: $o $(date)"
$S "docker exec model-lab python3 -m lab.cli lease paretoquant 90"
for j in $(seq 1 40); do u=$($S "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d '\r '); a=$($S "docker exec model-lab python3 -m lab.cli status" | head -1); echo "$a $u"; [ "$u" -lt 1200 ] && break; sleep 15; done
$S "docker run --rm -i --gpus all -m 14g --memory-swap 14g --entrypoint sh -v model-lab-data:/lab:ro -v pq-27b:/models:ro model-lab:latest -s" < runs.sh > raw.txt 2>&1
echo "done $(date)"
