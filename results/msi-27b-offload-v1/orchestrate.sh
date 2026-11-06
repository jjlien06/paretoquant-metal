#!/usr/bin/env bash
# Wait for a fair GPU turn on the MSI, run the 27B llama-bench sweep, release.
set -u
H=msi-vpn; S="ssh -o BatchMode=yes -o ConnectTimeout=10 $H"
TAG=${1:-sweep1}; NGLS=${2:-18,22,26,30,0}; MIN=${3:-75}
mgr(){ $S "docker exec lab-manager python3 -m manager.cli $*"; }
for i in $(seq 1 400); do
  out=$(mgr acquire paretoquant $MIN --purpose msi-27b-offload-$TAG 2>&1)
  if echo "$out" | grep -v "not granted" | grep -qi granted; then echo "granted $(date)"; break; fi
  sleep 30
done
echo "$out" | grep -v "not granted" | grep -qi granted || { echo "never granted"; mgr release paretoquant; exit 3; }
for j in $(seq 1 60); do
  r=$($S "docker exec model-lab python3 -m lab.cli lease paretoquant $MIN"); echo "lab lease: $r"
  echo "$r" | grep -q leased && break; sleep 20
done
for j in $(seq 1 60); do
  act=$($S "docker exec model-lab python3 -m lab.cli status" | grep '^activity'); used=$($S "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d '\r ')
  echo "$act used=$used"
  if echo "$act" | grep -q paused && [ "$used" -lt 1200 ]; then break; fi
  [ "$j" = 10 ] && $S "docker exec ollama ollama stop qwen3.5:9b"
  sleep 20
done
$S "powershell -NoProfile -Command \"\$a=New-ScheduledTaskAction -Execute powershell.exe -Argument '-NoProfile -ExecutionPolicy Bypass -File C:\\Users\\jerem\\paretoquant-msi\\bench.ps1 -Ngls $NGLS -Tag $TAG'; \$s=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2); Register-ScheduledTask -TaskName PQ-27B-bench -Action \$a -Settings \$s -Force | Out-Null; Start-ScheduledTask PQ-27B-bench\""
sleep 30
for k in $(seq 1 300); do
  st=$($S "powershell -NoProfile -Command \"(Get-ScheduledTask PQ-27B-bench).State\"" | tr -d '\r ')
  [ "$st" != Running ] && break
  if [ $((k % 5)) = 0 ]; then mgr acquire paretoquant $MIN --purpose msi-27b-offload-$TAG >/dev/null; $S "docker exec model-lab python3 -m lab.cli lease paretoquant $MIN" >/dev/null; fi
  sleep 30
done
$S "powershell -NoProfile -Command \"Get-Content C:\\Users\\jerem\\paretoquant-msi\\results\\$TAG\\run.log\""
$S "docker exec model-lab python3 -m lab.cli release paretoquant"; mgr release paretoquant
echo finished $(date)
