#!/bin/bash
# Start the two-Mac Qwen3.8-27B chat server (persistent until stop-chat.sh).
# Usage: start-chat.sh [--depth1] [--port N] [--hours H]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
KEY=/Users/jeremylien/.ssh/paretoquant_mini_ed25519
ALIAS='fe80::56:c507:c02a:a630%bridge0'
TARGET="${MINI_SSH_TARGET:-jeremylien@fe80::1456:cf50:88f2:90d9%bridge0}"
MODE=adaptive; PORT=8780; HOURS=8
while [ $# -gt 0 ]; do
  case "$1" in
    --depth1) MODE=depth1 ;;
    --port) PORT="$2"; shift ;;
    --hours) HOURS="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
RUNS="$HERE/runs"; mkdir -p "$RUNS"
CUR="$RUNS/current"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o "HostKeyAlias=$ALIAS" -o IdentitiesOnly=yes -o ConnectTimeout=5 -i "$KEY" "$TARGET")

if [ -e "$CUR" ]; then
  pid=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]+"/supervisor.json"))["pid"])' "$CUR" 2>/dev/null || echo "")
  if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q chat_supervise.py; then
    echo "Already running (supervisor pid $pid, run $(readlink "$CUR")). Use stop-chat.sh first."; exit 1
  fi
  rm -f "$CUR"
fi

echo "== preflight: bridge"
LOCAL_IP=$(ifconfig bridge0 | awk '/inet /{print $2; exit}')
REMOTE_IP=$("${SSH[@]}" "ifconfig bridge0 | awk '/inet /{print \$2; exit}'")
[ -n "$LOCAL_IP" ] && [ -n "$REMOTE_IP" ] || { echo "bridge0 IPv4 missing (local=$LOCAL_IP remote=$REMOTE_IP)"; exit 1; }
lif=$(route -n get "$REMOTE_IP" | awk '/interface:/{print $2}')
if [ "$lif" != bridge0 ]; then ping -b bridge0 -c 2 -W 1000 -t 4 "$REMOTE_IP" >/dev/null 2>&1 || true; lif=$(route -n get "$REMOTE_IP" | awk '/interface:/{print $2}'); fi
rif=$("${SSH[@]}" "route -n get $LOCAL_IP | awk '/interface:/{print \$2}'")
if [ "$rif" != bridge0 ]; then "${SSH[@]}" "ping -b bridge0 -c 2 -W 1000 -t 4 $LOCAL_IP >/dev/null 2>&1 || true"; rif=$("${SSH[@]}" "route -n get $LOCAL_IP | awk '/interface:/{print \$2}'"); fi
echo "M2 $LOCAL_IP -> mini $REMOTE_IP via $lif; mini -> M2 via $rif"
[ "$lif" = bridge0 ] && [ "$rif" = bridge0 ] || { echo "FAIL: routes must use bridge0 both ways"; exit 1; }

echo "== preflight: memory/disk"
check_host() { # $1=label, rest = pressure swap free_kb
  local label=$1 pressure=$2 free_kb=$3 swap=$4
  echo "$label: pressure=$pressure free=$((free_kb/1024/1024))GiB swap: $swap"
  [ "$pressure" != 4 ] || { echo "FAIL: $label critical memory pressure"; exit 1; }
  [ "$free_kb" -gt $((10*1024*1024)) ] || { echo "FAIL: $label <10 GiB free disk"; exit 1; }
}
check_host M2 "$(sysctl -n kern.memorystatus_vm_pressure_level)" "$(df -k /System/Volumes/Data | awk 'NR==2{print $4}')" "$(sysctl -n vm.swapusage)"
R=$("${SSH[@]}" 'echo "$(sysctl -n kern.memorystatus_vm_pressure_level)|$(df -k /System/Volumes/Data | awk "NR==2{print \$4}")|$(sysctl -n vm.swapusage)"')
check_host mini "$(cut -d'|' -f1 <<<"$R")" "$(cut -d'|' -f2 <<<"$R")" "$(cut -d'|' -f3 <<<"$R")"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "FAIL: port $PORT busy"; exit 1; fi

echo "== deploy remote sources (immutable, hash readback)"
REMOTE_ROOT=$("$PY" "$HERE/deploy.py" "$TARGET")
echo "remote root: $REMOTE_ROOT"

RUN="$RUNS/run-$(date +%Y%m%d-%H%M%S)"
nohup "$PY" -u "$HERE/chat_supervise.py" --output "$RUN" --remote-root "$REMOTE_ROOT" \
  --local-address "$LOCAL_IP" --remote-address "$REMOTE_IP" --ssh-target "$TARGET" \
  --ssh-host-key-alias "$ALIAS" --mode "$MODE" --http-port "$PORT" --session-hours "$HOURS" \
  >"$RUN.supervisor.log" 2>&1 &
SUP=$!
for _ in $(seq 50); do [ -f "$RUN/supervisor.json" ] && break; sleep 0.1; done
ln -sfn "$RUN" "$CUR"
echo "supervisor pid $SUP, run $RUN; loading model + strict admission (several minutes)…"
for i in $(seq 1 1200); do
  if ! kill -0 "$SUP" 2>/dev/null; then echo "FAIL: supervisor exited; see $RUN/result.json"; cat "$RUN.supervisor.log"; rm -f "$CUR"; exit 1; fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready": true'; then
    echo "READY after ${i}s: http://127.0.0.1:$PORT/   (stop with $HERE/stop-chat.sh)"
    curl -s "http://127.0.0.1:$PORT/health"; echo
    exit 0
  fi
  sleep 1
done
echo "FAIL: not healthy within 20 min; stopping"; "$HERE/stop-chat.sh"; exit 1
