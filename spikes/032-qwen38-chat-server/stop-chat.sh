#!/bin/bash
# Gracefully stop the chat server and verify no owned processes/listeners remain on either host.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
KEY=/Users/jeremylien/.ssh/paretoquant_mini_ed25519
ALIAS='fe80::56:c507:c02a:a630%bridge0'
TARGET="${MINI_SSH_TARGET:-jeremylien@fe80::1456:cf50:88f2:90d9%bridge0}"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o "HostKeyAlias=$ALIAS" -o IdentitiesOnly=yes -o ConnectTimeout=5 -i "$KEY" "$TARGET")
CUR="$HERE/runs/current"
[ -e "$CUR" ] || { echo "No current run (runs/current missing)."; }
if [ -e "$CUR" ]; then
  RUN=$(readlink "$CUR")
  read -r PID NONCE PORT < <("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]+"/supervisor.json"));print(d["pid"],d["nonce"],d["args"]["http_port"])' "$RUN")
  touch "$RUN/stop"   # supervisor polls this; it POSTs /shutdown then nonce-stops actors
  for _ in $(seq 1 150); do [ -f "$RUN/result.json" ] && break; sleep 1; done
  if [ ! -f "$RUN/result.json" ] && ps -p "$PID" -o command= | grep -q "chat_supervise.py --output $RUN"; then
    echo "supervisor did not finish in 150 s; sending SIGTERM to owned supervisor $PID"; kill -TERM "$PID"
    for _ in $(seq 1 60); do [ -f "$RUN/result.json" ] && break; sleep 1; done
  fi
  for _ in $(seq 1 20); do ps -p "$PID" >/dev/null 2>&1 || break; sleep 0.5; done
  echo "== supervisor result"
  "$PY" - "$RUN/result.json" <<'EOF'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps({k:r.get(k) for k in ("status","stop_reason","error_type","error_stage","shutdown_posted","ranks_exited_gracefully","owned_actor_cleanup_verified","services_closed_verified","listeners")}, indent=1))
for n,a in (r.get("actor_results") or {}).items():
    print(n, a and a.get("reason"), a and a.get("cleanup",{}).get("group_gone"), a and a.get("pressure"))
EOF
  rm -f "$CUR"
fi
echo "== independent readback"
LOCAL_PROCS=$(pgrep -fl '032-qwen38-chat-server|sources/chat_rank[.]py' | grep -v -e pgrep -e stop-chat.sh || true)
REMOTE_PROCS=$("${SSH[@]}" "pgrep -fl '032-qwen38-chat-server|sources/chat_rank[.]py' | grep -v pgrep || true")
LISTEN=$(lsof -nP -iTCP:"${PORT:-8780}" -sTCP:LISTEN 2>/dev/null || true)
echo "M2 owned processes: ${LOCAL_PROCS:-none}"
echo "mini owned processes: ${REMOTE_PROCS:-none}"
echo "M2 listener on ${PORT:-8780}: ${LISTEN:-none}"
if [ -z "$LOCAL_PROCS$REMOTE_PROCS$LISTEN" ]; then echo "CLEAN: no owned processes or listeners on either host"; exit 0; fi
echo "NOT CLEAN"; exit 1
