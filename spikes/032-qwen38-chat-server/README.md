# 032: interactive two-Mac Qwen3.8-27B chat server + voice web UI

This directory turns the 031 adaptive-MTP runtime into a persistent serving mode. The model loads once and both ranks stay resident until you stop it. The UI page is `apps/chat-ui/index.html`, served by the same server.

## Start / use / stop

```bash
cd ~/Projects/paretoquant-metal/spikes/032-qwen38-chat-server
./start-chat.sh            # adaptive depth 1/2, tau 0.5 (031 profile)
./start-chat.sh --depth1   # fallback: depth-1 MTP, rank1 verifier slot-qmv (021-style)
#   options: --port N (default 8780), --hours H (session deadline, default 8, max 12)
open http://127.0.0.1:8780/
./stop-chat.sh             # graceful shutdown, then verifies cleanup on both hosts
```

Port 8780 is the default because 8765 was already taken by an unrelated local process (`job-agent`).

`start-chat.sh` does the following:

1. **Bridge check.** Discovers both bridge0 IPv4 addresses through the pinned SSH alias. It then requires `route -n get` to name bridge0 in both directions. If not, it tries one bounded `ping -b bridge0` refresh.
2. **Resource check.** Checks memory pressure on both hosts and aborts on critical (4). It prints swap, requires at least 10 GiB free disk per host, and checks that the HTTP port is free.
3. **Deploy.** `deploy.py` copies the actor sources to the mini, in a fresh directory `~/Projects/paretoquant-dual-host/spikes/032-qwen38-chat-server-<digest12>`. The files are made read-only and the SHA-256 is read back. An existing directory is reused only if its hashes match; otherwise the deploy fails. Receipts go to `deploy-receipts/`.
4. **Launch.** Starts `chat_supervise.py` with nohup. Run receipts go to `runs/run-*`; `runs/current` points at the live run. The script waits for `/health` to report `ready` and then prints the URL.

`stop-chat.sh` stops the server in this order:

1. Touches `runs/current/stop`.
2. The supervisor POSTs `/shutdown`. Rank0 sends a shutdown frame to rank1 and both exit with code 0.
3. Every actor is stopped by nonce.
4. Process-group exit and listener closure are verified: HTTP, ring ports and the weight-server port.
5. As an independent check, it runs `pgrep` on both hosts and `lsof` on the port.

## API (bound to 127.0.0.1 only; Host/Origin must be loopback)

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-style. Body: `{messages:[{role,content}], stream, max_tokens<=512, temperature 0..2, enable_thinking=false}`. The SSE stream sends `data:` chunks, then a final chunk with `usage` and `x_stats` (tok/s, first-token time, rounds, accepted drafts), then `[DONE]`. |
| `POST /v1/cancel` | Stops the current generation at the next round boundary. Closing the client connection also cancels. |
| `GET /health` | Status. |
| `POST /shutdown` | Shuts the server down. |
| `GET /` | The UI page. |

Request limits:
- One request at a time. A concurrent request gets 429.
- Prompt plus `max_tokens` must be at most 4096. Otherwise you get 400 `context_length_exceeded` with the prompt token count.
- Multi-turn works by the client sending the full history. The server applies the model's chat template, with thinking off by default.

## UI

The page has:
- a transcript with streaming text;
- tok/s, first-token time and token counts for each reply;
- Stop and New chat buttons;
- controls for thinking, temperature and max tokens.

Voice:
- **Push-to-talk dictation:** hold the 🎤 button or Ctrl+Space. It uses the browser's Web Speech API (`SpeechRecognition`) and types into the input box.
- **Read replies aloud:** a toggle that uses `speechSynthesis`.

The page loads no external assets and makes no calls other than to its own server. Note that the browser vendor may route `SpeechRecognition` audio through its own speech service (Chrome does).

## Runtime / safety

- **Runtime.** Unchanged 031 runtime:
  - rank0 = M2, layers 36..63 plus head, native verifier;
  - rank1 = mini, layers 0..35, `slot-qmv-tuned-m4`;
  - group32, prefix rollback.
  - At every startup, the existing strict small+full width-2/width-3 and prefix gates run on both GPUs before serving. Pinned target is `10c35caa…` and sidecar is `b643c01b…`.
- **Generation loop.** `chat_rank.stream_generate` copies 031 `generate` / `generate_adaptive`, with the same backend op sequence. It adds a per-round streaming callback and cancellation.
- **Actor changes.** `remote_actor.py` (copy) admits the `chat_rank.py` entrypoint with a session deadline of at most 12 h. It also adds a sampled (~1 Hz) memory-pressure watchdog on each host:
  - pressure levels 1 and 2 are allowed (existing opt-in);
  - critical (4), a malformed sample, or a gap over 5 s aborts the owned child;
  - the supervisor then tears everything down.
- **Log volume.** High-volume per-layer and per-action events are dropped after admission, which keeps the 64 MiB log cap safe for long sessions. The count of dropped events is reported at shutdown.

## Measured (2026-11-06, run `runs/run-20261106-004047`) — one smoke test only

| Item | Result |
|---|---|
| Startup to healthy (preflight → deploy → load → admission) | 35 s after supervisor launch |
| Numerical admission | small + full passed on both ranks (rank0 native, rank1 slot-qmv-tuned-m4, width3 and prefix gates included) |
| Request | 1 streamed greedy request: "Write a Python function that checks whether a string is a palindrome." |
| Output | 85 tokens, `stop`, correct code; 32 SSE content chunks |
| Decode rate | **12.97 tok/s** post-first (EOS included) |
| Overall rate | 5.02 tok/s whole-request |
| First token | 10.45 s; this was the first request after load, so it includes cold first-use of shapes |
| Rounds | 30 rounds, 29 of them width-3; 55 accepted drafts |
| Shutdown | graceful, both ranks reported `clean_shutdown` |
| Cleanup | owned actor cleanup and listener closure verified; independent `pgrep`/`lsof` found nothing on either host |
| Memory pressure | warning (2) samples occurred on both hosts, which is allowed; zero critical samples |

## NOT tested (per user directive to cut testing to one smoke test)

- Token-for-token equality versus the batch 031 runtime for the same prompt. Exactness relies only on the startup gates plus the copied, identical op sequence.
- Multi-turn follow-ups, more than one turn, and the context-limit error path.
- Concurrent-request rejection (429), cancel/Stop, and the `--depth1` fallback.
- Temperature sampling. It is implemented as sample-and-accept-iff-equal-to-draft, but is never exercised. The MTP exactness claim covers greedy only.
- The UI in a browser: page load, mic dictation, read-aloud. The HTML was not even fetched.
- Memory-watchdog abort and the session deadline path.
- Warm first-token latency.

Other limitations:
- There is no prompt-prefix KV reuse between turns: every turn re-prefills the full history.
- Bounded at 4096 tokens total and 512 new tokens per turn.
