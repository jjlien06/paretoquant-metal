# 029 — chained MTP depth-2/3 acceptance diagnostic (27B, two hosts)

Copy of `023-qwen38-workload-bench` (schedule runner). Changes: `mtp_head.py`
(`BoundHead.forward_chain` returns logits + `m.norm(h)`, upstream mlx-vlm chaining convention)
and `mlx_backend.py` (`draft` logs chained d2/d3 under snapshot/restore of the head KV; d1 and
all committed state unchanged). `test_depth_chain.py` checks cache neutrality.
`analyze_depth2.py` scores acceptance offline. Runnable snapshots: `../029-…-a|b|c` (schedule only
differs). Results: `results/qwen38-mtp-depth2-acceptance-v1/README.md`.
