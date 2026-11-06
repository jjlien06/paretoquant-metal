"""Resolve exact repos/revisions and read real safetensors headers via HTTP range (no weight download)."""

import json
import struct
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/qwen38-mixed-3-4-plan-v1"
REPOS = {
    "q4": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    "q3": ("lukaskremla/Qwen3.8-27B-3bit-MLX", None),
    "bf16": ("mlx-community/Qwen3.8-27B-bf16", None),
}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def get_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"short read {len(data)} for {url} {start}-{end}")
    return data


def header(url):
    n = struct.unpack("<Q", get_range(url, 0, 7))[0]
    h = json.loads(get_range(url, 8, 8 + n - 1))
    h.pop("__metadata__", None)
    return 8 + n, h


def main():
    inv = {}
    for key, (repo, rev) in REPOS.items():
        meta = get_json(f"https://huggingface.co/api/models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true")
        sha = meta["sha"]
        if rev and sha != rev:
            sys.exit(f"revision mismatch {repo}: {sha} != {rev}")
        files = {s["rfilename"]: s for s in meta["siblings"]}
        cfg = get_json(f"https://huggingface.co/{repo}/resolve/{sha}/config.json")
        tensors = {}
        for name, s in sorted(files.items()):
            if not name.endswith(".safetensors"):
                continue
            url = f"https://huggingface.co/{repo}/resolve/{sha}/{name}"
            off, h = header(url)
            for t, d in h.items():
                tensors[t] = {"file": name, "dtype": d["dtype"], "shape": d["shape"],
                              "start": off + d["data_offsets"][0], "end": off + d["data_offsets"][1]}
        inv[key] = {
            "repo": repo, "sha": sha,
            "files": {n: {"size": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256")}
                      for n, s in files.items() if n.endswith(".safetensors")},
            "quantization": cfg.get("quantization"),
            "text_config_keys": {k: cfg.get("text_config", {}).get(k) for k in
                                 ("hidden_size", "intermediate_size", "num_hidden_layers", "layer_types",
                                  "num_attention_heads", "num_key_value_heads", "head_dim")},
            "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
            "tensors": tensors,
        }
        print(key, repo, sha, len(tensors), "tensors",
              round(sum(f["size"] or 0 for f in inv[key]["files"].values()) / 1e9, 3), "GB")
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
