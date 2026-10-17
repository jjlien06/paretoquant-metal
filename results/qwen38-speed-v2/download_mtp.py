"""Fetch only the pinned Qwen3.8 MTP sidecar; preserve the main checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPO = "mlx-community/Qwen3.8-27B-MTP-4bit"
REV = "b643c01b6d3b094e325edb6ebd832e16c486c575"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "models" / f"qwen3.8-27b-mtp-4bit-{REV}"
ALLOW = {"config.json", "model.safetensors.index.json", "model.safetensors", "README.md"}

def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()

def main():
    metadata = json.loads(get(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"))
    assert metadata["sha"] == REV
    entries = {item["rfilename"]: item for item in metadata["siblings"] if item["rfilename"] in ALLOW}
    assert set(entries) == ALLOW
    total = sum(item["size"] for item in entries.values())
    space = os.statvfs(ROOT)
    available = space.f_bavail * space.f_frsize
    if available - total < 8 * 1024**3:
        raise OSError("Refusing sidecar download below 8 GiB disk reserve")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, entry in entries.items():
        destination = DEST / name
        expected = entry.get("lfs", {}).get("sha256")
        if destination.exists():
            content = destination.read_bytes()
        else:
            content = get(f"https://huggingface.co/{REPO}/resolve/{REV}/{name}")
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != entry["size"] or (expected is not None and digest != expected):
            raise ValueError(f"Immutable size/hash verification failed: {name}")
        if not destination.exists():
            with destination.open("xb") as output:
                output.write(content)
        with destination.open("rb") as saved:
            readback = hashlib.file_digest(saved, "sha256").hexdigest()
        assert readback == digest
        rows.append({"name": name, "bytes": len(content), "sha256": digest,
                     "upstream_lfs_sha256": expected})
    receipt = {"repo": REPO, "revision": REV, "path": str(DEST), "download_bytes": total,
               "pre_download_available_bytes": available, "files": rows}
    receipt_path = Path(__file__).with_name("mtp-download.json")
    with receipt_path.open("x") as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
