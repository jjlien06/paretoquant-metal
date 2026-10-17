"""Serve one admitted pipeline rank's public checkpoint tensors, then exit."""

import argparse
import hashlib
import hmac
import json
import os
import socket
from pathlib import Path

from shard_plan import select_pipeline_rank
from stream_weights import checkpoint_inventory, exact, send_json, send_records, tensor_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--split", type=int, nargs="+", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    token = os.environ["PARETOQUANT_WEIGHT_TOKEN"].encode("ascii")
    if len(token) != 64:
        raise ValueError("Require a 256-bit hex session token")
    root = Path(args.source).resolve(strict=True)
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    tensors, provenance = checkpoint_inventory(root)
    plan = select_pipeline_rank(
        tensors,
        model_type=config["model_type"],
        num_hidden_layers=config["num_hidden_layers"],
        split=args.split,
        rank=args.rank,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )
    selected = {name: tensors[name] for name in plan["selected_keys"]}
    before = {
        p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
        for p in {item["path"] for item in provenance.values()}
    }
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = None
    try:
        listener.settimeout(45)
        listener.bind((args.bind, 0))
        listener.listen(1)
        print(
            json.dumps(
                {
                    "ready": True,
                    "address": listener.getsockname(),
                    "selected_bytes": plan["selected_bytes"],
                }
            ),
            flush=True,
        )
        connection, peer = listener.accept()
        listener.close()
        connection.settimeout(45)
        if peer[0] != args.peer or not hmac.compare_digest(exact(connection, 64), token):
            raise ValueError("Unexpected peer or session token")
        send_json(
            connection,
            {
                "config_text": config_bytes.decode(),
                "tensors": selected,
                "rank": args.rank,
                "split": args.split,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
        )
        send_records(connection, tensor_records(selected, provenance))
        if exact(connection, 1) != b"K":
            raise ValueError("Client did not acknowledge verified loading")
        after = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in before}
        if after != before or (root / "config.json").read_bytes() != config_bytes:
            raise ValueError("Source changed during serving")
        print(
            json.dumps(
                {
                    "completed": True,
                    "peer": peer[0],
                    "rank": args.rank,
                    "selected_tensor_count": len(selected),
                    "selected_bytes": plan["selected_bytes"],
                }
            ),
            flush=True,
        )
    finally:
        if connection is not None:
            connection.close()
        listener.close()


if __name__ == "__main__":
    main()
