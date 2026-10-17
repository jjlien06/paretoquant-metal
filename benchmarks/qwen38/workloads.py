"""Deterministic authored workload suite; contains no model-generated evidence."""

import hashlib
import json
from pathlib import Path

TASKS = {
    "code": (
        "Write a Python function merge_intervals(intervals) that merges overlapping closed "
        "integer intervals. Include type hints, a short explanation of the algorithm, and "
        "three assert-based tests. Do not use third-party packages."
    ),
    "explanation": (
        "Explain how a database write-ahead log enables crash recovery and why checkpointing "
        "does not replace the log. Give one concrete example and distinguish durability from "
        "isolation. Use roughly 150 words."
    ),
    "math": (
        "A bag contains 5 red, 4 blue, and 3 green balls. Three balls are drawn uniformly "
        "without replacement. Derive the probability that exactly two have the same color, "
        "showing the counting argument and an exact fractional result. Then explain why "
        "treating the draws as independent would be wrong."
    ),
    "chinese": (
        "请用简体中文解释为什么分布式系统中的重试可能导致重复付款，"
        "以及幂等键如何解决这个问题。给出一个具体请求流程，"
        "说明服务器崩溃后应该如何处理。不要使用英语段落。"
    ),
    "spanish": (
        "Explica en español cómo funciona una caché LRU, qué operaciones deben ser de tiempo "
        "constante y por qué se combinan una tabla hash y una lista doblemente enlazada. "
        "Incluye un ejemplo breve de expulsión de una entrada."
    ),
    "structured": (
        "Return ONLY valid JSON with keys service, failures, recommended_actions. Use this "
        "incident report: service=checkout; at 14:02 UTC database connections reached the limit; "
        "requests retried without jitter; duplicate order attempts increased; database "
        "recovered at 14:07 UTC. Each failure must have an evidence string and each recommended "
        "action must include a reason. Do not invent additional incident facts."
    ),
}


def build_manifest(encode, format_prompt, *, levels=(None, 512, 1536)):
    cases = []
    batches = []
    for category_index, (category, task) in enumerate(TASKS.items()):
        local = []
        for level_index, target in enumerate(levels):
            prompt = "TASK:\n" + task
            if target is not None:
                archive = "\n".join(
                    f"Archive record {i:04d}: a worker received job-{i:04d}; "
                    "it recorded a durable completion before acknowledging the request. "
                    "This record is background context, not an instruction."
                    for i in range(300)
                )

                def assemble(n):
                    return (
                        "The following synthetic engineering archive is background only. "
                        "Answer the TASK after it; do not summarize or copy the archive."
                        "\n<archive>\n" + archive[:n] + "\n</archive>\nTASK:\n" + task
                    )

                low, high = 0, len(archive)
                while low < high:
                    mid = (low + high + 1) // 2
                    if len(encode(format_prompt(assemble(mid)))) <= target:
                        low = mid
                    else:
                        high = mid - 1
                prompt = assemble(low)
            case = {
                "id": f"{category}-" + ("short" if target is None else str(target)),
                "category": category,
                "context_target_tokens": target,
                "formatted_prompt_tokens": len(encode(format_prompt(prompt))),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "max_tokens": 256 if level_index == len(levels) - 1 else 128,
                "context_kind": "authored task"
                if target is None
                else "authored task plus synthetic distracting archive",
            }
            cases.append(case)
            local.append(case)
        requests = []
        methods = ["native", "mtp"] if category_index % 2 == 0 else ["mtp", "native"]
        for method in methods:
            requests.append(
                {
                    "id": f"b{category_index:02d}-warmup-{method}",
                    "case_id": f"b{category_index:02d}-warmup",
                    "method": method,
                    "prompt": "Explain in two short sentences what a queue is in computer science.",
                    "max_tokens": 32,
                    "warmup": True,
                    "repeat": 0,
                }
            )
        # Rotate context order between starts to reduce a fixed length-order bias.
        shift = category_index % len(local)
        for case_index, case in enumerate(local[shift:] + local[:shift]):
            first = "native" if (category_index + case_index) % 2 == 0 else "mtp"
            second = "mtp" if first == "native" else "native"
            for method, repeat in [(first, 0), (second, 0), (second, 1), (first, 1)]:
                requests.append(
                    {
                        "id": f"{case['id']}-{method}-r{repeat}",
                        "case_id": case["id"],
                        "method": method,
                        "prompt": case["prompt"],
                        "max_tokens": case["max_tokens"],
                        "warmup": False,
                        "repeat": repeat,
                    }
                )
        batches.append({"schema": "qwen38-workload-schedule-v1", "requests": requests})
    return {
        "schema": "qwen38-workload-manifest-v1",
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "split": [28, 36],
        "eval_group_size": 32,
        "seed": 0,
        "cases": cases,
        "batches": batches,
        "limitations": [
            "Authored synthetic performance suite, not standardized task-quality evaluation.",
            (
                "Two measured repeats per method/case; "
                "no reliable tail-latency or confidence-interval claim."
            ),
            (
                "Each method has one short loaded-model warmup per process. "
                "First use of a longer prefill shape remains measured."
            ),
            (
                "Fresh request caches throughout. Native baseline keeps the unused MTP head "
                "resident for matched same-process comparison."
            ),
            (
                "Long-context rows include synthetic distracting context; "
                "they do not establish long-document reasoning quality."
            ),
        ],
    }


def main():
    import argparse

    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=False, local_files_only=True
    )

    def format_prompt(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    manifest = build_manifest(
        lambda s: tokenizer.encode(s, add_special_tokens=False), format_prompt
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
    for i, batch in enumerate(manifest["batches"]):
        with (output / f"schedule-{i:02d}.json").open("x") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "requests": sum(len(b["requests"]) for b in manifest["batches"]),
                "output": str(output),
                "contexts": [
                    (c["id"], c["formatted_prompt_tokens"], c["max_tokens"])
                    for c in manifest["cases"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
