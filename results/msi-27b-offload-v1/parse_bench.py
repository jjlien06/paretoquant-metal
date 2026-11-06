import json, re, sys
raw = open(sys.argv[1]).read()
for m in re.finditer(r"### BEGIN (\S+).*?\n(.*?)### EXIT \S+ (\d+)", raw, re.S):
    name, body, code = m.groups()
    try:
        rows = json.loads(body[body.index("["):body.rindex("]") + 1])
    except ValueError:
        print(f"{name:24s} exit={code} no-json"); continue
    out = []
    for r in rows:
        kind = "pp" if r["n_gen"] == 0 else "tg"
        out.append(f"{kind}{r['n_prompt'] or r['n_gen']} {r['avg_ts']:.2f}±{r['stddev_ts']:.2f}")
    print(f"{name:24s} exit={code} " + " | ".join(out))
