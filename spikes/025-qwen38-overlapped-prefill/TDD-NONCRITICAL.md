# Explicit noncritical pressure policy — CPU implementation

This is an opted-in policy amendment, not a replacement of the original normal
criterion, a numerical admission, a memory-fit result, or permission to run GPUs.
The authoritative user amendment is
`results/qwen38-prefill-overlap-v1/memory-acceptance-amendment-v1.json`
(SHA256 `a0341821118cfd83d45f0f8979152aaffae3f296b372f8dc29b58aacc35d409f`).

## Vertical RED → GREEN evidence

New logs live under
`results/qwen38-prefill-overlap-v1/noncritical-runtime-work/`.

1. `01-core-{red,green}.txt`: helper/guard/receipt/external verification tracer.
   RED lacked `policy_name`; GREEN records actual levels1/2 in required v3 samples.
2. `02-bindings-{red,green}.txt`: independent startup/result/three-owner policy
   binding, exact types, schema/expected-policy separation, gap/replay/invalid
   samples. RED exposed the absent expected-policy API and bool-as-level forgery.
3. `03-actor-{red,green}.txt`: real `run_owned` with mocked Popen, sampler,
   cleanup and signals, plus actual actor-main descriptor propagation. RED
   rejected the new keyword/job field; GREEN retains the false legacy flag.
4. `04-cli-{red,green}.txt`: both CLI choices/defaults, both-rank options,
   effective overlap guard, conflict rejection, mocked probe process-start and
   verification producer. RED lacked the CLI/policy records.
5. `05-controller-{red,green}.txt`: mocked full controller path, all three job
   descriptors/startup/PID/result bindings including remote weight server, and
   schedule verifier external-policy propagation. RED skipped noncritical checks
   and lacked descriptor policy. One old test double was adapted to explicitly
   assert its unchanged default `expected_policy="normal"`.
6. `06-live-evidence-{red,green}.txt`: numerical-verifier envelope policy on both
   ranks across rank result/verification/process-start/live admission. RED
   accepted all sixteen missing/conflicting policy/legacy-flag mutations.
   Synthetic dictionaries exercise validation only, not model arithmetic.
7. `07-registry-{red,green}.txt`: exact source registry includes the new policy
   test and the sibling agent's `test_stream_lifetime.py` (not edited here).
8. `08-result-metadata-{red,green}.txt`: explicit controller startup/final and
   probe result policy. RED lacked those fields; GREEN discloses truthful flags.
9. `09-overlap-policy-{red,green}.txt`: isolated old hardcoded normal-true check
   rejects a noncritical live overlap envelope; policy-aware check accepts it.
   Numerical producer seams are explicitly mocked; no physical gate claim.

Additional regressions retain noncritical pre-Popen stop/deadline checks, gated
bootstrap expiry, warning2→critical4 owned-unreaped cleanup, strict-normal level2
rejection, original v2 receipt fields, unchanged atol1e-6/rtol1e-5, and externally
selected policy rather than receipt-selected policy.

## Execution guard and limitations

The task-local `run_guarded_cpu.py` wraps the existing CPU runner. It forbids all
real child creation (therefore sysctl/SSH), network and process signals in addition
to the existing MLX/model import guard. Tests mock process/model seams explicitly.
The final broad run excludes the existing MLX-dependent modules and skips the
existing GPU/owned-process fixtures. Four existing child-spawning CLI tests are
explicitly deselected, not represented as passing. The first broad attempt tried
one additional CLI help subprocess; the audit guard blocked it BEFORE launch,
and that failed attempt is retained separately. The corrected final run has zero
blocked attempts. Counts and exact commands are in the implementation JSON/XML.

No GPU/MLX numerical execution, sysctl process, SSH, network, actual runtime child,
process signalling, model load, deployment, fourth full-model trial, or physical
64MiB diagnostic was performed. Warning tolerance is not memory optimization or
crash prevention. Loader-buffer changes belong to the separate streaming agent.

All original frozen phase snapshots, historical receipts/acceptance criteria,
023/024 and broad results remain outside the write scope. New source hashes need
fresh independent review, immutable parent-owned freeze/deployment/readback and
physical small/full/native256 gates before any hardware pilot. No performance
acceptance or production-safety guarantee is implied.
