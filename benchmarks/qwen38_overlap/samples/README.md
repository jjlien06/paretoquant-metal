# TEST fixtures — not benchmark results

Every file below `TEST_ONLY_NOT_MODEL_RESULTS/` is artificial CPU test data.
No model, GPU, original teacher prediction, host-pressure measurement, live
capability, or physical runtime receipt was produced to generate it.

- `TEST-record.json` / `TEST-trusted.json`: one artificial14-request normalized
  batch using a tiny TEST source/tensor/cache plan and one-token EOS output.
  Its prompt lengths exercise the frozen schedule but its outputs are **not**
  the original native model outputs. Do not use this trusted input for hardware.
- `TEST-timing-rows.json`: deliberately chosen synthetic84-row timings to
  discriminate median-of-paired-ratios from ratio-of-medians. They are not
  measured request durations.
- `TEST-collect-report/` and `TEST-audit-report/`: actual CLI readback of that TEST
  contract. Both exit2, preserve18 case rows and70 missing requests, and report
  `accepted: false` / integration pending.
- `TEST-analyze-report/`: actual descriptive reduction of synthetic timing rows.
  Its numeric target flag is a TEST arithmetic result, not GPU provenance;
  `evidence_validation: not_evaluated`, `accepted: false` remain mandatory.
- `PROVENANCE.json`: explicit non-execution label.

Each report directory includes JSON, formula-safe CSV, Markdown and actual file
hashes. Frozen full84/pilot6 schedules in `../frozen-v1/` are real **plans**, not
executed requests. Only the exclusive parent hardware owner can collect real
outputs after documented runtime integration, independent review and live gates.
