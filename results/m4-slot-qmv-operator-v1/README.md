# Slot-shared QMV operator gate — partial acceleration work

All112 random quantized-array cases on the physical M4 returned exactly equal finite output values to concatenated stock width-one QMV calls and passed the unchanged atol=rtol=.002 gate. Seven geometries cover3/4bit,F16/F32,widths1/2/5/9,groups32/64/128,fast/slow/tail paths andK896/4864/5120. This is numerical/operator evidence, not checkpoint/speculation or speed admission. No timing comparison was performed. Actor elapsed6.967710709seconds, deadline150seconds, peak MLX allocation4,436,024bytes.

Exactness means NumPy element equality, not independently audited storage-byte identity. Peak allocation is not a residency guarantee. Source bindings, guarded full32B source equality and owned child/group cleanup were verified. No weights downloaded or default working decoder edited. Numerical tolerances were not loosened. Full greedy speculation/cache gates, separate stable attention, independent review and physical32B timing remain pending; no speedup is claimed.

Parent independently reran312 CPU pytest cases,2 stdlib tests and Ruff/format. Eighteen numbered RED/GREEN cycles have actual nonzero RED and zero GREEN exits; the10-file worker manifest matches stored source/support bytes. Source derives pinned MIT MLX0.32.3 QMV helpers with license/attribution. New scheduling stages packed bytes once per row/block for slot reuse; source structure does not prove compiler traffic, absence of spills or bandwidth gain.

`evidence/` contains exact executed sources, raw reports/rows and process/startup/preflight/cleanup/readback receipts. `source/` retains the candidate/tests/license/TDD/worker verification and manifest. `verified-summary.json` is computed from receipts; `integrity.json` hashes the archive excluding itself.
