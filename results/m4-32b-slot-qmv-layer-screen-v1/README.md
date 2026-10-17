# Actual32B gate/down weights: isolated slot-sharing screen

Eight physical M4 F16 operator cases, widths1/2/5/9, using genuine layer0 gate/down affine4bit/group64 parameters passed exact finite output values against concatenated stock width-one QMV. Each of six loaded tensor byte digests equals the previously admitted full32B rank1 record. Random input activations are not captured32B activations. Only two projections were loaded, not the whole model.

After two warmups, each case has eight interleaved AB/BA trials, three newly evaluated graph-build+GPU calls per sample, explicit GPU synchronization around timers. These are isolated-call medians, not native batched-GEMM comparisons, whole-model costs, end-to-end timings, statistical significance, a selector admission, or inference speedup. Width-one custom dispatch is screened but the proposed teacher overlay retains stock width-one execution.

projection | width | serial width-one ms | shared slots ms | isolated ratio
gate_proj | 1 | 1.1553 | 1.1814 | 0.978x
gate_proj | 2 | 2.0372 | 1.1214 | 1.817x
gate_proj | 5 | 4.7908 | 3.6187 | 1.324x
gate_proj | 9 | 7.6263 | 13.8551 | 0.550x
down_proj | 1 | 1.1422 | 1.2053 | 0.948x
down_proj | 2 | 1.9494 | 1.1971 | 1.628x
down_proj | 5 | 4.5636 | 3.7616 | 1.213x
down_proj | 9 | 7.6890 | 14.2688 | 0.539x

Stock/shared output arrays are element-equal; storage-byte identity was not audited independently. Original numerical tolerance.002 is unchanged. Only metadata and six tensor payloads were read from existing checkpoint files, no weight downloads or writes. Exact deployed source readbacks, unchanged guarded full32B sources and owned actor process/group cleanup passed. Peak MLX allocation is not system residency. Whole-pipeline timings and distributed numerical gates remain pending. Raw samples, hashes, startup/resource/actor/cleanup receipts and exact executed scripts are in evidence/. Integrity hashes all archive files excluding itself.
