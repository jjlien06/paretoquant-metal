# Observed broad-suite results

18 measured cases; two measured repeats per method/case. Rates count emitted EOS and exclude prefill/first-token latency. Whole-request speedup includes prefill/first-token latency but excludes loading and admission.

| Case | Native decode tok/s | MTP decode tok/s | Native TTFT s | MTP TTFT s | Whole-request ratio | Draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| chinese-1536 | 6.73 | 9.60 | 15.81 | 29.35 | 0.964x | 75.3% |
| chinese-512 | 7.11 | 10.18 | 6.23 | 10.24 | 1.063x | 84.1% |
| chinese-short | 7.51 | 9.82 | 1.01 | 1.23 | 1.266x | 74.0% |
| code-1536 | 7.45 | 11.04 | 15.08 | 25.26 | 1.020x | 93.2% |
| code-512 | 7.59 | 11.21 | 6.01 | 9.44 | 1.097x | 93.9% |
| code-short | 7.58 | 11.24 | 1.07 | 1.20 | 1.426x | 92.4% |
| explanation-1536 | 7.35 | 9.69 | 14.96 | 27.19 | 0.894x | 75.4% |
| explanation-512 | 7.39 | 10.21 | 6.16 | 8.82 | 1.097x | 81.4% |
| explanation-short | 7.52 | 10.28 | 1.02 | 1.20 | 1.322x | 78.9% |
| math-1536 | 7.51 | 10.92 | 15.00 | 26.18 | 0.989x | 90.3% |
| math-512 | 7.52 | 10.76 | 6.02 | 8.88 | 1.108x | 88.2% |
| math-short | 7.52 | 11.02 | 1.48 | 1.69 | 1.390x | 91.0% |
| spanish-1536 | 6.89 | 9.82 | 15.53 | 28.27 | 0.969x | 79.0% |
| spanish-512 | 7.25 | 9.95 | 7.08 | 9.03 | 1.128x | 76.4% |
| spanish-short | 7.28 | 9.55 | 1.06 | 1.27 | 1.270x | 75.3% |
| structured-1536 | 7.03 | 11.09 | 15.51 | 29.49 | 0.984x | 98.4% |
| structured-512 | 7.31 | 11.11 | 6.20 | 9.98 | 1.101x | 100.0% |
| structured-short | 7.44 | 11.30 | 1.56 | 1.77 | 1.432x | 100.0% |

Ratios above1favor MTP; below1favor native. These are medians of paired ratios, not ratios of pooled throughput.

Original numerical gates, native output equality and owned cleanup passed in the saved receipts. Independent review found an action-transcript validation gap in the original verifier; the separate stronger post-hoc audit subsequently passed all84 requests (168 rank-request records). See strict-action-audit.json and broad-verification.json for the composed verification and assurance limits. Original raw receipts are unchanged.
