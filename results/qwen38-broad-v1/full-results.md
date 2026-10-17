# Qwen3.8 workload evidence analysis

**Coverage: complete** — 84/84 requests; 72 measured, 12 explicit warmups; 24 requested cases total (18 measured; 6 warmup; case sets may overlap).

Complete scheduled observed requests; not a quality or runtime verification claim.

Statistics are descriptive: sample stdev is undefined for n<2. Lower-tail minima are not a confidence interval. Two repeats cannot establish reliable tail latency. Cases are never pooled as tokens/sec. Token events include any emitted EOS. Single-token outputs are excluded only from post-first decode.

No quality scoring or runtime verification is performed. Rank agreement, checkpoint identity, placement, eval32, gate receipts, cleanup, and resident-sidecar memory baselines must be established upstream.

EOS token identity checked using supplied metadata: True.

## Per-case measured statistics

| Case | Method | Metric | n | Median | Min | Max | Sample stdev |
|---|---|---|---:|---:|---:|---:|---:|
| b00-warmup | native | post_first_tps | 0 | — | — | — | — |
| b00-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b00-warmup | native | wall_seconds | 0 | — | — | — | — |
| b00-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b00-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b00-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b00-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b00-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b01-warmup | native | post_first_tps | 0 | — | — | — | — |
| b01-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b01-warmup | native | wall_seconds | 0 | — | — | — | — |
| b01-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b01-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b01-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b01-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b01-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b02-warmup | native | post_first_tps | 0 | — | — | — | — |
| b02-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b02-warmup | native | wall_seconds | 0 | — | — | — | — |
| b02-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b02-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b02-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b02-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b02-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b03-warmup | native | post_first_tps | 0 | — | — | — | — |
| b03-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b03-warmup | native | wall_seconds | 0 | — | — | — | — |
| b03-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b03-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b03-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b03-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b03-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b04-warmup | native | post_first_tps | 0 | — | — | — | — |
| b04-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b04-warmup | native | wall_seconds | 0 | — | — | — | — |
| b04-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b04-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b04-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b04-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b04-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| b05-warmup | native | post_first_tps | 0 | — | — | — | — |
| b05-warmup | native | first_token_seconds | 0 | — | — | — | — |
| b05-warmup | native | wall_seconds | 0 | — | — | — | — |
| b05-warmup | native | whole_request_tps | 0 | — | — | — | — |
| b05-warmup | mtp | post_first_tps | 0 | — | — | — | — |
| b05-warmup | mtp | first_token_seconds | 0 | — | — | — | — |
| b05-warmup | mtp | wall_seconds | 0 | — | — | — | — |
| b05-warmup | mtp | whole_request_tps | 0 | — | — | — | — |
| chinese-1536 | native | post_first_tps | 2 | 6.72743 | 6.51589 | 6.93896 | 0.299157 |
| chinese-1536 | native | first_token_seconds | 2 | 15.8078 | 15.4405 | 16.1752 | 0.519547 |
| chinese-1536 | native | wall_seconds | 2 | 53.7499 | 52.1895 | 55.3103 | 2.20676 |
| chinese-1536 | native | whole_request_tps | 2 | 4.76682 | 4.62843 | 4.9052 | 0.195707 |
| chinese-1536 | mtp | post_first_tps | 2 | 9.59906 | 9.55416 | 9.64396 | 0.0634984 |
| chinese-1536 | mtp | first_token_seconds | 2 | 29.3539 | 27.2431 | 31.4646 | 2.98506 |
| chinese-1536 | mtp | wall_seconds | 2 | 55.9196 | 53.6845 | 58.1546 | 3.16079 |
| chinese-1536 | mtp | whole_request_tps | 2 | 4.58533 | 4.40206 | 4.7686 | 0.259181 |
| chinese-512 | native | post_first_tps | 2 | 7.10537 | 7.01738 | 7.19336 | 0.124438 |
| chinese-512 | native | first_token_seconds | 2 | 6.23009 | 6.19848 | 6.2617 | 0.0447009 |
| chinese-512 | native | wall_seconds | 2 | 24.1066 | 23.8537 | 24.3596 | 0.357777 |
| chinese-512 | native | whole_request_tps | 2 | 5.31032 | 5.25459 | 5.36605 | 0.0788128 |
| chinese-512 | mtp | post_first_tps | 2 | 10.1844 | 10.1444 | 10.2244 | 0.0565508 |
| chinese-512 | mtp | first_token_seconds | 2 | 10.2351 | 9.51291 | 10.9574 | 1.02138 |
| chinese-512 | mtp | wall_seconds | 2 | 22.7054 | 21.9342 | 23.4766 | 1.09062 |
| chinese-512 | mtp | whole_request_tps | 2 | 5.64394 | 5.45225 | 5.83564 | 0.271099 |
| chinese-short | native | post_first_tps | 2 | 7.50989 | 7.4771 | 7.54267 | 0.0463685 |
| chinese-short | native | first_token_seconds | 2 | 1.01347 | 1.00717 | 1.01978 | 0.00891282 |
| chinese-short | native | wall_seconds | 2 | 17.9248 | 17.8573 | 17.9924 | 0.0955035 |
| chinese-short | native | whole_request_tps | 2 | 7.14103 | 7.11413 | 7.16793 | 0.0380474 |
| chinese-short | mtp | post_first_tps | 2 | 9.82464 | 9.57662 | 10.0727 | 0.350755 |
| chinese-short | mtp | first_token_seconds | 2 | 1.23246 | 1.20059 | 1.26432 | 0.0450578 |
| chinese-short | mtp | wall_seconds | 2 | 14.1674 | 13.809 | 14.5258 | 0.506855 |
| chinese-short | mtp | whole_request_tps | 2 | 9.04062 | 8.81192 | 9.26933 | 0.323439 |
| code-1536 | native | post_first_tps | 2 | 7.44567 | 7.38091 | 7.51044 | 0.0915941 |
| code-1536 | native | first_token_seconds | 2 | 15.081 | 14.832 | 15.33 | 0.35208 |
| code-1536 | native | wall_seconds | 2 | 49.3317 | 48.7848 | 49.8786 | 0.773419 |
| code-1536 | native | whole_request_tps | 2 | 5.19 | 5.13247 | 5.24754 | 0.0813686 |
| code-1536 | mtp | post_first_tps | 2 | 11.0361 | 10.9952 | 11.0771 | 0.0578958 |
| code-1536 | mtp | first_token_seconds | 2 | 25.2624 | 25.0446 | 25.4802 | 0.307976 |
| code-1536 | mtp | wall_seconds | 2 | 48.3686 | 48.2366 | 48.5007 | 0.186761 |
| code-1536 | mtp | whole_request_tps | 2 | 5.29273 | 5.27828 | 5.30718 | 0.0204362 |
| code-512 | native | post_first_tps | 2 | 7.59347 | 7.57417 | 7.61276 | 0.027289 |
| code-512 | native | first_token_seconds | 2 | 6.01348 | 6.01158 | 6.01537 | 0.00268044 |
| code-512 | native | wall_seconds | 2 | 22.7385 | 22.6941 | 22.7829 | 0.062786 |
| code-512 | native | whole_request_tps | 2 | 5.62924 | 5.61825 | 5.64023 | 0.0155436 |
| code-512 | mtp | post_first_tps | 2 | 11.2106 | 11.1951 | 11.2262 | 0.021996 |
| code-512 | mtp | first_token_seconds | 2 | 9.43746 | 8.49633 | 10.3786 | 1.33096 |
| code-512 | mtp | wall_seconds | 2 | 20.766 | 19.8092 | 21.7229 | 1.35319 |
| code-512 | mtp | whole_request_tps | 2 | 6.17703 | 5.89241 | 6.46165 | 0.402517 |
| code-short | native | post_first_tps | 2 | 7.57759 | 7.51598 | 7.6392 | 0.0871295 |
| code-short | native | first_token_seconds | 2 | 1.06607 | 1.0368 | 1.09533 | 0.0413868 |
| code-short | native | wall_seconds | 2 | 17.8271 | 17.6616 | 17.9927 | 0.234111 |
| code-short | native | whole_request_tps | 2 | 7.18069 | 7.11401 | 7.24737 | 0.0942988 |
| code-short | mtp | post_first_tps | 2 | 11.2351 | 11.2143 | 11.256 | 0.0294398 |
| code-short | mtp | first_token_seconds | 2 | 1.19885 | 1.19529 | 1.20242 | 0.00504409 |
| code-short | mtp | wall_seconds | 2 | 12.5027 | 12.4782 | 12.5272 | 0.0346639 |
| code-short | mtp | whole_request_tps | 2 | 10.2378 | 10.2178 | 10.2579 | 0.0283845 |
| explanation-1536 | native | post_first_tps | 2 | 7.35171 | 7.2007 | 7.50271 | 0.213555 |
| explanation-1536 | native | first_token_seconds | 2 | 14.9588 | 14.9055 | 15.012 | 0.0752512 |
| explanation-1536 | native | wall_seconds | 2 | 43.9438 | 43.2953 | 44.5924 | 0.917222 |
| explanation-1536 | native | whole_request_tps | 2 | 4.87091 | 4.79902 | 4.9428 | 0.101669 |
| explanation-1536 | mtp | post_first_tps | 2 | 9.69285 | 9.59134 | 9.79437 | 0.143561 |
| explanation-1536 | mtp | first_token_seconds | 2 | 27.1863 | 26.9867 | 27.3859 | 0.282266 |
| explanation-1536 | mtp | wall_seconds | 2 | 49.1636 | 48.7339 | 49.5934 | 0.607773 |
| explanation-1536 | mtp | whole_request_tps | 2 | 4.35314 | 4.31509 | 4.3912 | 0.0538147 |
| explanation-512 | native | post_first_tps | 2 | 7.39215 | 7.38524 | 7.39905 | 0.00976374 |
| explanation-512 | native | first_token_seconds | 2 | 6.15586 | 6.1537 | 6.15803 | 0.00306631 |
| explanation-512 | native | wall_seconds | 2 | 23.3363 | 23.3181 | 23.3545 | 0.0257586 |
| explanation-512 | native | whole_request_tps | 2 | 5.48503 | 5.48075 | 5.48931 | 0.00605439 |
| explanation-512 | mtp | post_first_tps | 2 | 10.207 | 10.1841 | 10.2299 | 0.0323394 |
| explanation-512 | mtp | first_token_seconds | 2 | 8.82124 | 8.75838 | 8.8841 | 0.0889028 |
| explanation-512 | mtp | wall_seconds | 2 | 21.2638 | 21.2288 | 21.2987 | 0.0494804 |
| explanation-512 | mtp | whole_request_tps | 2 | 6.01965 | 6.00975 | 6.02956 | 0.0140076 |
| explanation-short | native | post_first_tps | 2 | 7.51762 | 7.48114 | 7.5541 | 0.0515863 |
| explanation-short | native | first_token_seconds | 2 | 1.02445 | 1.0226 | 1.0263 | 0.00261859 |
| explanation-short | native | wall_seconds | 2 | 17.9185 | 17.8384 | 17.9986 | 0.113309 |
| explanation-short | native | whole_request_tps | 2 | 7.1436 | 7.11166 | 7.17554 | 0.0451732 |
| explanation-short | mtp | post_first_tps | 2 | 10.2847 | 10.2626 | 10.3069 | 0.0313508 |
| explanation-short | mtp | first_token_seconds | 2 | 1.20045 | 1.19968 | 1.20121 | 0.00108064 |
| explanation-short | mtp | wall_seconds | 2 | 13.5489 | 13.5215 | 13.5763 | 0.0387221 |
| explanation-short | mtp | whole_request_tps | 2 | 9.44731 | 9.42822 | 9.4664 | 0.027 |
| math-1536 | native | post_first_tps | 2 | 7.50703 | 7.49712 | 7.51695 | 0.014021 |
| math-1536 | native | first_token_seconds | 2 | 15.0016 | 14.8536 | 15.1497 | 0.209373 |
| math-1536 | native | wall_seconds | 2 | 48.9699 | 48.8667 | 49.073 | 0.14593 |
| math-1536 | native | whole_request_tps | 2 | 5.22773 | 5.21671 | 5.23874 | 0.0155786 |
| math-1536 | mtp | post_first_tps | 2 | 10.9214 | 10.9044 | 10.9384 | 0.0240577 |
| math-1536 | mtp | first_token_seconds | 2 | 26.1818 | 24.9043 | 27.4593 | 1.80663 |
| math-1536 | mtp | wall_seconds | 2 | 49.5304 | 48.2166 | 50.8443 | 1.85806 |
| math-1536 | mtp | whole_request_tps | 2 | 5.17218 | 5.03498 | 5.30938 | 0.194027 |
| math-512 | native | post_first_tps | 2 | 7.51592 | 7.51331 | 7.51854 | 0.00369541 |
| math-512 | native | first_token_seconds | 2 | 6.02038 | 6.00734 | 6.03343 | 0.0184503 |
| math-512 | native | wall_seconds | 2 | 22.9178 | 22.9107 | 22.925 | 0.0101422 |
| math-512 | native | whole_request_tps | 2 | 5.58517 | 5.58342 | 5.58692 | 0.0024717 |
| math-512 | mtp | post_first_tps | 2 | 10.7567 | 10.7427 | 10.7708 | 0.0198563 |
| math-512 | mtp | first_token_seconds | 2 | 8.88423 | 8.48691 | 9.28155 | 0.561892 |
| math-512 | mtp | wall_seconds | 2 | 20.6908 | 20.3089 | 21.0727 | 0.540098 |
| math-512 | mtp | whole_request_tps | 2 | 6.18843 | 6.07421 | 6.30266 | 0.161538 |
| math-short | native | post_first_tps | 2 | 7.5219 | 7.49348 | 7.55032 | 0.0401936 |
| math-short | native | first_token_seconds | 2 | 1.47556 | 1.46615 | 1.48496 | 0.0132981 |
| math-short | native | wall_seconds | 2 | 18.3598 | 18.3054 | 18.4142 | 0.0769237 |
| math-short | native | whole_request_tps | 2 | 6.97181 | 6.95115 | 6.99246 | 0.0292103 |
| math-short | mtp | post_first_tps | 2 | 11.02 | 10.9999 | 11.04 | 0.0283244 |
| math-short | mtp | first_token_seconds | 2 | 1.68523 | 1.68429 | 1.68617 | 0.00133251 |
| math-short | mtp | wall_seconds | 2 | 13.2098 | 13.1898 | 13.2298 | 0.0282889 |
| math-short | mtp | whole_request_tps | 2 | 9.68979 | 9.67511 | 9.70446 | 0.0207507 |
| spanish-1536 | native | post_first_tps | 2 | 6.88679 | 6.82968 | 6.9439 | 0.0807682 |
| spanish-1536 | native | first_token_seconds | 2 | 15.5296 | 15.5121 | 15.5471 | 0.0247407 |
| spanish-1536 | native | wall_seconds | 2 | 52.5595 | 52.2699 | 52.8491 | 0.409546 |
| spanish-1536 | native | whole_request_tps | 2 | 4.87082 | 4.84398 | 4.89765 | 0.0379536 |
| spanish-1536 | mtp | post_first_tps | 2 | 9.81741 | 9.7614 | 9.87343 | 0.0792198 |
| spanish-1536 | mtp | first_token_seconds | 2 | 28.274 | 28.1554 | 28.3926 | 0.167729 |
| spanish-1536 | mtp | wall_seconds | 2 | 54.2491 | 54.2195 | 54.2787 | 0.0418722 |
| spanish-1536 | mtp | whole_request_tps | 2 | 4.71898 | 4.7164 | 4.72155 | 0.00364235 |
| spanish-512 | native | post_first_tps | 2 | 7.25074 | 7.21311 | 7.28837 | 0.0532142 |
| spanish-512 | native | first_token_seconds | 2 | 7.07525 | 6.19665 | 7.95385 | 1.24253 |
| spanish-512 | native | wall_seconds | 2 | 24.5912 | 23.6217 | 25.5607 | 1.37108 |
| spanish-512 | native | whole_request_tps | 2 | 5.21322 | 5.00769 | 5.41875 | 0.290663 |
| spanish-512 | mtp | post_first_tps | 2 | 9.94826 | 9.93734 | 9.95917 | 0.0154362 |
| spanish-512 | mtp | first_token_seconds | 2 | 9.02582 | 8.93071 | 9.12094 | 0.134513 |
| spanish-512 | mtp | wall_seconds | 2 | 21.7919 | 21.6828 | 21.901 | 0.154321 |
| spanish-512 | mtp | whole_request_tps | 2 | 5.87389 | 5.84448 | 5.9033 | 0.0415965 |
| spanish-short | native | post_first_tps | 2 | 7.27683 | 7.26735 | 7.28632 | 0.0134182 |
| spanish-short | native | first_token_seconds | 2 | 1.05648 | 1.04017 | 1.0728 | 0.0230761 |
| spanish-short | native | wall_seconds | 2 | 18.5092 | 18.4701 | 18.5482 | 0.0552581 |
| spanish-short | native | whole_request_tps | 2 | 6.91553 | 6.90093 | 6.93013 | 0.0206459 |
| spanish-short | mtp | post_first_tps | 2 | 9.54528 | 9.48825 | 9.60232 | 0.080655 |
| spanish-short | mtp | first_token_seconds | 2 | 1.27465 | 1.27441 | 1.27489 | 0.000341945 |
| spanish-short | mtp | wall_seconds | 2 | 14.5801 | 14.5009 | 14.6594 | 0.112086 |
| spanish-short | mtp | whole_request_tps | 2 | 8.77933 | 8.73161 | 8.82706 | 0.0674917 |
| structured-1536 | native | post_first_tps | 2 | 7.03399 | 6.78999 | 7.27798 | 0.34506 |
| structured-1536 | native | first_token_seconds | 2 | 15.5056 | 15.4961 | 15.5151 | 0.0134253 |
| structured-1536 | native | wall_seconds | 2 | 51.3748 | 50.1211 | 52.6285 | 1.77303 |
| structured-1536 | native | whole_request_tps | 2 | 4.92753 | 4.80728 | 5.04778 | 0.170057 |
| structured-1536 | mtp | post_first_tps | 2 | 11.0876 | 11.074 | 11.1012 | 0.0192223 |
| structured-1536 | mtp | first_token_seconds | 2 | 29.4942 | 28.5649 | 30.4234 | 1.31419 |
| structured-1536 | mtp | wall_seconds | 2 | 52.2223 | 51.3209 | 53.1237 | 1.27479 |
| structured-1536 | mtp | whole_request_tps | 2 | 4.84612 | 4.76247 | 4.92977 | 0.118298 |
| structured-512 | native | post_first_tps | 2 | 7.314 | 7.25962 | 7.36838 | 0.0769047 |
| structured-512 | native | first_token_seconds | 2 | 6.2026 | 6.19502 | 6.21018 | 0.0107194 |
| structured-512 | native | wall_seconds | 2 | 23.5675 | 23.446 | 23.6891 | 0.171868 |
| structured-512 | native | whole_request_tps | 2 | 5.43135 | 5.40334 | 5.45935 | 0.0396085 |
| structured-512 | mtp | post_first_tps | 2 | 11.1119 | 11.1018 | 11.122 | 0.0142903 |
| structured-512 | mtp | first_token_seconds | 2 | 9.9784 | 9.85559 | 10.1012 | 0.173674 |
| structured-512 | mtp | wall_seconds | 2 | 21.4076 | 21.2744 | 21.5408 | 0.188372 |
| structured-512 | mtp | whole_request_tps | 2 | 5.97941 | 5.94221 | 6.01662 | 0.0526147 |
| structured-short | native | post_first_tps | 2 | 7.43663 | 7.42274 | 7.45052 | 0.0196426 |
| structured-short | native | first_token_seconds | 2 | 1.55668 | 1.55525 | 1.55812 | 0.0020304 |
| structured-short | native | wall_seconds | 2 | 18.6344 | 18.6039 | 18.6648 | 0.0430775 |
| structured-short | native | whole_request_tps | 2 | 6.86904 | 6.85782 | 6.88027 | 0.0158793 |
| structured-short | mtp | post_first_tps | 2 | 11.3011 | 11.2667 | 11.3355 | 0.0486776 |
| structured-short | mtp | first_token_seconds | 2 | 1.77474 | 1.77399 | 1.77548 | 0.00105097 |
| structured-short | mtp | wall_seconds | 2 | 13.0127 | 12.9792 | 13.0462 | 0.0473546 |
| structured-short | mtp | whole_request_tps | 2 | 9.83662 | 9.81131 | 9.86193 | 0.0357965 |

## Coverage, acceptance, and memory

| Case | Method | Observed / expected measured | Decode excluded | Decode below 10 (%) | Acceptance accepted / rounds | Rank 0 peak bytes | Rank 1 peak bytes |
|---|---|---:|---:|---:|---|---:|---:|
| b00-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b00-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b01-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b01-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b02-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b02-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b03-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b03-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b04-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b04-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| b05-warmup | native | 0 / 0 | 0 | — | not supplied | — | — |
| b05-warmup | mtp | 0 / 0 | 0 | — | not supplied | — | — |
| chinese-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289692190 | 9763209104 |
| chinese-1536 | mtp | 2 / 2 | 0 | 100 | 220 / 292 (rate 0.753425; 0 rows without counters) | 8424656622 | 9763299568 |
| chinese-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247896582 | 9702982036 |
| chinese-512 | mtp | 2 / 2 | 0 | 0 | 116 / 138 (rate 0.84058; 0 rows without counters) | 8378723846 | 9702965848 |
| chinese-short | native | 2 / 2 | 0 | 100 | not supplied | 7911526224 | 9364905402 |
| chinese-short | mtp | 2 / 2 | 0 | 50 | 108 / 146 (rate 0.739726; 0 rows without counters) | 7963286830 | 9519578060 |
| code-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289622558 | 9763237776 |
| code-1536 | mtp | 2 / 2 | 0 | 0 | 246 / 264 (rate 0.931818; 0 rows without counters) | 8424656571 | 9763258260 |
| code-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247913314 | 9702981528 |
| code-512 | mtp | 2 / 2 | 0 | 0 | 124 / 132 (rate 0.939394; 0 rows without counters) | 8378823166 | 9702982391 |
| code-short | native | 2 / 2 | 0 | 100 | not supplied | 7895893300 | 9380387754 |
| code-short | mtp | 2 / 2 | 0 | 0 | 122 / 132 (rate 0.924242; 0 rows without counters) | 7963401518 | 9519713228 |
| explanation-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289606174 | 9763184524 |
| explanation-1536 | mtp | 2 / 2 | 0 | 100 | 184 / 244 (rate 0.754098; 0 rows without counters) | 8424656568 | 9763282828 |
| explanation-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247913670 | 9703096216 |
| explanation-512 | mtp | 2 / 2 | 0 | 0 | 114 / 140 (rate 0.814286; 0 rows without counters) | 8378723846 | 9703030680 |
| explanation-short | native | 2 / 2 | 0 | 100 | not supplied | 7889270986 | 9373124002 |
| explanation-short | mtp | 2 / 2 | 0 | 0 | 112 / 142 (rate 0.788732; 0 rows without counters) | 7963286830 | 9519352780 |
| math-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289704474 | 9763299208 |
| math-1536 | mtp | 2 / 2 | 0 | 0 | 242 / 268 (rate 0.902985; 0 rows without counters) | 8424754875 | 9763250060 |
| math-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247896582 | 9703128980 |
| math-512 | mtp | 2 / 2 | 0 | 0 | 120 / 136 (rate 0.882353; 0 rows without counters) | 8378707462 | 9703128984 |
| math-short | native | 2 / 2 | 0 | 100 | not supplied | 7935691968 | 9401267150 |
| math-short | mtp | 2 / 2 | 0 | 0 | 122 / 134 (rate 0.910448; 0 rows without counters) | 7963340078 | 9519778764 |
| spanish-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8289675806 | 9763241872 |
| spanish-1536 | mtp | 2 / 2 | 0 | 100 | 226 / 286 (rate 0.79021; 0 rows without counters) | 8424672983 | 9763266777 |
| spanish-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247913662 | 9702933080 |
| spanish-512 | mtp | 2 / 2 | 0 | 100 | 110 / 144 (rate 0.763889; 0 rows without counters) | 8378707462 | 9702948752 |
| spanish-short | native | 2 / 2 | 0 | 100 | not supplied | 7910917596 | 9395306114 |
| spanish-short | mtp | 2 / 2 | 0 | 100 | 110 / 146 (rate 0.753425; 0 rows without counters) | 7963401518 | 9519762380 |
| structured-1536 | native | 2 / 2 | 0 | 100 | not supplied | 8302406126 | 9771069356 |
| structured-1536 | mtp | 2 / 2 | 0 | 0 | 250 / 254 (rate 0.984252; 0 rows without counters) | 8437386890 | 9776349272 |
| structured-512 | native | 2 / 2 | 0 | 100 | not supplied | 8247896578 | 9702981528 |
| structured-512 | mtp | 2 / 2 | 0 | 0 | 128 / 128 (rate 1; 0 rows without counters) | 8378723846 | 9702997904 |
| structured-short | native | 2 / 2 | 0 | 100 | not supplied | 7988336092 | 9450392406 |
| structured-short | mtp | 2 / 2 | 0 | 0 | 128 / 128 (rate 1; 0 rows without counters) | 7988336225 | 9519343404 |

## Paired ratios (MTP / native throughput)

| Case | Repeat | Decode speedup | Whole-request speedup |
|---|---:|---:|---:|
| b00-warmup | median | — | — |
| b01-warmup | median | — | — |
| b02-warmup | median | — | — |
| b03-warmup | median | — | — |
| b04-warmup | median | — | — |
| b05-warmup | median | — | — |
| chinese-1536 | 0 | 1.48007 | 1.03028 |
| chinese-1536 | 1 | 1.37689 | 0.897427 |
| chinese-1536 | median | 1.42848 | 0.963855 |
| chinese-512 | 0 | 1.44561 | 1.03762 |
| chinese-512 | 1 | 1.42137 | 1.08751 |
| chinese-512 | median | 1.43349 | 1.06256 |
| chinese-short | 0 | 1.33542 | 1.29317 |
| chinese-short | 1 | 1.28079 | 1.23865 |
| chinese-short | median | 1.30811 | 1.26591 |
| code-1536 | 0 | 1.48968 | 1.03404 |
| code-1536 | 1 | 1.47489 | 1.00586 |
| code-1536 | median | 1.48229 | 1.01995 |
| code-512 | 0 | 1.47056 | 1.04471 |
| code-512 | 1 | 1.48216 | 1.15012 |
| code-512 | median | 1.47636 | 1.09741 |
| code-short | 0 | 1.49207 | 1.43629 |
| code-short | 1 | 1.47345 | 1.4154 |
| code-short | median | 1.48276 | 1.42584 |
| explanation-1536 | 0 | 1.332 | 0.899161 |
| explanation-1536 | 1 | 1.30544 | 0.888402 |
| explanation-1536 | median | 1.31872 | 0.893782 |
| explanation-512 | 0 | 1.37641 | 1.09842 |
| explanation-512 | 1 | 1.38518 | 1.09652 |
| explanation-512 | median | 1.38079 | 1.09747 |
| explanation-short | 0 | 1.36441 | 1.31926 |
| explanation-short | 1 | 1.37179 | 1.32574 |
| explanation-short | median | 1.3681 | 1.3225 |
| math-1536 | 0 | 1.45448 | 0.961104 |
| math-1536 | 1 | 1.45517 | 1.01776 |
| math-1536 | median | 1.45483 | 0.989434 |
| math-512 | 0 | 1.42982 | 1.12811 |
| math-512 | 1 | 1.43256 | 1.0879 |
| math-512 | median | 1.43119 | 1.10801 |
| math-short | 0 | 1.46219 | 1.38785 |
| math-short | 1 | 1.46793 | 1.39187 |
| math-short | median | 1.46506 | 1.38986 |
| spanish-1536 | 0 | 1.40575 | 0.962992 |
| spanish-1536 | 1 | 1.44567 | 0.974726 |
| spanish-1536 | median | 1.42571 | 0.968859 |
| spanish-512 | 0 | 1.37768 | 1.1671 |
| spanish-512 | 1 | 1.36645 | 1.08942 |
| spanish-512 | median | 1.37206 | 1.12826 |
| spanish-short | 0 | 1.3056 | 1.26528 |
| spanish-short | 1 | 1.31785 | 1.27372 |
| spanish-short | median | 1.31173 | 1.2695 |
| structured-1536 | 0 | 1.63093 | 1.02548 |
| structured-1536 | 1 | 1.52531 | 0.943478 |
| structured-1536 | median | 1.57812 | 0.984479 |
| structured-512 | 0 | 1.50668 | 1.08845 |
| structured-512 | 1 | 1.53204 | 1.1135 |
| structured-512 | median | 1.51936 | 1.10097 |
| structured-short | 0 | 1.51786 | 1.43068 |
| structured-short | 1 | 1.52144 | 1.43336 |
| structured-short | median | 1.51965 | 1.43202 |

## Equal-case aggregate

```json
{
  "interpretation": "Equal case weight: geometric mean of each case's median paired ratio. Lower-tail minimum is descriptive, not a confidence interval; tiny repeat counts do not establish tail reliability. No cross-case pooled tokens/sec; no quality scoring or runtime/numerical gate verification.",
  "decode": {
    "case_count": 18,
    "excluded_case_count": 6,
    "geometric_mean": 1.4290156456292908,
    "lower_tail_min": 1.3081083585863307,
    "case_speedup_statistics": {
      "count": 18,
      "median": 1.4323418263949617,
      "min": 1.3081083585863307,
      "max": 1.57812139122329,
      "stdev": 0.0760537713517794
    }
  },
  "whole_request": {
    "case_count": 18,
    "excluded_case_count": 6,
    "geometric_mean": 1.1284876578368026,
    "lower_tail_min": 0.8937815777234467,
    "case_speedup_statistics": {
      "count": 18,
      "median": 1.099220907403411,
      "min": 0.8937815777234467,
      "max": 1.4320197409983786,
      "stdev": 0.16974938559591932
    }
  }
}
```

## Missing request IDs

```json
[]
```

## Failures (0)

```json
[]
```

## Explicit first-loaded warmups (excluded from measured statistics)

```json
[
  {
    "id": "b00-warmup-native",
    "case_id": "b00-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.568999500013888,
    "first_token_seconds": 0.3735162499942817,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865509692,
      9320782699
    ],
    "output_tokens": 32,
    "post_first_tps": 7.388898525540563,
    "whole_request_tps": 7.003721493053946
  },
  {
    "id": "b00-warmup-mtp",
    "case_id": "b00-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 3.776770124968607,
    "first_token_seconds": 0.7198357079760171,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963462958,
      9519553484
    ],
    "output_tokens": 32,
    "post_first_tps": 10.140878334739606,
    "whole_request_tps": 8.472848211874158
  },
  {
    "id": "b01-warmup-mtp",
    "case_id": "b01-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 3.7051361660123803,
    "first_token_seconds": 0.7272755409940146,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963536686,
      9519586252
    ],
    "output_tokens": 32,
    "post_first_tps": 10.410158131497107,
    "whole_request_tps": 8.636659643858572
  },
  {
    "id": "b01-warmup-native",
    "case_id": "b01-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.761781125038397,
    "first_token_seconds": 0.5721910410211422,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865182012,
      9320192875
    ],
    "output_tokens": 32,
    "post_first_tps": 7.399291906447125,
    "whole_request_tps": 6.720174480875992
  },
  {
    "id": "b02-warmup-native",
    "case_id": "b02-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.547848957998212,
    "first_token_seconds": 0.4187785829999484,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865526076,
      9320782699
    ],
    "output_tokens": 32,
    "post_first_tps": 7.507743192682453,
    "whole_request_tps": 7.036293486335387
  },
  {
    "id": "b02-warmup-mtp",
    "case_id": "b02-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 3.6689805000205524,
    "first_token_seconds": 0.690617749991361,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963462958,
      9519553484
    ],
    "output_tokens": 32,
    "post_first_tps": 10.408403073029357,
    "whole_request_tps": 8.721768894607303
  },
  {
    "id": "b03-warmup-mtp",
    "case_id": "b03-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 3.6632583750179037,
    "first_token_seconds": 0.7022987080272287,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963528494,
      9519586252
    ],
    "output_tokens": 32,
    "post_first_tps": 10.469578611824309,
    "whole_request_tps": 8.735392572423615
  },
  {
    "id": "b03-warmup-native",
    "case_id": "b03-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.667984833009541,
    "first_token_seconds": 0.5490744159906171,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865067324,
      9320192875
    ],
    "output_tokens": 32,
    "post_first_tps": 7.526262254190117,
    "whole_request_tps": 6.855206506609186
  },
  {
    "id": "b04-warmup-native",
    "case_id": "b04-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.660857584036421,
    "first_token_seconds": 0.4371494590304792,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865526076,
      9320635243
    ],
    "output_tokens": 32,
    "post_first_tps": 7.339522306588453,
    "whole_request_tps": 6.865689290657791
  },
  {
    "id": "b04-warmup-mtp",
    "case_id": "b04-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.154165917017963,
    "first_token_seconds": 0.7916833340423182,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963462958,
      9519537100
    ],
    "output_tokens": 32,
    "post_first_tps": 9.219378609410196,
    "whole_request_tps": 7.703110718064666
  },
  {
    "id": "b05-warmup-mtp",
    "case_id": "b05-warmup",
    "method": "mtp",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 3.7430419590091333,
    "first_token_seconds": 0.7263340419740416,
    "prompt_tokens": 26,
    "accepted_drafts": 15,
    "verification_rounds": 17,
    "teacher_forwards": 17,
    "peak_mlx_bytes_by_rank": [
      7963528494,
      9519569868
    ],
    "output_tokens": 32,
    "post_first_tps": 10.276102576900353,
    "whole_request_tps": 8.549196175313812
  },
  {
    "id": "b05-warmup-native",
    "case_id": "b05-warmup",
    "method": "native",
    "warmup": true,
    "sample_repeat": 0,
    "prompt_sha256": "e99af5d8080cd12aeaa81dd07ec21190a350881feba8ef0d6003efdb39f46f66",
    "max_tokens": 32,
    "token_ids": [
      32,
      6951,
      369,
      264,
      13094,
      795,
      5759,
      421,
      25224,
      383,
      264,
      5339,
      30463,
      11,
      5339,
      81600,
      318,
      37,
      25166,
      8,
      7784,
      11,
      1332,
      5255,
      513,
      3568,
      310,
      279,
      1142,
      321,
      6747,
      494
    ],
    "finish_reason": "length",
    "wall_seconds": 4.806375457963441,
    "first_token_seconds": 0.5738272079615854,
    "prompt_tokens": 26,
    "accepted_drafts": 0,
    "verification_rounds": 0,
    "teacher_forwards": 0,
    "peak_mlx_bytes_by_rank": [
      7865083708,
      9320192875
    ],
    "output_tokens": 32,
    "post_first_tps": 7.324192937431112,
    "whole_request_tps": 6.657823609468714
  }
]
```

## Provenance (as supplied, not verified here)

```json
{
  "selected_batches": [
    0,
    1,
    2,
    3,
    4,
    5
  ],
  "full_suite_selected": true,
  "eos_token_ids": [
    248044,
    248046
  ],
  "profile": {
    "schema": "paretoquant-qwen38-fast-profile-v2",
    "status": "Verified bounded warm two-prompt >=10 tps pilot",
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    "local_runtime": "/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback",
    "source_sha256": {
      "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
      "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
      "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
      "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
      "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
      "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
      "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
      "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
      "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
      "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
      "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
      "mtp_runtime.py": "948cc881706a5463fc78628036873faf2ccf306a8b08b17faef48cbba6b64af4",
      "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
      "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
      "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
      "probe.py": "625ad275e76658400c8f70a2ba6e1096fc78f9f254ba9d67b14a172a9cbfe1ce",
      "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
      "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
      "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
      "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
      "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
      "remote_actor.py": "6479bcc8c8626c0590a394ba8fcb22fe22f58900cac8d84b68e4fc2497f9ce8c"
    },
    "parameters": {
      "remote_root": "/Users/jeremylien/Projects/paretoquant-dual-host/spikes/021-qwen38-prefix-rollback-20261016-v1",
      "remote_source": "/Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
      "local_metadata": "/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only",
      "output": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/021-prefix-full-prose-confirm-v2",
      "ssh_key": "/Users/jeremylien/.ssh/paretoquant_mini_ed25519",
      "split": [
        28,
        36
      ],
      "max_tokens": 96,
      "repeats": 3,
      "deadline": 220.0,
      "reserve_bytes": 8589934592,
      "local_address": "169.254.47.213",
      "remote_address": "169.254.214.155",
      "ssh_target": "jeremylien@fe80::14b2:4360:a750:94e5%bridge0",
      "prompt": "Why do two Macs connected by Thunderbolt not behave like a single shared-memory GPU? Explain in exactly two short sentences.",
      "execution": "phased",
      "eval_group_size": 32,
      "eval_group_schedule": null,
      "ssh_host_key_alias": "fe80::56:c507:c02a:a630%bridge0",
      "verification_gate_only": false,
      "prefix_rollback": true,
      "rank0_verifier": "native",
      "rank1_verifier": "slot-qmv",
      "target_revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
      "target_config_sha256": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
      "target_index_sha256": "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
      "sidecar": "/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-mtp-4bit-b643c01b6d3b094e325edb6ebd832e16c486c575",
      "small_model_local": "/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control",
      "small_model_remote": "/Users/jeremylien/Projects/paretoquant-dual-host/checkpoints/control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462",
      "small_config_sha256": "ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772",
      "small_index_sha256": "6e48f2fa5d6f033a6d77bf833abfa9698ca24d1715ecea4c67447bcfaee44650"
    },
    "evidence": "final-verification-v2.json",
    "target_dtype": "native BF16; original packed affine Q4/g64 unchanged",
    "sidecar_revision": "b643c01b6d3b094e325edb6ebd832e16c486c575",
    "warnings": [
      "Validate current bridge addresses and memory; do not kill unrelated apps",
      "Speculative acceptance and therefore throughput depend on prompt",
      "Whole-request throughput and cold setup are slower than post-first decoding",
      "All numerical gates rerun before every benchmark; no bypass flag",
      "Original source/failed evidence unchanged; no FP16 or requantization adopted"
    ]
  },
  "schedule_sha256": [
    "9794402e6fbfc6f680cd64a6579044103f0c83bd21664a50094187140d77072a",
    "557f406ed196a695626646281f693b865b34d339a7e54d73ad0b7bacab22e4d2",
    "11c5fe99205bed6cf64bfb7b4b2470c3bb74218f4d5f4f70a8d1f77e49855fc1",
    "78ec6bd11f5eb54fbc32175712456565fdada3e409fba878200ba6591c865288",
    "55a599b4986542a7d0db03aa9981f97bb222db702a5cd1b88b630e4516e08533",
    "6ada5280a4ae4889e8b257e86ba0a17316d304d8f2f916da9b31687a9ce52af2"
  ],
  "workloads": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/workloads",
  "runtime_normalization": "source-bound 023 controller and actual rank receipts; CPU readback",
  "memory_scope": "independent local-rank process MLX peaks; unused MTP head resident on rank0 for both methods",
  "trials": [
    {
      "batch_index": 0,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-00-v1",
      "expected_sources": {
        "benchmark_schedule.json": "9794402e6fbfc6f680cd64a6579044103f0c83bd21664a50094187140d77072a",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    },
    {
      "batch_index": 1,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-01-v1",
      "expected_sources": {
        "benchmark_schedule.json": "557f406ed196a695626646281f693b865b34d339a7e54d73ad0b7bacab22e4d2",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    },
    {
      "batch_index": 2,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-02-v1",
      "expected_sources": {
        "benchmark_schedule.json": "11c5fe99205bed6cf64bfb7b4b2470c3bb74218f4d5f4f70a8d1f77e49855fc1",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    },
    {
      "batch_index": 3,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-03-v1",
      "expected_sources": {
        "benchmark_schedule.json": "78ec6bd11f5eb54fbc32175712456565fdada3e409fba878200ba6591c865288",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    },
    {
      "batch_index": 4,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-04-v1",
      "expected_sources": {
        "benchmark_schedule.json": "55a599b4986542a7d0db03aa9981f97bb222db702a5cd1b88b630e4516e08533",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    },
    {
      "batch_index": 5,
      "trialdir": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/batch-05-v1",
      "expected_sources": {
        "benchmark_schedule.json": "6ada5280a4ae4889e8b257e86ba0a17316d304d8f2f916da9b31687a9ce52af2",
        "cache_state.py": "cdc01b5dc1dd91b44b4c8b454832a4f3fd44b9111f981239b19821ed712e3837",
        "changed_source_sha256.json": "92fb66e428738c432de615e6ee8d677bf113749895adcc7df84ddb3d5765c698",
        "cpu_communication.py": "43f5223f52aafe748ccee1478a96c0a61bb65d86815bfb8294375bd063b98e67",
        "distributed_control.py": "894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e",
        "mlx_backend.py": "507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47",
        "mtp_head.py": "95d05fc38dd532874ec820c39a0e1ec8e0a684906a896213b19609c721713d48",
        "mtp_runtime.py": "3517a8c44877e9b3175c5a6ee411752d74ed9279be58591bec2418f6205330cf",
        "numerical_gate.py": "43c95b98143466a8c4ee5a4dae3401f33bfc76f398b9db81e1f14a760a177e8c",
        "phased_decode.py": "b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790",
        "prefix_rollback.py": "2aa7defa57238aa4abfc90a31def2d4ed6852caef7f7ec8a95140474bc0f7130",
        "probe.py": "8bf2ac39f1171815bd7f952e0fdc1b8976e5c8668371bd576f5bca17006301ca",
        "rank_control.py": "e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5",
        "remote_actor.py": "21b3b77319947d824cd9d3836f0695d184d2fce6181016b0f88f15cb0045a3d0",
        "shard_plan.py": "231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465",
        "sidecar_config.json": "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
        "slot_qmv.py": "3ee7b0cc9a522b13e576fd288c9ed615a9380a956f4c7a5a8cf1da0637c65454",
        "slot_qmv_body.metal": "03f9ac57ad8fb1ab3d9424a2d677477e8cb81d5ca7527680a34b9dd18b6525df",
        "slot_qmv_header.metal": "936aa5fefd86bb53bf3ae8ba66a512145f9d04e22350097dad7c4abfabd3c7aa",
        "speculative.py": "6b94c442fbfd917b003c02001802f7e81a009a4c5874d5bc807c6b3156b770d7",
        "stream_server.py": "6d1d4bbb47e2f3c2f969388dba9f7dfa9f0524813c187ab8bf16a9af4d91caa1",
        "stream_weights.py": "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9",
        "supervise.py": "c4647cf873211adfe64f56ba812f546bb726666cb651684275ace855b8114596",
        "test_batched_snapshot.py": "be1002feb6fd661085b73bf4ef0f31bbb6bba2312231314cd90a1959bfb12b66",
        "test_cache_gate.py": "5ce8116c15a22f44a191b04b1fcf69b523f690ceaf7ebd88589340f52a7a6c42",
        "test_cli_schedule.py": "3d494ffee6cf4ccdd5e8939942aff5508108fc677068f4b27ce4298968545e07",
        "test_config_immutability.py": "30e281bae3adc0c2c493cb88600d69630b8f6103e0c8065257e81defba854d69",
        "test_dispatch_schedule.py": "05ca31b22aa6af70538cb12dd145db30903fcad7ca162c8d901935bc005e6b9c",
        "test_evidence_hardening.py": "f495a0ec7fae35b9172749fc28c2aa77931b6bf8bd2109a23acbc939ffa88a2a",
        "test_head.py": "bf976833baa25e5da6ddff8bb59338ab046950864cf3a4f2ae30b1ea0e651485",
        "test_integration.py": "c77b6ba8fbe9334d584b72e1ff0ea61fe098351d585f1efb33b7e0ec66df255d",
        "test_phased_cpu.py": "7b65176cc4751ba82ff92ec2ae257a22d501344dd0e901c1d245cc42dddf910e",
        "test_prefix_integration.py": "377d62e2ed2ff9f09196afd32a1aa788be54b39550060dab0ef042254c8532d5",
        "test_prefix_rollback.py": "d0f6f040542e4a4ed2bed704bf8af9b5cc15f63d6c4aadd7885b61e8283724d7",
        "test_speculative.py": "9cd3a4b0b956284e21801ac6805f93e5bd1fdc76b4f60d9cfd7bde1d77fdea17",
        "test_two_rank_schedule.py": "b67f9a1025738e22b7da3a0f7eb0f9041aa83cc81c03dbb0a063ad6ad4f7a708",
        "test_unbound_schedule.py": "f446a36ba5fe48f1cc88f1558c5ad4647af9c286e7bc004bd0cc7b85ae866f5a",
        "test_workload_caps.py": "117e18df911429b210bc96ed998dcbcac6e23304223758ebe3c4b9b9467fa503",
        "test_workload_cleanup.py": "e478df18900b76dc5dc949d8808db6e6f0cea2e977468d5538fc124b7300f9de",
        "test_workload_evidence.py": "b28e3f076ac78cfea2878f21af1677a406c810b268af3c274c3fd40ed6f76f12",
        "test_workload_runtime.py": "5a9e95b2e207bdf11410218771959224b683ed6bb4c306696018c3f38c24e267",
        "test_workload_schedule.py": "ae27e162390368b19d943f0c6f974722481d59a40b9aa18d8e00d34b0470f2d4",
        "test_workload_snapshot.py": "e8d26f9e3841f320b118a4ec8b148768cb60d72027b1e626fe5234bd71bc120f",
        "test_workload_timings.py": "761027cf05c067526ad2267a1a81bcd4f36774bdabc5e802abdec1cf7a00e748",
        "verification.py": "df7bdaa0eb811caa6f70cc51f479103c1e526d85e4b24913e875e0cd8a444478",
        "verifier.py": "ab5b3c25ebe14f17d6b1f1127b99e20d32b3fecee3d4fccd403f6100977a172f",
        "width_gate.py": "81528ae022bed8c910b9d1dcf045990cf9a8968a3978e106becdda14a4ee403c",
        "workload_evidence.py": "fd352236657af89f5295a0a07030eb8dbf4134a949b9e10484517670a415f3ae",
        "workload_runtime.py": "67dd60dee8c67618b7311c8ef4662b42c0486347d7354bf23063e860dc620a16",
        "workload_schedule.py": "0bad9bead01dd6fd78e8d1c732fa5ef83a1101deb950aded42333c09dabeeb9a"
      }
    }
  ],
  "collection_only": true,
  "manifest": "/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-broad-v1/all-six-trials.json"
}
```
