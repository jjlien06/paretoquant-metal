# 031 width-3 verify kernel
See results/qwen38-width3-kernel-v1/README.md. New: slot_qmv_t.py, slot_qmv_{header,body}_t.metal, geometry_table.json, tuned modes in verifier.py, tune.py/op_screen.py/profile_width.py/small_exact.py screens, summarize.py, test_tuned.py. Rejected candidates kept: slot_qmv_w.py/slot_qmv_header_w.metal (word-load; first version inexact due to type punning), MODE 2.
