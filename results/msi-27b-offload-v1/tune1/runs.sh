B=/lab/engines/llama.cpp@b11968/build/bin/llama-bench
M=/models/Qwen3.8-27B-UD-Q4_K_M.gguf
run(){ name=$1; shift; echo "### BEGIN $name $(date -u +%FT%TZ)"; $B -m $M -fa 1 -b 2048 -ub 512 -p 512 -n 128 -r 3 -o json "$@" 2>/tmp/err; echo "### EXIT $name $? $(date -u +%FT%TZ)"; tail -3 /tmp/err | sed 's/^/### ERR /'; }
run A1_ngl28_t8 -ngl 28 -t 8
run A2_ngl32_q8kv -ngl 32 -t 8 -ctk q8_0 -ctv q8_0
run A3_ngl34_q8kv -ngl 34 -t 8 -ctk q8_0 -ctv q8_0
run A4_ngl36_q8kv -ngl 36 -t 8 -ctk q8_0 -ctv q8_0
run A5_attnGPU_ffnCPU22 -ngl 99 -t 8 -ot 'blk\.(2[2-9]|[3-5][0-9]|6[0-3])\.ffn_.*=CPU'
run A6_attnGPU_ffnCPU26 -ngl 99 -t 8 -ot 'blk\.(2[6-9]|[3-5][0-9]|6[0-3])\.ffn_.*=CPU'
run A7_ngl28_t6 -ngl 28 -t 6
run A8_ngl28_t16 -ngl 28 -t 16
