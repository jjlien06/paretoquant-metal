B=/lab/engines/llama.cpp@b11968/build/bin/llama-bench
M=/models/Qwen3.8-27B-UD-Q4_K_M.gguf
run(){ name=$1; shift; echo "### BEGIN $name $(date -u +%FT%TZ)"; $B -m $M -fa 1 -b 2048 -ub 512 -p 512 -n 128 -r 3 -o json "$@" 2>/tmp/err; echo "### EXIT $name $? $(date -u +%FT%TZ)"; tail -3 /tmp/err | sed 's/^/### ERR /'; }
run D1_ngl28_loadnone -ngl 28 -t 8 -lm none
free -g | sed 's/^/### MEM /'
run D2_ngl32_loadnone -ngl 32 -t 8 -lm none
