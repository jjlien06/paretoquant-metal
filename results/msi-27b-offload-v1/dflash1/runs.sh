S=/lab/engines/llama.cpp@b11968/build/bin/llama-server
M=/models/Qwen3.8-27B-UD-Q4_K_M.gguf; MD=/models/mtp-Qwen3.8-27B-Q4_0.gguf
cat > /tmp/p.py <<'P'
import json,sys,urllib.request,time
prompts=["Write a Python function that merges two sorted lists, then explain its time complexity.",
"Explain how a rocket engine's turbopump works, step by step, for an engineering student.",
"List the first 40 prime numbers separated by commas, then describe the pattern of gaps.",
"Summarize the causes and consequences of the 2008 financial crisis in five paragraphs."]
for rep in range(1):
  for i,p in enumerate(prompts):
    body={"messages":[{"role":"user","content":p}],"max_tokens":256,"temperature":0,"seed":1,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=900))
    tm=r.get("timings",{})
    print("ROW",json.dumps({"rep":rep,"prompt":i,"wall":round(time.time()-t,2),"n":r["usage"]["completion_tokens"],"pp_tps":tm.get("prompt_per_second"),"tg_tps":tm.get("predicted_per_second"),"draft_n":tm.get("draft_n"),"draft_acc":tm.get("draft_n_accepted"),"head":r["choices"][0]["message"]["content"][:80]}),flush=True)
P
serve(){ name=$1; shift; echo "### BEGIN $name $(date -u +%FT%TZ)"
  $S -m $M -ngl 22 -fa on -t 8 -c 4096 -np 1 --host 127.0.0.1 --port 8080 --no-webui "$@" >/tmp/s.log 2>&1 & pid=$!
  for k in $(seq 1 120); do python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')" 2>/dev/null && break; sleep 2; done
  python3 /tmp/p.py; kill $pid; wait $pid 2>/dev/null
  grep -i -E "error|mtp|draft|dflash|offload|out of memory|alloc|CUDA0 model buffer" /tmp/s.log | grep -v "draft acceptance" | head -16 | sed "s/^/### LOG /"; grep "draft acceptance" /tmp/s.log | sed "s/^/### ACC /"; tail -3 /tmp/s.log | sed "s/^/### TAIL /"; echo "### END $name"; }
DD=/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
serve D1_off
serve D2_dflash7 --spec-type draft-dflash -md $DD -ngld 99 --spec-draft-n-max 7
serve D3_dflash15 --spec-type draft-dflash -md $DD -ngld 99 --spec-draft-n-max 15
serve D4_mtp3 --spec-type draft-mtp -md $MD -ngld 99 --spec-draft-n-max 3
serve D5_off_repeat
