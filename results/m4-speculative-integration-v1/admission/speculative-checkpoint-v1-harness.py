from pathlib import Path
import sys,json,hashlib
root=Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-spec-integration-v2-1791220091510084000');sys.path.insert(0,str(root));sys.path.insert(1,'/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-draft-screen-1791216672479820000')
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_module
from unittest.mock import patch
import speculative_decode,phased_decode,draft_screen
checkpoint=Path('/Users/jeremylien/Projects/paretoquant-dual-host/small-model');before=draft_screen.inventory(checkpoint);expected_config='43a1c2f3b692e8cca5974f4753514e94ee4c969f8853b6d02acc16c7e64805da';assert hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()==expected_config
prev=mx.default_device();mx.set_default_device(mx.gpu);oldcache=mx.set_cache_limit(256*1024*1024);mx.reset_peak_memory();created=[];original_make_cache=cache_module.make_prompt_cache
try:
 model,tokenizer=load(str(checkpoint));mx.eval(model.parameters());nbytes=sum(x.nbytes for _,x in __import__('mlx.utils',fromlist=['tree_flatten']).tree_flatten(model.parameters()));assert nbytes==260563712
 reference=json.loads((Path('/Users/jeremylien/Projects/paretoquant-dual-host/spikes/009-draft-screen-1791216672479820000')/'reference.json').read_text());prompts=[reference['formatted_prompt'],tokenizer.apply_chat_template([{'role':'user','content':'Write three short facts about the Moon.'}],tokenize=False,add_generation_prompt=True)]
 rows=[]
 def tracked(owner):
  cache=original_make_cache(owner);created.append(cache);return cache
 for pi,prompt in enumerate(prompts):
  for k in [1,4,8]:
   created.clear();events=[]
   with patch.object(cache_module,'make_prompt_cache',tracked):
    control=phased_decode.phased_generate(model,tokenizer,prompt,max_tokens=32,eval_group_size=8,prefill_step_size=32)
    trial=speculative_decode.speculative_generate(model,tokenizer,prompt,draft_model=model,draft_tokenizer=tokenizer,max_tokens=32,draft_tokens=k,eval_group_size=8,prefill_step_size=32,progress=events.append)
   offsets=[state.offset for cache in created for state in cache];offset=control['prompt_tokens']+len(control['token_ids'])-1;kvclose=all(np.allclose(np.array(l[..., :offset, :]),np.array(r[..., :offset, :]),rtol=.002,atol=.002) for other in created[1:] for a,b in zip(created[0],other) for l,r in zip(a.state[:2],b.state[:2]));rounds=[e for e in events if e['phase']=='speculative_round_done'];row={'prompt_index':pi,'k':k,'control':control,'speculative':trial,'all_cache_offsets_match':all(n==offset for n in offsets),'live_kv_close':kvclose,'rounds':rounds,'passed':control['token_ids']==trial['token_ids'] and control['finish_reason']==trial['finish_reason'] and all(n==offset for n in offsets) and kvclose};rows.append(row)
 after=draft_screen.inventory(checkpoint);assert before==after
 report={'passed':all(r['passed'] for r in rows),'cases':rows,'case_count':len(rows),'model_bytes':nbytes,'same_model_self_draft':True,'scope':'Real mixed3/4bit0.5B unchanged checkpoint self-draft correctness, no32B orspeedup claim','prompt_strings':prompts,'inventory':before,'config_sha256':expected_config,'sources':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['speculative_decode.py','phased_decode.py']},'peak_mlx_bytes':mx.get_peak_memory(),'device':mx.device_info()}
 with(root/'checkpoint-v1.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'passed':report['passed'],'cases':len(rows),'failures':[(r['prompt_index'],r['k']) for r in rows if not r['passed']],'peak_mlx_bytes':report['peak_mlx_bytes']}))
finally:
 mx.set_cache_limit(oldcache);mx.set_default_device(prev)
