"""Serial guarded builds and evaluation after successful preceding receipts."""
import json,pathlib,subprocess,sys,os
ROOT=pathlib.Path(__file__).resolve().parent
PY='/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'
Q4=ROOT.parents[1]/'checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818'

def run(name,argv,seconds):
    with (ROOT/(name+'.log')).open('x') as f:
        cmd=[PY,str(ROOT/'guard.py'),'--receipt',str(ROOT/(name+'-guard.jsonl')),'--seconds',str(seconds),'--',PY,*map(str,argv)]
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT/'dependency'),'PYTHONDONTWRITEBYTECODE':'1'},timeout=seconds+45)
    if r.returncode:raise RuntimeError(f'{name} failed: {r.returncode}')
    print(name+' completed',flush=True)

if __name__=='__main__':
    assert json.loads((ROOT/'sensitivity-v2.json').read_text())['passed']
    assert json.loads((ROOT/'small-parity-v2.json').read_text())['passed']
    # The initial q4 evaluation must already be finished, not concurrently running.
    exits=[json.loads(x) for x in (ROOT/'eval-q4-v2-guard.jsonl').read_text().splitlines()]
    assert exits[-1]['event']=='exit' and exits[-1]['returncode']==0 and exits[-1]['failure'] is None
    for idx,label in [(0,'mixed05'),(2,'mixed15')]:
        target=ROOT/('checkpoint-'+label)
        run('build-'+label+'-v2',[ROOT/'build_mixed.py','--q4',Q4,'--q3',ROOT/'q3-source','--plan',ROOT/'allocator-plans-v2.json','--plan-index',idx,'--output',target],600)
        assert json.loads((target/'mixed-manifest.json').read_text())['complete']
        run('eval-'+label+'-v2',[ROOT/'stream_v2.py','evaluate','--model',target,'--data',ROOT/'data/test.json','--output',ROOT/('eval-'+label+'-v2.json')],1200)
    run('eval-q3-v2',[ROOT/'stream_v2.py','evaluate','--model',ROOT/'q3-source','--data',ROOT/'data/test.json','--output',ROOT/'eval-q3-v2.json'],1200)
