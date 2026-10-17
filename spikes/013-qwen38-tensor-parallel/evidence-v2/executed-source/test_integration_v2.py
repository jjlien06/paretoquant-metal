"""Real two-process native/candidate controls; unchanged original gates."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import pytest
from test_tp import checkpoint


@pytest.mark.parametrize('precise',[False,True],ids=['native','high-precision'])
def test_two_process_original_gates(checkpoint,tmp_path,precise):
    path,cfg,full=checkpoint
    from mlx.utils import tree_flatten
    import mlx.core as mx
    dtype='bf16' if any(a.dtype==mx.bfloat16 for _,a in tree_flatten(full.parameters())) else 'f32'
    mode='precise' if precise else 'native'
    evidence=Path(__file__).parent/'evidence-v2'
    evidence.mkdir(exist_ok=True)
    sockets=[]
    for _ in (0,1):
        s=socket.socket();s.bind(('127.0.0.1',0));sockets.append(s)
    hosts=[[f'127.0.0.1:{s.getsockname()[1]}'] for s in sockets]
    for s in sockets:s.close()
    hf=tmp_path/'ring.json';hf.write_text(json.dumps(hosts))
    processes=[]
    reports=[]
    try:
        for rank in (0,1):
            output=evidence/f'{mode}-{dtype}-rank{rank}.json'
            assert not output.exists(),'Refuse evidence overwrite'
            cmd=[sys.executable,str(Path(__file__).with_name('runner.py')),
                '--model',str(path),'--budget-bytes',str(2**20),'--reference-small','--output',str(output)]
            if precise:cmd.append('--high-precision-row-reductions')
            processes.append(subprocess.Popen(cmd,env=dict(os.environ,MLX_HOSTFILE=str(hf),MLX_RANK=str(rank)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        receipts=[]
        for rank,p in enumerate(processes):
            out,err=p.communicate(timeout=45)
            receipts.append(dict(rank=rank,returncode=p.returncode,stdout=out,stderr=err))
            reports.append(json.loads((evidence/f'{mode}-{dtype}-rank{rank}.json').read_text()))
        with (evidence/f'{mode}-{dtype}-processes.json').open('x') as f:json.dump(receipts,f,indent=2)
        for r in reports:
            assert len(r['numerics']['steps'])==5
            assert r['collective_calls']==30
            assert r['layers_executed']==[0,1,2]
            assert len(r['tokens'])==3
            assert set(r['source_tensor_sha256'])==set(r['loading']['tensor_sha256'])
        assert reports[0]['tokens']==reports[1]['tokens']
        assert reports[0]['loading']['replicated_sha256']==reports[1]['loading']['replicated_sha256']
        assert all(r['numerics']['passed'] for r in reports),[(r['rank'],r['numerics']) for r in reports]
        assert all(r['returncode']==0 for r in receipts),receipts
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait(timeout=5)
