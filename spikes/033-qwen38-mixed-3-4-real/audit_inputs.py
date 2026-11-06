"""Bounded CPU inventory audit using real downloaded headers and raw source hashes."""
import hashlib,json,pathlib
from build_mixed import scan_checkpoint,hash_range
from allocate_real import PFX,decoder_modules,PARTS,text_tensor
from guard import preflight
ROOT=pathlib.Path(__file__).resolve().parent
Q4=ROOT.parents[1]/'checkpoints/qwen38-27b-10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
Q3=ROOT/'q3-source'
if __name__=='__main__':
    preflight();a=scan_checkpoint(Q4);b=scan_checkpoint(Q3)
    assert set(a)==set(b)
    mods=decoder_modules(a);assert mods==decoder_modules(b) and len(mods)==496
    quant={m+'.'+part for m in mods for part in PARTS}
    rows=[]
    for name in sorted(a):
        if text_tensor(name) and name not in quant:
            # Embedding/head are quantized too; they are fixed q4 in mixed plans.
            if '.embed_tokens.' in name or '.lm_head.' in name:continue
            ia,ib=a[name],b[name]
            assert ia['dtype']==ib['dtype'] and ia['shape']==ib['shape']
            ha,hb=hash_range(ia),hash_range(ib)
            rows.append({'tensor':name,'q4_sha256':ha,'q3_sha256':hb,'equal':ha==hb})
    assert all(r['equal'] for r in rows),'floating tensor mismatch: calibrate q4-float mixed versus source q3 must be labelled'
    inv={}
    for key,path,tensors in [('q4',Q4,a),('q3',Q3,b)]:
        cfg=json.loads((path/'config.json').read_text())
        inv[key]={'tensors':tensors,'config':cfg,'root':str(path),'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest()}
    inv['nonquantized_decoder_and_norm_equality']=rows
    with (ROOT/'actual-inventory-v2.json').open('x') as f:json.dump(inv,f,indent=1)
    print(json.dumps({'tensor_count':len(a),'decoder_quantized_modules':len(mods),'nonquantized_tensors_checked':len(rows),'all_nonquantized_equal':True}))
