"""Independent post-hoc evidence linkage, not a replacement for physical gates."""
import hashlib,json,math,pathlib,time,urllib.request
from build_mixed import scan_checkpoint,hash_range
from allocate_real import allocate_inventory,text_tensor,layer_of
ROOT=pathlib.Path(__file__).resolve().parent

def read(name):return json.loads((ROOT/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)
if __name__=='__main__':
    verdict={'schema_version':1,'time':time.time(),'checks':{},'scope':'source/data/allocation/build/reload/resource evidence linkage; no timing or general task quality claim'}
    c=verdict['checks'];gate=read('small-parity-v2.json');sens=read('sensitivity-v2.json');inv=read('actual-inventory-v2.json');plans=read('allocator-plans-v2.json')
    require(gate['passed'] and len(gate['cases'])==3,'small gate')
    require(all(x['bit_exact'] and all(y['relative_rmse']==0 for y in x['per_layer_hidden']) for x in gate['cases']),'actual exact witnesses')
    require(sens['passed'] and [x['layer'] for x in sens['layers']]==list(range(64)),'calibration coverage/spread')
    c['small_gate_bit_exact']=True;c['calibration_64_layers']=True
    require(allocate_inventory(inv,sens)['plans']==plans['plans'],'independent exact allocation recomputation')
    c['allocation_recomputed_identically']=True
    train=read('data/train.json');test=read('data/test.json')
    for d,n in [(train,3072),(test,6144)]:
        require(len(d['tokens'])==n and d['sequence_length']==256,'data sizing')
        require(hashlib.sha256(json.dumps(d['tokens'],separators=(',',':')).encode()).hexdigest()==d['selected_token_sha256'],'actual token digest')
    require(train['split']=='train' and test['split']=='test' and train['source_parquet_sha256']!=test['source_parquet_sha256'],'disjoint source splits')
    c['actual_token_digests_verified']=True
    evals={name:read('eval-'+name+'-v2.json') for name in ['q4','mixed05','mixed15','q3']}
    for name,e in evals.items():
        require(e['data']=={k:v for k,v in test.items() if k!='tokens'},name+' evaluation metadata')
        require((e['input_tokens'],e['scored_tokens'],e['independent_sequences'],e['sequence_length'])==(6144,6120,24,256),name+' eval protocol counts')
        require([r['sequence'] for r in e['sequences']]==list(range(24)) and all(r['scored_tokens']==255 for r in e['sequences']),name+' sequence coverage')
        require(not e['checkpoint']['missing'],name+' loaded tensor coverage')
        require(math.isclose(e['perplexity'],math.exp(e['nll_sum']/6120),rel_tol=1e-14),name+' perplexity calculation')
    c['four_matched_quality_populations']=True
    source_hashes={}
    for name in ['q4','q3']:
        e=evals[name];cp=e['checkpoint'];source_hashes[name]=cp['loaded_tensor_sha256']
        require(cp['config_sha256']==inv[name]['config_sha256']==sens[name+'_checkpoint']['config_sha256'],name+' source config binding')
        actual=scan_checkpoint(pathlib.Path(cp['path']))
        require(set(source_hashes[name])=={n for n in actual if text_tensor(n)},name+' exact trusted text set')
        for tensor,h in sens[name+'_checkpoint']['loaded_tensor_sha256'].items():
            require(source_hashes[name].get(tensor)==h,name+' sensitivity/evaluation tensor hash '+tensor)
    c['calibration_source_tensor_hashes_equal_quality_sources']=True
    for name,idx in [('mixed05',0),('mixed15',2)]:
        m=read('checkpoint-'+name+'/mixed-manifest.json');cp=evals[name]['checkpoint']
        require(m['complete'] and m['all_tensor_hashes_verified'] and m['plan']==plans['plans'][idx],name+' build plan')
        require(m['q3_layers']==plans['plans'][idx]['q3_layers'],name+' selected layers')
        require(set(m['tensors'])==set(cp['loaded_tensor_sha256'])==set(source_hashes['q4']),name+' exact tensor inventory')
        for tensor,record in m['tensors'].items():
            require(record['sha256']==source_hashes[record['source']][tensor]==cp['loaded_tensor_sha256'][tensor],name+' loaded/source tensor hash '+tensor)
            expected='q3' if layer_of(tensor) in m['q3_layers'] and tensor.rsplit('.',1)[0]+'.scales' in inv['q3']['tensors'] else 'q4'
            require(record['source']==expected,name+' source selection '+tensor)
        c[name+'_every_loaded_tensor_matches_selected_source']=True
    # Rebind current complete checkpoint shard bytes to pinned upstream metadata.
    lfs={}
    for name,repo,rev in [('q4','mlx-community/Qwen3.8-27B-4bit','10c35caafbb80f7dc6a7a432cdd11af10a6d4818'),('q3','lukaskremla/Qwen3.8-27B-3bit-MLX','b3503b31b36125107e12a2b57fbe690c37bfc053')]:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=60) as f:api=json.load(f)
        require(api['sha']==rev,'pinned revision')
        with (ROOT/('assurance-hf-'+name+'.json')).open('x') as f:json.dump(api,f,indent=1)
        cp=pathlib.Path(evals[name]['checkpoint']['path']);entries=[]
        for file in api['siblings']:
            if not file['rfilename'].endswith('.safetensors'):continue
            p=cp/file['rfilename'];actual=sha(p)
            require(p.stat().st_size==file['size'] and actual==file['lfs']['sha256'],name+' LFS shard '+file['rfilename'])
            entries.append({'file':file['rfilename'],'size':p.stat().st_size,'sha256':actual,'lfs_equal':True})
        lfs[name]=entries
    verdict['lfs_shards']=lfs;c['both_sources_current_shards_match_pinned_lfs']=True
    guards={}
    for name in ['small-v2','calibration-v2','input-audit','eval-q4-v2','build-mixed05-v2','eval-mixed05-v2','build-mixed15-v2','eval-mixed15-v2','eval-q3-v2']:
        rows=[json.loads(x) for x in (ROOT/(name+'-guard.jsonl')).read_text().splitlines()]
        require(rows[-1]['event']=='exit' and rows[-1]['returncode']==0 and rows[-1]['failure'] is None and rows[-1]['child_reaped'],name+' guard exit')
        samples=[x for x in rows if x['event']=='sample']
        require(all(x['pressure'] in (1,2) and x['rss_bytes']<3*2**30 for x in samples),name+' guard samples')
        guards[name]={'samples':len(samples),'peak_sampled_rss_bytes':max((x['rss_bytes'] for x in samples),default=0),'pressures':sorted({x['pressure'] for x in samples}),'reaped':True}
    verdict['guards']=guards
    verdict['artifact_sha256']={p.name:sha(p) for p in [ROOT/'stream_v2.py',ROOT/'allocate_real.py',ROOT/'build_mixed.py',ROOT/'actual-inventory-v2.json',ROOT/'sensitivity-v2.json',ROOT/'allocator-plans-v2.json',ROOT/'data/test.json',ROOT/'data/train.json']}
    verdict['passed']=True
    with (ROOT/'final-assurance-v2.json').open('x') as f:json.dump(verdict,f,indent=2)
    print(json.dumps(verdict,indent=2))
