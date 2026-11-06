"""One-layer-resident native Qwen3.5-family calibration and capped WikiText NLL.

No model arithmetic is reimplemented: native DecoderLayer, Embedding, RMSNorm,
quantized Linear, mask helpers and uncached GDN/SDPA paths are used verbatim.
Independent 256-token sequences reset positions/state; no cross-window cache.
"""
import argparse,copy,gc,hashlib,importlib.metadata,json,math,pathlib,resource,sys,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModelArgs,DecoderLayer
from mlx_lm.models.base import create_attention_mask,create_ssm_mask
from build_mixed import scan_checkpoint,hash_range
from guard import preflight

ROOT=pathlib.Path(__file__).resolve().parent
PFX='language_model.model.layers.'
EMBED='language_model.model.embed_tokens'
HEAD='language_model.lm_head'
NORM='language_model.model.norm'
mx.set_cache_limit(64*2**20)
mx.set_memory_limit(2300*2**20)
mx.set_wired_limit(1600*2**20)
DT={'U32':np.uint32,'F32':np.float32,'F16':np.float16,'BF16':np.uint16}

def dump(path,value):
    with pathlib.Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def release():
    gc.collect();mx.clear_cache()

def health(event):
    row=preflight();row.update(event=event,mlx_active=mx.get_active_memory(),mlx_peak=mx.get_peak_memory(),maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(row),flush=True)
    if row['maxrss_bytes']>=3*2**30:raise RuntimeError('process RSS high-water exceeds 3GiB')
    if row['mlx_peak']>=2300*2**20:raise RuntimeError('MLX peak exceeds bound')
    return row

class Checkpoint:
    def __init__(self,path):
        self.root=pathlib.Path(path);self.config=json.loads((self.root/'config.json').read_text())
        self.args=TextModelArgs.from_dict(copy.deepcopy(self.config.get('text_config',self.config)))
        self.inv=scan_checkpoint(self.root)
        self.quant=self.config['quantization']
        self.loaded_hashes={}
        self.nlayers=self.args.num_hidden_layers
        assert self.config['model_type']=='qwen3_5'
        assert self.args.num_experts==0
        types=self.config.get('text_config',self.config).get('layer_types')
        if types:assert types==['full_attention' if (i+1)%self.args.full_attention_interval==0 else 'linear_attention' for i in range(self.nlayers)]
        self.expected={n for n in self.inv if n.startswith(PFX) or n.startswith(EMBED+'.') or n.startswith(HEAD+'.') or n.startswith(NORM+'.')}
        assert all(not n.startswith('language_model.') or n in self.expected for n in self.inv),'unrecognized language model weights'
    def tensor(self,name):
        info=self.inv[name];size=info['end']-info['start'];assert size<=768*2**20
        with open(info['file'],'rb') as f:
            f.seek(info['start']);raw=f.read(size)
        assert len(raw)==size
        self.loaded_hashes[name]=hashlib.sha256(raw).hexdigest()
        assert info['dtype'] in DT,(name,info['dtype'])
        a=mx.array(np.frombuffer(raw,dtype=DT[info['dtype']]).reshape(info['shape']))
        if info['dtype']=='BF16':a=a.view(mx.bfloat16)
        mx.eval(a)
        return a
    def qcfg(self,path):
        v=self.quant.get(path,{k:self.quant[k] for k in ('bits','group_size','mode') if k in self.quant})
        assert isinstance(v,dict) and v['bits'] in (3,4) and v['group_size']==64 and v.get('mode','affine')=='affine',(path,v)
        return v
    def quantize(self,module,prefix):
        # These random initialization graphs are overwritten before any evaluation.
        def pred(path,m):
            p=prefix+('.'+path if path else '')
            if p+'.scales' not in self.inv:return False
            assert hasattr(m,'to_quantized')
            return self.qcfg(p)
        nn.quantize(module,group_size=64,bits=self.quant['bits'],mode='affine',class_predicate=pred)
    def assign(self,module,prefix,other=None,only_quantized_other=False):
        names=sorted(n for n in self.inv if n.startswith(prefix+'.'))
        selected=[]
        for n in names:
            use=other if other is not None and (not only_quantized_other or (n.rsplit('.',1)[0]+'.scales') in other.inv) else self
            if use is other and only_quantized_other:
                assert self.inv[n]['dtype']==other.inv[n]['dtype']
                if not n.endswith('.weight'):assert self.inv[n]['shape']==other.inv[n]['shape']
            selected.append((n[len(prefix)+1:],use.tensor(n)))
        module.load_weights(selected,strict=True);module.eval();mx.eval(module.parameters())
        return module
    def layer(self,i,other=None):
        prefix=PFX+str(i)
        health(f'before-layer-{i}')
        layer=DecoderLayer(self.args,i)
        # For demotions, use q3 quantized modules but q4 floating tensors.
        (other or self).quantize(layer,prefix)
        self.assign(layer,prefix,other,only_quantized_other=other is not None)
        return layer
    def embedding(self):
        health('before-embedding')
        module=nn.QuantizedEmbedding(self.args.vocab_size,self.args.hidden_size,**self.qcfg(EMBED))
        return self.assign(module,EMBED)
    def norm(self):
        return self.assign(nn.RMSNorm(self.args.hidden_size,eps=self.args.rms_norm_eps),NORM)
    def head(self):
        health('before-head')
        if self.args.tie_word_embeddings:return self.embedding()
        return self.assign(nn.QuantizedLinear(self.args.hidden_size,self.args.vocab_size,bias=False,**self.qcfg(HEAD)),HEAD)
    def manifest(self):
        return {'path':str(self.root),'config_sha256':hashlib.sha256((self.root/'config.json').read_bytes()).hexdigest(),'loaded_tensor_sha256':self.loaded_hashes,'expected_text_tensor_count':len(self.expected),'loaded_tensor_count':len(self.loaded_hashes),'missing':sorted(self.expected-self.loaded_hashes.keys())}

def embed(cp,ids):
    module=cp.embedding();h=module(mx.array(ids));mx.eval(h);del module;release();return h

def apply_layer(layer,h):
    outputs=[]
    for j in range(h.shape[0]):
        x=h[j:j+1]
        mask=create_ssm_mask(x,None) if layer.is_linear else create_attention_mask(x,None)
        y=layer(x,mask=mask,cache=None);mx.eval(y);outputs.append(y)
    result=mx.concatenate(outputs,axis=0);mx.eval(result)
    return result

def streamed_hidden(cp,ids,progress=None):
    h=embed(cp,ids)
    for i in range(cp.nlayers):
        layer=cp.layer(i);y=apply_layer(layer,h);del layer,h;h=y;release()
        if progress:progress(i)
    module=cp.norm();h=module(h);mx.eval(h);del module;release();return h

def project(cp,h):
    module=cp.head();fn=module.as_linear if cp.args.tie_word_embeddings else module
    logits=fn(h);mx.eval(logits);del module;release();return logits

def gate_small(cp,out):
    from mlx_lm.utils import load_model
    report={'scope':'uncached native full forward vs one-decoder-layer-resident; full sequence projection widths',
        'amended_gate':{'authorized_by':'Jeremy continuation request','argmax_equal':True,'max_abs_diff':0.125,'mean_abs_diff':0.001,'compute':'BF16'},
        'original_gate':{'atol':1e-5,'rtol':1e-5,'failure_preserved':'small-parity-r2.json'},
        'cases':[],'versions':{k:importlib.metadata.version(k) for k in ('mlx','mlx-lm','numpy')}}
    original_call=DecoderLayer.__call__
    for length in (17,64,256):
        health('before-small-native')
        model,cfg=load_model(cp.root,lazy=True);model.eval();mx.eval(model.parameters())
        ids=np.array([[101+(i*173)%4000 for i in range(length)]],dtype=np.int32)
        expected=model(mx.array(ids));mx.eval(expected)
        assert expected.dtype==mx.bfloat16,str(expected.dtype)
        ref=np.asarray(expected.astype(mx.float32)).copy();del expected
        native_hidden=[]
        def capture(layer,x,*args,**kwargs):
            y=original_call(layer,x,*args,**kwargs);mx.eval(y)
            native_hidden.append(np.asarray(y.astype(mx.float32)).copy())
            return y
        try:
            DecoderLayer.__call__=capture
            witnessed=model(mx.array(ids));mx.eval(witnessed)
            witness=np.asarray(witnessed.astype(mx.float32));wdiff=np.abs(witness-ref)
            witness_metrics={'max_abs_diff':float(wdiff.max()),'mean_abs_diff':float(wdiff.mean()),'argmax_equal':bool(np.array_equal(witness.argmax(-1),ref.argmax(-1)))}
            del witnessed,witness,wdiff
        finally:DecoderLayer.__call__=original_call
        del model;release()
        assert len(native_hidden)==cp.nlayers
        h=embed(cp,ids);layer_errors=[]
        for i in range(cp.nlayers):
            layer=cp.layer(i);y=apply_layer(layer,h)
            actual_hidden=np.asarray(y.astype(mx.float32));reference_hidden=native_hidden[i]
            delta=actual_hidden-reference_hidden
            layer_errors.append({'layer':i,'relative_rmse':float(np.sqrt(np.sum(delta.astype(np.float64)**2)/np.sum(reference_hidden.astype(np.float64)**2))),
                'max_abs_diff':float(np.abs(delta).max()),'dtype':str(y.dtype)})
            assert y.dtype==mx.bfloat16
            del layer,h,actual_hidden,reference_hidden,delta;h=y;release()
        del native_hidden
        norm=cp.norm();h=norm(h);mx.eval(h);del norm;release()
        logits=project(cp,h);actual=np.asarray(logits.astype(mx.float32));diff=np.abs(actual-ref)
        row={'length':length,'logits_shape':list(ref.shape),'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean()),
            'next_token_equal':bool(np.array_equal(actual.argmax(-1),ref.argmax(-1))),
            'original_allclose':bool(np.allclose(actual,ref,atol=1e-5,rtol=1e-5)),
            'bit_exact':bool(np.array_equal(actual,ref)),'per_layer_hidden':layer_errors,'capture_witness_vs_pristine':witness_metrics,'compute_dtype':str(logits.dtype)}
        row['passed']=row['next_token_equal'] and row['max_abs_diff']<=.125 and row['mean_abs_diff']<=.001
        report['cases'].append(row);print(json.dumps(row),flush=True)
        del logits,h,actual,ref,diff;release()
        if not row['passed']:
            report['passed']=False;dump(out,report);raise RuntimeError('amended small streaming gate failed')
    report['passed']=True;report['checkpoint']=cp.manifest();report['health']=health('small-finished');dump(out,report)

def data_ids(path):
    d=json.loads(pathlib.Path(path).read_text());ids=np.array(d['tokens'],dtype=np.int32);size=d['sequence_length'];assert len(ids)%size==0
    return ids.reshape(-1,size),d

def calibrate(cp,q3,data,out):
    ids,meta=data_ids(data);h=embed(cp,ids);rows=[]
    with pathlib.Path(str(out)+'.partial.jsonl').open('x') as log:
        for i in range(cp.nlayers):
            start=time.monotonic();layer=cp.layer(i);ref=apply_layer(layer,h);del layer;release()
            layer=cp.layer(i,q3);cand=apply_layer(layer,h);del layer;release()
            numerator=mx.sum((cand.astype(mx.float32)-ref.astype(mx.float32))**2)
            denominator=mx.sum(ref.astype(mx.float32)**2)
            residual=mx.sum((ref.astype(mx.float32)-h.astype(mx.float32))**2)
            mx.eval(numerator,denominator,residual)
            mse=float(numerator.item())/float(denominator.item());rerr=float(numerator.item())/float(residual.item())
            assert math.isfinite(mse) and mse>0
            row={'layer':i,'type':'linear_attention' if (i+1)%cp.args.full_attention_interval else 'full_attention','relative_mse':mse,'relative_rmse':math.sqrt(mse),'residual_relative_mse':rerr,'squared_error_sum':numerator.item(),'reference_squared_sum':denominator.item(),'wall_seconds':time.monotonic()-start}
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            del h,cand,numerator,denominator,residual;h=ref;release()
    losses=np.array([r['relative_mse'] for r in rows]);p10,p90=np.quantile(losses,[.1,.9]);spread={'minimum':float(losses.min()),'median':float(np.median(losses)),'maximum':float(losses.max()),'max_min_ratio':float(losses.max()/losses.min()),'p90_p10_ratio':float(p90/p10),'coefficient_of_variation':float(losses.std()/losses.mean())}
    # Predeclared meaningful spread: >=1.5 interdecile ratio and >=0.15 CV.
    passed=spread['p90_p10_ratio']>=1.5 and spread['coefficient_of_variation']>=.15
    dump(out,{'layers':rows,'spread':spread,'passed':passed,'spread_gate':{'p90_p10_ratio_minimum':1.5,'coefficient_of_variation_minimum':.15},'data':{k:v for k,v in meta.items() if k!='tokens'},'q4_checkpoint':cp.manifest(),'q3_checkpoint':q3.manifest(),'method':'sum ||layer3(h4)-layer4(h4)||^2 / sum ||layer4(h4)||^2; h4 is actual uncached native q4 hidden state; quantized decoder linears only demoted; no Fisher or final-logit KL claim','health':health('calibration-finished')})
    if not passed:raise RuntimeError('sensitivities do not meaningfully differ')

def evaluate(cp,data,out):
    ids,meta=data_ids(data);start=time.monotonic();h=streamed_hidden(cp,ids)
    module=cp.head();fn=module.as_linear if cp.args.tie_word_embeddings else module
    total=0.;count=0;seqs=[]
    for j in range(ids.shape[0]):
        seq_nll=0.
        logits=fn(h[j:j+1]).astype(mx.float32)
        labels=mx.array(ids[j:j+1,1:])
        targets=logits[:,:-1,:]
        selected=mx.take_along_axis(targets,labels[...,None],axis=-1).squeeze(-1)
        nll=mx.sum(mx.logsumexp(targets,axis=-1)-selected);mx.eval(nll)
        value=float(nll.item());assert math.isfinite(value)
        seq_nll+=value;count+=ids.shape[1]-1
        del logits,labels,targets,selected,nll
        total+=seq_nll;seqs.append({'sequence':j,'nll_sum':seq_nll,'scored_tokens':ids.shape[1]-1})
    assert count==ids.shape[0]*(ids.shape[1]-1)
    del module,h;release();manifest=cp.manifest();assert not manifest['missing'],manifest['missing']
    dump(out,{'perplexity':math.exp(total/count),'mean_nll':total/count,'nll_sum':total,'scored_tokens':count,'input_tokens':int(ids.size),'independent_sequences':ids.shape[0],'sequence_length':ids.shape[1],'context_reset':'each nonoverlapping 256-token sequence, no BOS; first token in each unscored','sequences':seqs,'wall_seconds':time.monotonic()-start,'data':{k:v for k,v in meta.items() if k!='tokens'},'checkpoint':manifest,'health':health('evaluation-finished')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['small','calibrate','evaluate']);ap.add_argument('--model',required=True);ap.add_argument('--q3');ap.add_argument('--data');ap.add_argument('--output',required=True);a=ap.parse_args()
    health('startup');cp=Checkpoint(a.model)
    if a.mode=='small':gate_small(cp,a.output)
    elif a.mode=='calibrate':calibrate(cp,Checkpoint(a.q3),a.data,a.output)
    else:evaluate(cp,a.data,a.output)
