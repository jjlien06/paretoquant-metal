"""Create a new candidate, preserving the originally executed stream.py."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'stream.py').read_text()
a=s.index('def project(');b=s.index('def data_ids(',a)
s=s[:a]+'''def project(cp,h):
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

''' + s[b:]
# All 256 positions use one head call. Select next-token targets only afterward.
a=s.index('        for offset in range(0,ids.shape[1]-1,16):');b=s.index('        total+=seq_nll;',a)
s=s[:a]+'''        logits=fn(h[j:j+1]).astype(mx.float32)
        labels=mx.array(ids[j:j+1,1:])
        targets=logits[:,:-1,:]
        selected=mx.take_along_axis(targets,labels[...,None],axis=-1).squeeze(-1)
        nll=mx.sum(mx.logsumexp(targets,axis=-1)-selected);mx.eval(nll)
        value=float(nll.item());assert math.isfinite(value)
        seq_nll+=value;count+=ids.shape[1]-1
        del logits,labels,targets,selected,nll
''' +s[b:]
with (p/'stream_v2.py').open('x') as f:f.write(s)
