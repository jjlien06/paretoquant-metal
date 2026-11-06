"""CPU-only tests of the actual evaluation function via AST extraction.

Synthetic fixtures are tests only, never recorded as model-quality measurements.
No MLX import or GPU operation occurs.
"""
import ast,json,math,pathlib,tempfile,unittest
from types import SimpleNamespace
import numpy as np
P=pathlib.Path(__file__).resolve().parent

def extracted_evaluate():
    tree=ast.parse((P/'stream_v2.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    code=compile(ast.Module(body=[node],type_ignores=[]),str(P/'stream_v2.py'),'exec')
    mx=SimpleNamespace(float32=np.float32,array=np.array,sum=np.sum,
        take_along_axis=np.take_along_axis,logsumexp=lambda x,axis:np.logaddexp.reduce(x,axis=axis),eval=lambda *x:None)
    saved={};ids=np.array([[0,1,2,0],[2,2,1,0]],dtype=np.int32)
    logits=np.array([[2.,0.,-1.],[0.,3.,-1.],[-1.,0.,2.]],dtype=np.float32)
    widths=[]
    def head(h):widths.append(h.shape[1]);return logits[h[...,0].astype(int)]
    ns={'mx':mx,'math':math,'time':SimpleNamespace(monotonic=lambda:0.),'health':lambda event:{},
        'data_ids':lambda data:(ids,{'tokens':ids.ravel().tolist(),'split':'test','sequence_length':4}),
        'streamed_hidden':lambda cp,i:i[...,None], 'release':lambda:None,
        'dump':lambda path,obj:saved.update(obj)}
    exec(code,ns)
    cp=SimpleNamespace(head=lambda:head,args=SimpleNamespace(tie_word_embeddings=False),manifest=lambda:{'missing':[]})
    ns['evaluate'](cp,'fixture','unused')
    return saved,ids,logits,widths

class EvaluationContract(unittest.TestCase):
    def test_all_positions_projected_once_per_sequence(self):
        result,ids,logits,widths=extracted_evaluate();self.assertEqual(widths,[4,4])
    def test_shifted_nll_matches_independent_scalar_reference(self):
        result,ids,logits,widths=extracted_evaluate();expected=0.
        for row in ids:
            for i in range(len(row)-1):
                ls=[float(v) for v in logits[row[i]]]
                expected+=math.log(sum(math.exp(v) for v in ls))-ls[row[i+1]]
        self.assertAlmostEqual(result['nll_sum'],expected,places=5)
        self.assertAlmostEqual(result['perplexity'],math.exp(expected/6),places=5)
    def test_no_cross_window_target(self):
        result,ids,logits,widths=extracted_evaluate();self.assertEqual(result['scored_tokens'],6)
        self.assertEqual([s['scored_tokens'] for s in result['sequences']],[3,3])
        self.assertEqual(result['input_tokens'],8)
    def test_nll_sequence_totals(self):
        result,ids,logits,widths=extracted_evaluate()
        self.assertEqual(sum(s['nll_sum'] for s in result['sequences']),result['nll_sum'])

if __name__=='__main__':unittest.main()
