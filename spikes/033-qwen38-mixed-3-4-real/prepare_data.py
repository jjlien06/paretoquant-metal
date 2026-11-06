"""Pinned WikiText2 raw split data; no MLX import or expected labels."""
import hashlib,json,pathlib,sys,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import pyarrow.parquet as pq
from tokenizers import Tokenizer
REV='b08601e04326c79dfdd32d625aee71d232d685c3'
REPO='Salesforce/wikitext'
def sha(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':
    out=ROOT/'data';out.mkdir(exist_ok=False)
    tokpath=ROOT.parents[1]/'models/qwen3.8-27b-4bit-tokenizer-only/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokpath));meta={'repo':REPO,'revision':REV,'tokenizer_sha256':sha(tokpath.read_bytes()),'sequence_length':256,'add_special_tokens':False,'splits':{}}
    for split,n in [('train',3072),('test',6144)]:
        name=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet';url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}'
        with urllib.request.urlopen(url,timeout=120) as src:raw=src.read()
        path=out/(split+'.parquet');path.write_bytes(raw)
        texts=pq.read_table(path,columns=['text'])['text'].to_pylist()
        text='\n\n'.join(texts);ids=tokenizer.encode(text,add_special_tokens=False).ids[:n]
        assert len(ids)==n
        record={'split':split,'tokens':ids,'sequence_length':256,'source_revision':REV,'source_parquet_sha256':sha(raw),'selected_token_sha256':sha(json.dumps(ids,separators=(',',':')).encode()),'all_text_sha256':sha(text.encode()),'selected_token_offset':0,'source_text_rows':len(texts),'requested_tokens':n}
        (out/(split+'.json')).write_text(json.dumps(record))
        meta['splits'][split]={k:v for k,v in record.items() if k!='tokens'}
    assert meta['splits']['train']['source_parquet_sha256']!=meta['splits']['test']['source_parquet_sha256']
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
