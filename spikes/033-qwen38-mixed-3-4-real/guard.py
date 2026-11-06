"""Own a single child with normal/warning pressure and 3GiB RSS gates."""
import argparse,json,os,pathlib,subprocess,time,sys

def pressure():
    r=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,timeout=5,check=True)
    value=int(r.stdout.strip())
    if value not in (1,2):raise RuntimeError(f'pressure not normal/warning: {value}')
    return value

def preflight():
    level=pressure()
    r=subprocess.run(['memory_pressure'],capture_output=True,text=True,timeout=5,check=True)
    return {'time':time.time(),'pressure':level,'memory_pressure':r.stdout}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',required=True);ap.add_argument('--seconds',type=int,default=1800);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    dest=pathlib.Path(a.receipt);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:
        pre=preflight();f.write(json.dumps({'event':'preflight',**pre})+'\n');f.flush()
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1'}
        command=a.command[1:] if a.command[0]=='--' else a.command
        child=subprocess.Popen(command,env=env)
        start=time.monotonic();last=start;failure=None;peak=0
        try:
            while child.poll() is None:
                now=time.monotonic()
                if now-start>a.seconds:raise RuntimeError('deadline exceeded')
                if now-last>8:raise RuntimeError('observation gap exceeded 8s')
                p=pressure();last=time.monotonic()
                r=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True,timeout=5)
                if r.returncode or not r.stdout.strip():
                    if child.poll() is not None:break
                    raise RuntimeError('missing RSS sample')
                rss=int(r.stdout.strip())*1024;peak=max(peak,rss)
                f.write(json.dumps({'event':'sample','time':time.time(),'pid':child.pid,'pressure':p,'rss_bytes':rss})+'\n');f.flush()
                if rss>=3*2**30:raise RuntimeError(f'RSS ceiling exceeded: {rss}')
                time.sleep(0.75)
        except BaseException as e:
            failure=repr(e)
            # Only signal the direct, still-unreaped child we created.
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        finally:
            rc=child.wait();post=None
            try:post=preflight()
            except Exception as e:failure=failure or repr(e)
            f.write(json.dumps({'event':'exit','time':time.time(),'returncode':rc,'failure':failure,'peak_rss_bytes':peak,'post':post,'child_reaped':True})+'\n')
            print(json.dumps({'returncode':rc,'failure':failure,'peak_rss_bytes':peak}),flush=True)
        sys.exit(1 if failure else rc)
