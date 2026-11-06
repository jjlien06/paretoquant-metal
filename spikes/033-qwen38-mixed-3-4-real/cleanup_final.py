"""Read-only end-of-trial ownership, pressure, disk, and chat checks."""
import json,os,pathlib,subprocess,sys,time

def call(argv):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=20)
    return {'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
if __name__=='__main__':
    rows=subprocess.run(['ps','-axo','pid=,ppid='],capture_output=True,text=True,check=True).stdout.splitlines()
    parents={int(r.split()[0]):int(r.split()[1]) for r in rows}
    ancestors=set();pid=os.getpid()
    while pid and pid not in ancestors:ancestors.add(pid);pid=parents.get(pid,0)
    probe=call(['pgrep','-fl',r'033-qwen38-mixed-3-4-real|(^|[ /])(guard|stream(_v2)?|download_q3|allocate_real|build_mixed|finish_stages|assure(_v2)?)\.py([[:space:]]|$)'])
    if probe["returncode"] not in (0,1):raise RuntimeError("pgrep failed")
    remaining=[]
    for row in probe['stdout'].splitlines():
        pid=int(row.split(None,1)[0])
        if pid not in ancestors:remaining.append(row)
    result={'time':time.time(),'pgrep':probe,'excluded_current_and_ancestor_pids':sorted(ancestors),'remaining_experiment_processes':remaining,'passed':not remaining,'pressure':call(['sysctl','kern.memorystatus_vm_pressure_level','vm.swapusage']),'memory_pressure':call(['memory_pressure']),'disk':call(['df','-h','/System/Volumes/Data']),'chat_process_readback':call(['pgrep','-fl','chat_rank.py']),'chat_listener_readback':call(['lsof','-nP','-iTCP:8780','-sTCP:LISTEN'])}
    with pathlib.Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
