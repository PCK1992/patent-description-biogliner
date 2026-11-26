"""Share one allocated GPU between independent workers; keep score.py unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def workers_for(gpu, cpus):
    if "MIG" in gpu or cpus < 2:
        return 1
    if "A100" in gpu or "H200" in gpu:
        return 3 if cpus >= 4 else 2
    return 2 if gpu == "NVIDIA A40" else 1


def text_receipt(start, end):
    return Path("texts_complete.json" if (start,end)==(1976,2024) else f"texts_years_{start}_{end}.json")


def score_shard(year, shard, n, worker_script=Path(__file__).resolve()):
    # An exec starts each worker with a clean CUDA runtime. Keep the parent's lock
    # open in each worker so that a killed launcher cannot permit a competing retry.
    with open(f"logs/score_{year}_{shard}_8_False.lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd=lock.fileno()
        cmds=[[sys.executable,"-u",str(worker_script),"--part",str(fd),str(year),str(part),str(8*n)]
              for part in range(shard,8*n,8)]
        jobs = []
        try:
            for cmd in cmds:
                jobs.append(subprocess.Popen(cmd,pass_fds=(fd,)))
            failed=[i for i,job in enumerate(jobs) if job.wait()!=0]
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.terminate()
                    job.wait()
        for i in failed:
            print(f"Retrying subshard {cmds[i][-2]} alone from its checkpoints", flush=True)
            subprocess.run(cmds[i],pass_fds=(fd,),check=True)


if __name__ == "__main__":
    if sys.argv[1:2]==["--part"]:
        fd,year,part,modulus=map(int,sys.argv[2:])
        assert modulus in (8,16,24) and 0<=part<modulus
        assert os.path.samestat(os.fstat(fd),os.stat(f"logs/score_{year}_{part%8}_8_False.lock"))
        fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        from score import main
        main(year,part,modulus,False)
        sys.exit(0)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",type=int)
    p.add_argument("shards",help="inclusive original task range")
    args=p.parse_args()
    first,last=map(int,args.shards.split("-"))
    assert 0 <= first <= args.task <= last <= 391
    start,end=1976+first//8,1976+last//8
    coverage=json.loads(text_receipt(start,end).read_text())
    roster=json.loads(Path("original_roster_audit.json").read_text())
    pilot=json.loads(Path("pilot/1976/PASS.json").read_text())
    assert roster["passed"] and roster["original_unique"]==1403822
    assert pilot["n"]>=6 and pilot["signature"]["pilot"] and pilot["signature"]["source"]=="1976.parquet"
    assert coverage["nonempty"]>0
    if (start,end)==(1976,2024):
        assert coverage["roster"]==roster["original_unique"]
    else:
        assert coverage["grant_years"]==[start,end] and coverage["roster_sha256"]==roster["key_sha256"]
    # The launcher does not need its own persistent CUDA context.
    gpu=subprocess.check_output([sys.executable,"-c","import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"], text=True).strip()
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    n=workers_for(gpu,cpus)
    if n>1:
        os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    print(f"Task {args.task}: {gpu}, {cpus} CPUs, {n} workers",flush=True)
    score_shard(1976+args.task//8,args.task%8,n)
