#!/usr/bin/env python3
"""Phase 3 runner: seeds 2-3 of the quartet (n=4 for the primary endpoint)
plus the fly-hubcap mechanism arm (cap=40, seeds 0-1). All at the Phase 2
config: 150M bytes, T=8, w1024, batch 64, lr 3e-3. Tag "p3".

Idempotent, ckpt auto-resume, persistent deadline (150h), failure gate (4
strikes), livelock-proof marker. Seeds 0-1 cells of the quartet already
exist in result_p2_*.json; the report pools them.
"""
import json
import msvcrt
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DEADLINE_H = 220.0  # extended 23/09: sleep episodes (now fixed) burned ~20h
JOB_EST_H = 16.0
TRAIN_BYTES = 150_000_000
T = 8
WIDTH = 1024
TAG = "p3"
PROGRESS = "_phase3_progress.json"
MARKER_DONE = "_phase3_all_done"
LOCK = "_phase3_runner.lock"
COND_ORDER = {"fly": 0, "shuffled": 1, "fly-hubcap": 2, "er": 3, "fly-uniform": 4}


def jobs():
    J = []
    # quartet seeds 2,3 (prio 0-7)
    for seed in (2, 3):
        for c in ("fly", "shuffled", "er", "fly-uniform"):
            J.append(dict(condition=c, seed=seed, prio=seed * 10 + COND_ORDER[c]))
    # hubcap mechanism arm seeds 0,1 (prio 40-41: after the quartet seeds)
    for seed in (0, 1):
        J.append(dict(condition="fly-hubcap", seed=seed, prio=40 + seed))
    return sorted(J, key=lambda x: x["prio"])


def result_file(j):
    return f"result_{TAG}_{j['condition']}.json"


def done(j):
    f = result_file(j)
    if not os.path.exists(f):
        return False
    try:
        d = json.load(open(f))
    except Exception:
        return False
    entries = [d] + list(d.get("runs", []))
    for e in entries:
        if (e.get("seed", d.get("seed")) == j["seed"]
                and (e.get("train_bytes") or d.get("train_bytes")) == TRAIN_BYTES
                and e.get("T", 4) == T and e.get("width", 0) == WIDTH):
            return True
    return False


def main():
    lock = open(LOCK, "w")
    lock.write("0"); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another run_phase3 holds the lock; exiting", flush=True)
        return
    lock.seek(0); lock.write(f"{os.getpid():<10}"); lock.flush()

    st = {"t0": time.time(), "done": [], "failed": [], "current": None}
    if os.path.exists(PROGRESS):
        try:
            old = json.load(open(PROGRESS))
            if isinstance(old, dict) and old.get("t0"):
                st = old; st.setdefault("done", []); st.setdefault("failed", [])
        except Exception:
            pass
    t0 = st.get("t0") or time.time()
    st["t0"] = t0
    logf = open("_phase3_pipeline.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    py = sys.executable
    remaining_h = lambda: DEADLINE_H - (time.time() - t0) / 3600.0

    for j in jobs():
        if done(j):
            continue
        lbl = f"{TAG}:{j['condition']}:s{j['seed']}"
        if st.get("failed", []).count(lbl) >= 4:
            say(f"skip {lbl}: too many prior failures")
            continue
        if remaining_h() < JOB_EST_H:
            say(f"deadline guard ({remaining_h():.1f}h left) -> not starting {lbl}")
            break
        jlog = f"_phase3_job_{TAG}_{j['condition']}_s{j['seed']}.log"
        cmd = [py, "flylm.py", "--condition", j["condition"],
               "--train-bytes", str(TRAIN_BYTES), "--T", str(T),
               "--width", str(WIDTH), "--tag", TAG,
               "--seed", str(j["seed"]), "--batch", "64", "--lr", "3e-3",
               "--val-bytes", "150000", "--ckpt-every", "5000"]
        for attempt in (1, 2):
            say(f"=== start {lbl} attempt {attempt}")
            st["current"] = {"lbl": lbl, "started": time.strftime("%m-%d %H:%M:%S")}
            json.dump(st, open(PROGRESS, "w"), indent=1)
            with open(jlog, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=env).returncode
            tail = ""
            try:
                lines = open(jlog, errors="replace").read().splitlines()
                tail = " | ".join([l for l in lines
                                   if l.startswith(("RESULT", "resumed"))][-2:])
            except Exception:
                pass
            if rc == 0:
                say(f"finished {lbl}: {tail}")
                st.setdefault("done", []).append(lbl)
                break
            say(f"FAILED {lbl} rc={rc} tail={tail}")
            st.setdefault("failed", []).append(lbl)
            time.sleep(30)
        st["current"] = None
        st["last_update"] = time.strftime("%m-%d %H:%M:%S")
        json.dump(st, open(PROGRESS, "w"), indent=1)

    actionable = [f"{TAG}:{j['condition']}:s{j['seed']}" for j in jobs()
                  if not done(j) and st.get("failed", []).count(f"{TAG}:{j['condition']}:s{j['seed']}") < 4]
    if not actionable:
        open(MARKER_DONE, "w").write(time.strftime("%m-%d %H:%M:%S"))
        say("no actionable jobs remain -> _phase3_all_done written")
    say("runner exit")


if __name__ == "__main__":
    main()