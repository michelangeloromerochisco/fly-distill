#!/usr/bin/env python3
"""Phase 2 runner: max-scale arms — does capacity reverse the fly topology's
disadvantage?

Config (fixed for all jobs): 150M train bytes (7.5x Phase 1), T=8, width=1024
(the two strongest Phase-1 levers combined), batch 64, lr 3e-3 — identical
protocol to Phase 1 otherwise, so numbers stay comparable. Val region is
necessarily [150M, 150.15M) (held-out tail); Phase 1 used [20M, 20.15M) —
noted in the report.

Jobs: {fly, shuffled, er, fly-uniform} x seeds {0,1}; seed 0 first so the
seed-0 quartet (~60h) gives the first complete answer.

Idempotent (skips completed jobs), ckpt auto-resume (native in flylm.py,
ckpt every 5000 steps ~ every 2 min), 122h deadline guard (persists across
relaunches via _phase2_progress.json t0; to EXTEND past the deadline, delete
_phase2_stopped and _phase2_progress.json), single-instance lock, marker
_phase2_all_done written when no actionable work remains (done OR failure-
gated) — prevents the watchdog relaunch livelock seen in phase 1b.
"""
import json
import msvcrt
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DEADLINE_H = 122.0
JOB_EST_H = 16.0          # per-job time estimate (150M T=8 w1024 ~ 15h + val)
TRAIN_BYTES = 150_000_000
T = 8
WIDTH = 1024
TAG = "p2"
PROGRESS = "_phase2_progress.json"
MARKER_DONE = "_phase2_all_done"
STOP_FILE = "_phase2_stopped"
LOCK = "_phase2_runner.lock"
COND_ORDER = {"fly": 0, "shuffled": 1, "er": 2, "fly-uniform": 3}


def jobs():
    J = []
    for seed in (0, 1):
        for c in ("fly", "shuffled", "er", "fly-uniform"):
            J.append(dict(condition=c, seed=seed,
                          prio=seed * 10 + COND_ORDER[c]))
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


def failed_count(st, lbl):
    return st.get("failed", []).count(lbl)


def main():
    lock = open(LOCK, "w")
    lock.write("0")
    lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another run_phase2 holds the lock; exiting", flush=True)
        return
    lock.seek(0); lock.write(f"{os.getpid():<10}"); lock.flush()

    st = {"t0": time.time(), "done": [], "failed": [], "current": None}
    if os.path.exists(PROGRESS):
        try:
            old = json.load(open(PROGRESS))
            if isinstance(old, dict) and old.get("t0"):
                st = old
                st.setdefault("done", []); st.setdefault("failed", [])
        except Exception:
            pass
    # deadline clock PERSISTS across relaunches (reboot-safe)
    t0 = st.get("t0") or time.time()
    st["t0"] = t0
    logf = open("_phase2_pipeline.log", "a")

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
        if failed_count(st, lbl) >= 4:
            say(f"skip {lbl}: too many prior failures")
            continue
        if remaining_h() < JOB_EST_H:
            say(f"deadline guard ({remaining_h():.1f}h left < {JOB_EST_H}h est) "
                f"-> not starting {lbl}")
            if not os.path.exists(STOP_FILE):
                open(STOP_FILE, "w").write(time.strftime("%m-%d %H:%M:%S"))
            break
        jlog = f"_phase2_job_{TAG}_{j['condition']}_s{j['seed']}.log"
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

    # marker when NO actionable work remains (done or failure-gated) —
    # prevents watchdog relaunch livelock
    actionable = [f"{TAG}:{j['condition']}:s{j['seed']}" for j in jobs()
                  if not done(j) and failed_count(st, f"{TAG}:{j['condition']}:s{j['seed']}") < 4]
    if not actionable:
        open(MARKER_DONE, "w").write(time.strftime("%m-%d %H:%M:%S"))
        say("no actionable jobs remain -> _phase2_all_done written")
    say("runner exit")


if __name__ == "__main__":
    main()