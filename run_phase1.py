#!/usr/bin/env python3
"""Phase 1 overnight pipeline: multi-seed error bars + capacity arms (24h budget).

All jobs: 20M train bytes, val 150k, batch 64, lr 3e-3, TinyStories slice.
  P0 multi-seed : seeds {1,2} x {fly, shuffled, er}, T=4          -> error bars
  P1 t8         : seed 0 x {fly, shuffled, er}, T=8, tag "t8"     -> temporal depth
  P2 w1024      : seed 0 x {fly, shuffled, er}, width=1024, "w1024" -> readout capacity
  P3 matrix     : fly-uniform seed 0 @20M (completes 4-cond grid)
  P4-P6 fill    : fly-uniform seeds {1,2}; t8/w1024 fly-uniform; seeds {3,4}

Idempotent: a job is skipped if its (tag, condition, train_bytes, T, width, seed)
entry already exists in the result JSON. flylm.py auto-resumes from checkpoints.
Deadline-aware: stops launching new jobs after DEADLINE_H hours (leaves the
final window for the watchdog to write the report). Writes _phase1_all_done
once P0-P3 are complete. Single-instance via lock file.
"""
import json
import msvcrt
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DEADLINE_H = 23.0
VAL_BYTES = 150_000
PROGRESS = "_phase1_progress.json"
MARKER_DONE = "_phase1_all_done"
STOP_FILE = "_phase1_stopped"
LOCK = "_phase1_runner.lock"


def jobs():
    J = []

    def add(c, seed, T, width, tag, prio, ckpt_every):
        J.append(dict(condition=c, seed=seed, T=T, width=width, tag=tag, prio=prio,
                      train_bytes=20_000_000, batch=64, lr=3e-3,
                      ckpt_every=ckpt_every))

    for seed in (1, 2):
        for c in ("fly", "shuffled", "er"):
            add(c, seed, 4, 0, "", 0, 10_000)                    # P0
    for c in ("fly", "shuffled", "er"):
        add(c, 0, 8, 0, "t8", 1, 2500)                           # P1
    for c in ("fly", "shuffled", "er"):
        add(c, 0, 4, 1024, "w1024", 2, 10_000)                   # P2
    add("fly-uniform", 0, 4, 0, "", 3, 10_000)                   # P3
    for c in ("fly", "shuffled", "er", "fly-uniform"):           # P4 fill
        add(c, 1, 4, 0, "", 4, 10_000)
        add(c, 2, 4, 0, "", 4, 10_000)
    add("fly-uniform", 0, 8, 0, "t8", 5, 2500)                   # P5 fill
    add("fly-uniform", 0, 4, 1024, "w1024", 5, 10_000)
    for seed in (3, 4):                                          # P6 fill
        for c in ("fly", "shuffled", "er"):
            add(c, seed, 4, 0, "", 6, 10_000)
    return J


def result_file(j):
    if j["tag"]:
        return f"result_{j['tag']}_{j['condition']}.json"
    return f"result_{j['condition']}.json"


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
                and (e.get("train_bytes") or d.get("train_bytes")) == j["train_bytes"]
                and e.get("T", 4) == j["T"]
                and e.get("width", 0) == j["width"]):
            return True
    return False


def main():
    lock = open(LOCK, "w")
    lock.write("0")
    lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another run_phase1 holds the lock; exiting", flush=True)
        return
    lock.seek(0)
    lock.write(f"{os.getpid():<10}")   # overwrite in place; no truncate while locked
    lock.flush()

    st = {"t0": time.time(), "done": [], "failed": [], "current": None}
    if os.path.exists(PROGRESS):
        try:
            old = json.load(open(PROGRESS))
            if isinstance(old, dict) and old.get("t0"):
                st = old
                st.setdefault("done", [])
                st.setdefault("failed", [])
        except Exception:
            pass
    t0 = st["t0"]

    logf = open("_phase1_pipeline.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    py = sys.executable
    js = sorted(jobs(), key=lambda x: x["prio"])

    for j in js:
        if done(j):
            continue
        lbl = f"{j['tag'] or 'base'}:{j['condition']}:s{j['seed']}"
        if st.get("failed", []).count(lbl) >= 4:
            say(f"skip {lbl}: too many prior failures")
            continue
        el_h = (time.time() - t0) / 3600
        if el_h + 0.6 > DEADLINE_H:
            say(f"deadline guard ({el_h:.2f}h elapsed) -> not starting {lbl}")
            if not os.path.exists(STOP_FILE):
                open(STOP_FILE, "w").write(time.strftime("%m-%d %H:%M:%S"))
            break
        jlog = f"_phase1_job_{lbl.replace(':', '_')}.log"
        cmd = [py, "flylm.py", "--condition", j["condition"],
               "--train-bytes", str(j["train_bytes"]), "--val-bytes", str(VAL_BYTES),
               "--batch", str(j["batch"]), "--T", str(j["T"]), "--lr", str(j["lr"]),
               "--seed", str(j["seed"]), "--width", str(j["width"]),
               "--tag", j["tag"], "--ckpt-every", str(j["ckpt_every"])]
        for attempt in (1, 2):
            say(f"=== start {lbl} attempt {attempt} "
                f"(T={j['T']} w={j['width']} {j['train_bytes']:,}B)")
            st["current"] = {"lbl": lbl, "started": time.strftime("%m-%d %H:%M:%S")}
            json.dump(st, open(PROGRESS, "w"), indent=1)
            with open(jlog, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=env).returncode
            tail = ""
            try:
                lines = open(jlog, errors="replace").read().splitlines()
                keep = [l for l in lines if l.startswith(("RESULT", "resumed"))]
                tail = " | ".join(keep[-2:])
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
        if not st.get("done") and len(st.get("failed", [])) >= 12:
            say(">=12 failures with zero completions -> writing STOP file")
            open(STOP_FILE, "w").write(time.strftime("%m-%d %H:%M:%S"))
            break

    all_done = all(done(j) for j in js)
    if all_done:
        open(MARKER_DONE, "w").write(time.strftime("%m-%d %H:%M:%S"))
        say("all jobs complete -> _phase1_all_done written")
    elif all(done(j) for j in js if j["prio"] <= 3):
        say("P0-P3 complete; fillers done/skipped -> runner exit")
    else:
        say("runner exit; core incomplete (deadline or failures)")
    say("runner exit")


if __name__ == "__main__":
    main()