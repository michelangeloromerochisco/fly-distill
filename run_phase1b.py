#!/usr/bin/env python3
"""Phase 1b runner: KD arms (fly/shuffled/er @ 40M, alpha 0.5, T=2 teacher temp).

Modes:
  --wait-for-report 1 : poll until report_phase1.txt exists (Phase 1 finished),
                        then start. Never starts while phase-1 procs are alive.
  --wait-for-report 0 : start immediately (manual use only).
Idempotent (skips completed jobs), ckpt auto-resume, 23h deadline guard,
single-instance lock, marker _phase1b_all_done, stop file _phase1b_stopped.
"""
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DEADLINE_H = 23.0
PROGRESS = "_phase1b_progress.json"
MARKER_DONE = "_phase1b_all_done"
STOP_FILE = "_phase1b_stopped"
LOCK = "_phase1b_runner.lock"
REPORT1 = "report_phase1.txt"


def jobs():
    J = []
    # prio 0: KD @ 20M — same budget as the Phase 1 baselines, so the
    # KD-vs-hard and gap-change comparisons are exact (seed 0).
    for c in ("fly", "shuffled", "er"):
        J.append(dict(condition=c, prio=0, train_bytes=20_000_000, tag="kd20"))
    # prio 1: the long 40M distillation run ("way longer" arm).
    for c in ("fly", "shuffled", "er"):
        J.append(dict(condition=c, prio=1, train_bytes=40_000_000, tag="kd40"))
    # prio 2: gentle KD (alpha 0.25, temp 1.0) — alpha=0.5/tau=2 hurt the
    # student (kd20 ~ +0.2 bpb vs hard-only); capacity-gap correction.
    for c in ("fly", "shuffled", "er"):
        J.append(dict(condition=c, prio=2, train_bytes=20_000_000, tag="kd20g",
                      kd_alpha=0.25, kd_temp=1.0))
    return J


def defaults(j):
    return dict(seed=0, T=4, width=0, train_bytes=j.get("train_bytes", 40_000_000),
                tag=j.get("tag", "kd40"), batch=64, lr=3e-3,
                kd_alpha=0.5, kd_temp=2.0, ckpt_every=5000)


def result_file(j):
    tag = j.get("tag", "kd40")
    return f"result_{tag}_{j['condition']}.json" if tag else f"result_{j['condition']}.json"


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
        if (e.get("seed", d.get("seed")) == j.get("seed", 0)
                and (e.get("train_bytes") or d.get("train_bytes")) == j.get("train_bytes", 40_000_000)
                and e.get("T", 4) == j.get("T", 4)
                and e.get("width", 0) == j.get("width", 0)
                and e.get("kd_alpha", 0.5) == j.get("kd_alpha", 0.5)):
            return True
    return False


def phase1_active():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
          "| ForEach-Object { \"$($_.CommandLine)\" }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return True  # can't see -> assume active, do not risk collision
    for l in r.stdout.splitlines():
        if "flylm.py --condition" in l or "run_phase1.py" in l or "teacher.py" in l:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-report", type=int, default=1)
    args = ap.parse_args()

    lock = open(LOCK, "w")
    lock.write("0")
    lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another run_phase1b holds the lock; exiting", flush=True)
        return
    lock.seek(0); lock.write(f"{os.getpid():<10}"); lock.flush()

    if args.wait_for_report:
        print("waiting for report_phase1.txt ...", flush=True)
        t_wait0 = time.time()
        while not os.path.exists(REPORT1):
            if (time.time() - t_wait0) / 3600 > 26:
                print("26h wait timeout; proceeding (phase1 likely finished)", flush=True)
                break
            time.sleep(300)
        while phase1_active():
            print("phase1 procs still alive; holding KD launch", flush=True)
            time.sleep(300)

    st = {"t0": time.time(), "done": [], "failed": [], "current": None}
    if os.path.exists(PROGRESS):
        try:
            old = json.load(open(PROGRESS))
            if isinstance(old, dict) and old.get("t0"):
                st = old; st.setdefault("done", []); st.setdefault("failed", [])
        except Exception:
            pass
    # deadline clock starts when JOBS actually begin, not during the report wait
    t0 = time.time()
    st["t0"] = t0
    logf = open("_phase1b_pipeline.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    py = sys.executable

    for j in sorted(jobs(), key=lambda x: x["prio"]):
        if done(j):
            continue
        lbl = f"{j.get('tag','kd40')}:{j['condition']}"
        if st.get("failed", []).count(lbl) >= 4:
            say(f"skip {lbl}: too many prior failures")
            continue
        if (time.time() - t0) / 3600 + 0.6 > DEADLINE_H:
            say(f"deadline guard -> not starting {lbl}")
            if not os.path.exists(STOP_FILE):
                open(STOP_FILE, "w").write(time.strftime("%m-%d %H:%M:%S"))
            break
        jlog = f"_phase1b_job_{lbl.replace(':', '_')}.log"
        cmd = [py, "kd_flylm.py", "--condition", j["condition"],
               "--train-bytes", str(j.get("train_bytes", 40_000_000)),
               "--tag", j.get("tag", "kd40"), "--seed", str(j.get("seed", 0)),
               "--kd-alpha", str(j.get("kd_alpha", 0.5)),
               "--kd-temp", str(j.get("kd_temp", 2.0)),
               "--ckpt-every", "5000"]
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
        if not st.get("done") and len(st.get("failed", [])) >= 12:
            say(">=12 failures with zero completions -> STOP file")
            open(STOP_FILE, "w").write(time.strftime("%m-%d %H:%M:%S"))
            break

    if all(done(j) for j in jobs()):
        open(MARKER_DONE, "w").write(time.strftime("%m-%d %H:%M:%S"))
        say("all phase1b jobs complete -> marker written")
    say("runner exit")


if __name__ == "__main__":
    main()