#!/usr/bin/env python3
"""Overnight watchdog v2 (runner-independent).

Liveness rule: active work = any python process whose command line contains
'flylm.py --condition' or '_pipeline_all_done.script' (Win32_Process query).
If no active work AND pipeline incomplete -> relaunch tier pipeline
(flylm.py auto-resumes from checkpoints; lost work <= last checkpoint).
Exits when pipeline is complete or MAX_HOURS reached. Writes watchdog.log.
"""
import subprocess, sys, time, os, json, glob

MAX_HOURS = 12
MARKER = "_pipeline_all_done"
LOG = "watchdog.log"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def active_training():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
          "| Select-Object -ExpandProperty CommandLine")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    for line in r.stdout.splitlines():
        l = line.strip()
        if "flylm.py --condition" in l or MARKER + ".script" in l:
            return True
    return False

def pipeline_done():
    conds = ["fly", "shuffled", "er", "fly-uniform"]
    for c in conds:
        fs = glob.glob(f"result_{c}.json")
        if not fs:
            return False
        d = json.load(open(fs[0]))
        entries = [{"train_bytes": d.get("train_bytes"), "val_bpb": d["val_bpb"]}] + d.get("runs", [])
        if not any((e.get("train_bytes") or 0) >= 10_000_000 for e in entries):
            return False
    # Tier 2: 20M for the three decisive conditions
    for c in ["fly", "shuffled", "er"]:
        d = json.load(open(f"result_{c}.json"))
        entries = [{"train_bytes": d.get("train_bytes")}] + d.get("runs", [])
        if not any((e.get("train_bytes") or 0) >= 20_000_000 for e in entries):
            return False
    return True

def pick_python():
    """Prefer the interpreter that has torch; try current, then PATH python."""
    candidates = [sys.executable, "python"]
    for c in candidates:
        try:
            r = subprocess.run([c, "-c", "import torch"], capture_output=True, timeout=120)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return "python"

def run_pipeline(py):
    log("relaunching tier pipeline (auto-resume from checkpoints)")
    script = "\n".join([
        "import subprocess, sys, time",
        'COND = ["fly", "shuffled", "er", "fly-uniform"]',
        'TIER1 = ["--train-bytes", "10000000", "--val-bytes", "100000", "--batch", "64", "--T", "4", "--seed", "0", "--ckpt-every", "5000"]',
        'TIER2 = ["--train-bytes", "20000000", "--val-bytes", "150000", "--batch", "64", "--T", "4", "--seed", "0", "--ckpt-every", "10000"]',
        'for tier, cfg in (("T1", TIER1), ("T2", TIER2)):',
        '    conds = COND if tier == "T1" else ["fly", "shuffled", "er"]',
        '    for c in conds:',
        '        print(f"=== {tier} {c} start {time.strftime(\'%H:%M:%S\')}", flush=True)',
        '        r = subprocess.run([sys.executable, "flylm.py", "--condition", c] + cfg, capture_output=True, text=True)',
        '        tail = [l for l in r.stdout.splitlines() if l.startswith(("RESULT", "resumed", "  [ckpt]"))][-3:]',
        '        print("\\n".join(tail), flush=True)',
        '        if r.returncode != 0: print("STDERR:", r.stderr[-1500:], flush=True)',
        'print("ALL DONE", flush=True)',
    ])
    with open(MARKER + ".script", "w") as f:
        f.write(script)
    r = subprocess.run([py, MARKER + ".script"], capture_output=True, text=True)
    with open(MARKER + ".log", "w") as f:
        f.write(r.stdout + "\n--STDERR--\n" + r.stderr[-3000:])
    log("pipeline run returned; tail: " + " | ".join(r.stdout.splitlines()[-4:]))
    return "ALL DONE" in r.stdout

py = pick_python()
log(f"watchdog v2 started (python={py})")
t0 = time.time()
while time.time() - t0 < MAX_HOURS * 3600:
    if pipeline_done():
        log("pipeline complete -> exiting")
        sys.exit(0)
    if not active_training():
        log("no active training and work incomplete -> running pipeline")
        done = run_pipeline(py)
        if done:
            log("pipeline printed ALL DONE")
            continue  # loop will verify pipeline_done() and exit
    time.sleep(240)
log("max hours reached; exiting")