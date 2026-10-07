#!/usr/bin/env python3
"""Phase 1 watchdog: keeps the 24h experiment alive, then writes the final report.

Liveness: python processes whose command line contains 'flylm.py --condition'
or 'run_phase1.py'. If nothing is active and work remains, relaunch
run_phase1.py (idempotent; flylm auto-resumes from checkpoints).
The runner stops launching new jobs at 23h. At 24h (+45 min grace for a
running job) kill stragglers, run report_phase1.py -> report_phase1.txt, exit.
"""
import os
import subprocess
import sys
import time

MAX_HOURS = 24.0
GRACE_MIN = 45
LOG = "watchdog_phase1.log"
STOP_FILE = "_phase1_stopped"
MARKER_DONE = "_phase1_all_done"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def proc_list():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
          "| ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if "|" in line:
            pid, _, cl = line.partition("|")
            out.append((pid.strip(), cl.strip()))
    return out


def is_work(cl):
    return "flylm.py --condition" in cl or "run_phase1.py" in cl


def active_training():
    return any(is_work(cl) for _, cl in proc_list())


def kill_training():
    for pid, cl in proc_list():
        if is_work(cl):
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)


def pick_python():
    for c in (sys.executable, "python"):
        try:
            r = subprocess.run([c, "-c", "import torch"], capture_output=True, timeout=120)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return "python"


py = pick_python()
log(f"phase1 watchdog started (python={py})")
t0 = time.time()
finalized = False
while not finalized:
    el_h = (time.time() - t0) / 3600
    if el_h >= MAX_HOURS:
        log("24h budget reached")
        if active_training():
            log(f"grace: up to {GRACE_MIN} min for the active run to finish")
            g0 = time.time()
            while active_training() and time.time() - g0 < GRACE_MIN * 60:
                time.sleep(60)
            if active_training():
                log("grace expired -> killing active runs (checkpoints are current)")
                kill_training()
                time.sleep(20)
        finalized = True
        break
    if os.path.exists(MARKER_DONE):
        # grace: marker may appear while the runner is between jobs; only
        # finalize once nothing has been active for 30 min
        age = time.time() - os.path.getmtime(MARKER_DONE)
        if age > 1800 and not active_training():
            log("all experiments complete (marker idle 30min) -> finalizing")
            finalized = True
            break
    if not active_training():
        if os.path.exists(STOP_FILE):
            log("runner hit its deadline guard; waiting out the final window")
        else:
            log("no active training -> launching runner (idempotent, auto-resume)")
            try:
                subprocess.run([py, "run_phase1.py"], capture_output=True, text=True,
                               timeout=max(600, int((MAX_HOURS - el_h) * 3600) - 900))
            except subprocess.TimeoutExpired:
                log("runner timeout -> killing tree")
                kill_training()
            except Exception as e:
                log(f"runner launch error: {e}")
    time.sleep(240)

log("writing final report")
try:
    r = subprocess.run([py, "report_phase1.py"], capture_output=True, text=True,
                       timeout=300)
    body = r.stdout if r.returncode == 0 else (r.stdout + "\n--STDERR--\n"
                                               + r.stderr[-2000:])
except Exception as e:
    body = f"report_phase1.py failed to run: {e}"
with open("report_phase1.txt", "w") as f:
    f.write(body)
log("report_phase1.txt written; watchdog exiting")