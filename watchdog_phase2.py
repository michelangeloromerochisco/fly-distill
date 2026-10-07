#!/usr/bin/env python3
"""Phase 2 watchdog: keep run_phase2.py alive across failures and reboots.

- Relaunches the runner whenever it exits without _phase2_all_done (runner
  itself is idempotent, so relaunch is safe).
- 130h global cap, then writes report_phase2.txt and exits.
- Does NOT fight with a running job: only relaunches when no flylm.py proc
  is alive and the lock is free.
"""
import msvcrt
import os
import subprocess
import sys
import time

MAX_HOURS = 130.0
LOG = "watchdog_phase2.log"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OURS = ("flylm.py --condition", "run_phase2.py")

# single-instance lock (prevents duplicate watchdogs from Startup-folder
# autostart + manual launch racing each other)
_lk = open("_phase2_watchdog.lock", "w")
_lk.write("0"); _lk.flush(); _lk.seek(0)
try:
    msvcrt.locking(_lk.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    print("another watchdog_phase2 holds the lock; exiting", flush=True)
    sys.exit(0)


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def proc_lines():
    ps = ("Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*'}"
          " | ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return [l.strip() for l in r.stdout.splitlines() if "|" in l]


def proc_alive(tag):
    lines = proc_lines()
    if lines is None:
        return True
    return any(tag in l for l in lines)


py = sys.executable
log(f"phase2 watchdog started (python={py})")
t0 = time.time()

while time.time() - t0 < MAX_HOURS * 3600:
    if os.path.exists("_phase2_all_done"):
        log("phase2 marker present -> writing report and exiting")
        break
    # a job (or the runner) is actively working -> let it run
    if proc_alive("flylm.py --condition") or proc_alive("run_phase2.py"):
        time.sleep(240)
        continue
    # nothing alive: either work remains (relaunch) or nothing actionable
    if os.path.exists("_phase2_stopped") and not os.path.exists("_phase2_progress.json"):
        log("stop file present with no progress state -> exiting")
        break
    if os.path.exists("_phase2_stopped"):
        # deadline guard fired; clear it and continue unless all done
        st = {}
        try:
            import json
            st = json.load(open("_phase2_progress.json"))
        except Exception:
            pass
        done, failed = st.get("done", []), st.get("failed", [])
        # if every not-done job is failure-gated, treat as complete
        log("stop file present; work may remain after failure gating -> relaunching runner")
        os.remove("_phase2_stopped")
    log("relaunching run_phase2.py")
    subprocess.Popen([py, "run_phase2.py"],
                     stdout=open("_phase2_runner.log", "a"),
                     stderr=subprocess.STDOUT)
    time.sleep(240)

log("watchdog cap reached -> finalizing phase2")
if proc_alive("run_phase2.py") or proc_alive("flylm.py --condition"):
    for l in (proc_lines() or []):
        pid, _, cl = l.partition("|")
        if any(p in cl for p in OURS):
            subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"],
                           capture_output=True)
    time.sleep(10)
try:
    r = subprocess.run([py, "report_phase2.py"], capture_output=True, text=True,
                       timeout=300)
    body = r.stdout if r.returncode == 0 else (r.stdout + "\n--STDERR--\n"
                                               + r.stderr[-2000:])
except Exception as e:
    body = f"report_phase2.py failed: {e}"
with open("report_phase2.txt", "w") as f:
    f.write(body)
log("report_phase2.txt written; exiting")