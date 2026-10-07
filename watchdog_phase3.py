#!/usr/bin/env python3
"""Phase 3 watchdog: keep run_phase3.py alive across failures and reboots.
Same pattern as watchdog_phase2.py with phase-3 filenames and a 165h cap."""
import msvcrt
import ctypes
import os
import subprocess
import sys
import threading
import time

MAX_HOURS = 165.0
LOG = "watchdog_phase3.log"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OURS = ("flylm.py --condition", "run_phase3.py")

# keep the machine awake while phase 3 runs (process-scoped; does NOT change
# system power settings). ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000003.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _keep_awake_loop():
    """Refresh SetThreadExecutionState every 60s on a dedicated thread."""
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    while True:
        time.sleep(60)
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


threading.Thread(target=_keep_awake_loop, daemon=True).start()

_lk = open("_phase3_watchdog.lock", "w")
_lk.write("0"); _lk.flush(); _lk.seek(0)
try:
    msvcrt.locking(_lk.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    print("another watchdog_phase3 holds the lock; exiting", flush=True)
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
log(f"phase3 watchdog started (python={py})")
t0 = time.time()

while time.time() - t0 < MAX_HOURS * 3600:
    if os.path.exists("_phase3_all_done"):
        log("phase3 marker present -> writing report and exiting")
        break
    if proc_alive("flylm.py --condition") or proc_alive("run_phase3.py"):
        time.sleep(240)
        continue
    log("relaunching run_phase3.py")
    subprocess.Popen([py, "run_phase3.py"],
                     stdout=open("_phase3_runner.log", "a"),
                     stderr=subprocess.STDOUT)
    time.sleep(240)

log("watchdog cap reached -> finalizing phase3")
if proc_alive("run_phase3.py") or proc_alive("flylm.py --condition"):
    for l in (proc_lines() or []):
        pid, _, cl = l.partition("|")
        if any(p in cl for p in OURS):
            subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"],
                           capture_output=True)
    time.sleep(10)
try:
    r = subprocess.run([py, "report_phase3.py"], capture_output=True, text=True,
                       timeout=300)
    body = r.stdout if r.returncode == 0 else (r.stdout + "\n--STDERR--\n"
                                               + r.stderr[-2000:])
except Exception as e:
    body = f"report_phase3.py failed: {e}"
with open("report_phase3.txt", "w") as f:
    f.write(body)
log("report_phase3.txt written; exiting")