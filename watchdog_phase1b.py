#!/usr/bin/env python3
"""Phase 1b watchdog: teacher (concurrent, small) -> probs cache -> KD runner.

Timeline:
  1. Teacher trains CONCURRENTLY with phase-1 (its ~0.5 GB footprint is safe
     next to phase-1's ~0.8-1.0 GB; ckpt auto-resume, relaunched if dead).
  2. When the teacher finishes, build the one-time probs cache (inference
     only, also concurrent-safe).
  3. Launch run_phase1b.py --wait-for-report 1: the runner itself holds KD
     TRAINING until phase-1 has freed the GPU (report / stop / 25h) — the
     OOM-heavy part never overlaps phase-1.
  4. 26h cap: kill stragglers, write report_phase1b.txt, exit.
"""
import json
import os
import subprocess
import sys
import time

MAX_HOURS = 47.0
LOG = "watchdog_phase1b.log"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OURS = ("flylm.py --condition", "kd_flylm.py --condition", "teacher.py",
        "probs_cache.py", "run_phase1.py", "run_phase1b.py")


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def proc_lines():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
          "| ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
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


def kill_ours():
    for l in (proc_lines() or []):
        pid, _, cl = l.partition("|")
        if any(p in cl for p in OURS):
            subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"],
                           capture_output=True)


def pick_python():
    for c in (sys.executable, "python"):
        try:
            r = subprocess.run([c, "-c", "import torch"], capture_output=True, timeout=120)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return "python"


def teacher_done():
    return os.path.exists("result_teacher.json")


def probs_ready():
    """Valid probs cache for 40M matching the CURRENT teacher ckpt mtime."""
    tp = "ckpt_teacher.pt"
    if not os.path.exists(tp) or not teacher_done():
        return False, None
    mt = os.path.getmtime(tp)
    for f in sorted(os.listdir(".")):
        if f.startswith("probs_") and f.endswith(".npy"):
            try:
                m = json.load(open(f + ".meta.json"))
                if (m.get("rows", 0) >= m.get("train_bytes", 0) - m.get("ctx", 512)
                        and m.get("train_bytes", 0) >= 40_000_000
                        and abs(m.get("teacher_mtime", -1) - mt) < 2):
                    return True, f
            except Exception:
                pass
    return False, None


def phase1_free(el_h):
    if os.path.exists("report_phase1.txt"):
        return True
    if os.path.exists("_phase1_stopped") and not proc_alive("flylm.py --condition"):
        return True
    return el_h > 25.0


py = pick_python()
log(f"phase1b watchdog started (python={py})")
t0 = time.time()
kd_launched = False

while time.time() - t0 < MAX_HOURS * 3600:
    el_h = (time.time() - t0) / 3600

    # all phase1b jobs done -> finalize immediately (no churn)
    if os.path.exists("_phase1b_all_done") and not proc_alive("kd_flylm.py") \
            and not proc_alive("run_phase1b.py"):
        log("phase1b marker present -> finalizing")
        break

    # --- stage 1: teacher (concurrent with phase-1) ---
    if not teacher_done():
        if not proc_alive("teacher.py"):
            log("launching teacher (concurrent with phase-1)")
            subprocess.Popen([py, "teacher.py"],
                             stdout=open("_phase1b_teacher.log", "a"),
                             stderr=subprocess.STDOUT)
        time.sleep(240)
        continue

    # --- stage 2: one-time probs cache (inference only, concurrent-safe) ---
    ok, pfile = probs_ready()
    if not ok:
        if not proc_alive("probs_cache.py"):
            log("building teacher-probs cache (one-time)")
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            subprocess.Popen([py, "probs_cache.py", "--train-bytes", "40000000",
                              "--g", "16"],
                             stdout=open("_phase1b_probs.log", "a"),
                             stderr=subprocess.STDOUT, env=env)
        time.sleep(240)
        continue
    log(f"probs cache ready: {pfile}")

    # --- stage 3: KD runner (runner itself gates KD training on phase-1 free) ---
    if not kd_launched:
        log("teacher+probs ready -> launching report-gated KD runner")
        subprocess.Popen([py, "run_phase1b.py", "--wait-for-report", "1"],
                         stdout=open("_phase1b_runner.log", "w"),
                         stderr=subprocess.STDOUT)
        kd_launched = True
    if phase1_free(el_h) and not proc_alive("kd_flylm.py --condition") \
            and not proc_alive("run_phase1b.py"):
        log("KD runner died with work remaining -> relaunching")
        subprocess.Popen([py, "run_phase1b.py", "--wait-for-report", "1"],
                         stdout=open("_phase1b_runner.log", "a"),
                         stderr=subprocess.STDOUT)
    time.sleep(240)

log("26h cap reached -> finalizing phase1b")
kill_ours()
time.sleep(10)
try:
    r = subprocess.run([py, "report_phase1b.py"], capture_output=True, text=True,
                       timeout=300)
    body = r.stdout if r.returncode == 0 else (r.stdout + "\n--STDERR--\n"
                                               + r.stderr[-2000:])
except Exception as e:
    body = f"report_phase1b.py failed: {e}"
with open("report_phase1b.txt", "w") as f:
    f.write(body)
log("report_phase1b.txt written; exiting")