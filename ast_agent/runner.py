"""Run a task's test command inside a working copy of the repository."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class TestResult:
    passed: bool
    returncode: int
    output: str          # tail of combined stdout/stderr
    duration_s: float
    timed_out: bool = False


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the shell *and* everything it started (e.g. a pytest stuck in an infinite loop)."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_tests(cmd: str, cwd: str, timeout: int = 300, tail_chars: int = 4000) -> TestResult:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = cwd + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Use the same interpreter that runs the agent, so `python -m pytest` works in venvs/notebooks.
    if cmd.startswith("python "):
        cmd = f'"{sys.executable}" ' + cmd[len("python "):]
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            start_new_session=(os.name == "posix"))
    try:
        out, _ = proc.communicate(timeout=timeout)
        return TestResult(proc.returncode == 0, proc.returncode, (out or "")[-tail_chars:], time.time() - t0)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out = ""
        return TestResult(False, -1, ((out or "") + "\n[timed out]")[-tail_chars:], time.time() - t0, True)
