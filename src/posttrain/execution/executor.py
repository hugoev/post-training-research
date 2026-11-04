"""Run generated programs against test cases in isolated, resource-limited subprocesses.

Threat model: this guards against accidental damage and the common reward hacks seen
in RL on code (early exit, printing fake results, infinite loops, memory blowups,
deleting files). It is not a security boundary against deliberately adversarial code;
run inside a container or gVisor for that.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

_RUNNER = Path(__file__).with_name("_runner.py")


class Status(StrEnum):
    PASSED = "passed"
    FAILED = "failed"  # ran to completion but at least one assertion failed
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    CRASHED = "crashed"  # process died without reporting completion


@dataclass
class ExecutionResult:
    status: Status
    tests_passed: int
    tests_total: int
    test_results: list[bool] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status is Status.PASSED

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0

    @property
    def compiled(self) -> bool:
        return self.status is not Status.SYNTAX_ERROR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionResult:
        return cls(**{**d, "status": Status(d["status"])})


@dataclass
class ExecutionJob:
    solution: str
    tests: list[str]
    setup: str = ""
    test_prelude: str = ""


class Executor:
    def __init__(
        self,
        timeout_s: float = 20.0,
        per_test_timeout_s: float = 5.0,
        memory_mb: int = 2048,
        max_workers: int | None = None,
    ):
        self.timeout_s = timeout_s
        self.per_test_timeout_s = per_test_timeout_s
        self.memory_mb = memory_mb
        self.max_workers = max_workers or os.cpu_count() or 4

    @classmethod
    def from_config(cls, cfg) -> Executor:
        return cls(cfg.timeout_s, cfg.per_test_timeout_s, cfg.memory_mb, cfg.max_workers)

    def run(
        self, solution: str, tests: list[str], setup: str = "", test_prelude: str = ""
    ) -> ExecutionResult:
        """Run `setup`, then `solution`, then `test_prelude`, then each test snippet.

        All share one namespace, which is also importable as the module `solution`.
        """
        if not tests:
            raise ValueError("Refusing to run without tests: an empty suite would pass anything")
        nonce = secrets.token_hex(16)
        payload = {
            "nonce": nonce,
            "solution": solution,
            "setup": setup,
            "test_prelude": test_prelude,
            "tests": list(tests),
            "per_test_timeout_s": self.per_test_timeout_s,
            "memory_mb": self.memory_mb,
        }
        with tempfile.TemporaryDirectory(prefix="posttrain-exec-") as tmp:
            payload_path = Path(tmp) / "payload.json"
            payload_path.write_text(json.dumps(payload))
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": tmp,
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            start = time.perf_counter()
            proc = subprocess.Popen(
                [sys.executable, "-I", str(_RUNNER), str(payload_path)],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc.pid)
                stdout, stderr = proc.communicate()
            finally:
                _kill_group(proc.pid)
            duration = time.perf_counter() - start
        return _parse(stdout, stderr, nonce, len(tests), timed_out, duration)

    def run_many(self, jobs: list[ExecutionJob]) -> list[ExecutionResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(
                pool.map(lambda j: self.run(j.solution, j.tests, j.setup, j.test_prelude), jobs)
            )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _parse(
    stdout: bytes, stderr: bytes, nonce: str, num_tests: int, timed_out: bool, duration: float
) -> ExecutionResult:
    results = [False] * num_tests
    errors: list[str | None] = [None] * num_tests
    status: Status | None = None
    error: str | None = None
    done = False

    for line in stdout.decode(errors="replace").splitlines():
        if not line.startswith(nonce):
            continue
        try:
            event = json.loads(line[len(nonce):])
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "syntax_error":
            status, error = Status.SYNTAX_ERROR, event.get("error")
        elif kind == "load_error":
            error = event.get("error")
            status = Status.TIMEOUT if error == "Timeout" else Status.RUNTIME_ERROR
        elif kind == "test":
            index = event["index"]
            results[index] = bool(event["passed"])
            errors[index] = event.get("error")
        elif kind == "done":
            done = True

    passed = sum(results)
    if status is None:
        if not done:
            status = Status.TIMEOUT if timed_out else Status.CRASHED
            error = "Timeout" if timed_out else _tail(stderr)
        elif passed == num_tests:
            status = Status.PASSED
        else:
            error = next(e for r, e in zip(results, errors, strict=True) if not r)
            if error == "Timeout":
                status = Status.TIMEOUT
            elif error and error.startswith("AssertionError"):
                status = Status.FAILED
            else:
                status = Status.RUNTIME_ERROR
    return ExecutionResult(status, passed, num_tests, results, error, duration)


def _tail(stderr: bytes, limit: int = 500) -> str | None:
    text = stderr.decode(errors="replace").strip()
    return text[-limit:] or None
