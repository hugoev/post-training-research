"""Subprocess entry point that runs untrusted solution code against test snippets.

Invoked by executor.py as `python -I _runner.py <payload.json>`. Must stay
self-contained (stdlib only, no package imports).

Protocol: every event is written as one line `<nonce><json>` to the original stdout.
The nonce is read from the payload file, which is deleted before untrusted code runs,
so solution output (which is captured, and fd 1 redirected to /dev/null) cannot
forge results. A result only counts if the final `done` event is emitted, so
`sys.exit(0)` or `os._exit(0)` before the asserts never registers as a pass.
"""

import builtins
import contextlib
import io
import json
import os
import resource
import signal
import sys
import types


class _TestTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _TestTimeout()


def _set_limits(memory_mb):
    if memory_mb:
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _disable_destructive_functions():
    import shutil
    import subprocess

    for name in (
        "kill", "killpg", "system", "remove", "removedirs", "rmdir", "unlink",
        "rename", "renames", "truncate", "replace", "fork", "forkpty", "setuid", "chmod",
        "chown", "execv", "execve", "execvp", "spawnv", "popen",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    subprocess.Popen = None


def _describe(exc):
    if isinstance(exc, _TestTimeout):
        return "Timeout"
    message = str(exc)
    if len(message) > 300:
        message = message[:300] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def main():
    payload_path = sys.argv[1]
    with open(payload_path) as f:
        payload = json.load(f)
    os.unlink(payload_path)

    nonce = payload["nonce"]
    out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)

    def emit(event):
        out.write(nonce + json.dumps(event) + "\n")
        out.flush()

    per_test_timeout = float(payload["per_test_timeout_s"])
    signal.signal(signal.SIGALRM, _on_alarm)
    _set_limits(payload.get("memory_mb"))

    def run(code, filename, namespace):
        compiled = compile(code, filename, "exec")
        captured = io.StringIO()
        signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                exec(compiled, namespace)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    emit({"event": "start"})
    try:
        compile(payload["solution"], "<solution>", "exec")
    except (SyntaxError, ValueError) as exc:
        emit({"event": "syntax_error", "error": _describe(exc)})
        emit({"event": "done"})
        return

    _disable_destructive_functions()
    # The solution lives in a real module named `solution` so pytest-style test files
    # can `from solution import f`; it is not named __main__, so demo blocks do not run.
    module = types.ModuleType("solution")
    sys.modules["solution"] = module
    namespace = module.__dict__
    namespace["__builtins__"] = builtins
    try:
        if payload.get("setup"):
            run(payload["setup"], "<setup>", namespace)
        run(payload["solution"], "<solution>", namespace)
        if payload.get("test_prelude"):
            run(payload["test_prelude"], "<test_prelude>", namespace)
    except BaseException as exc:
        emit({"event": "load_error", "error": _describe(exc)})
        emit({"event": "done"})
        return

    for index, test in enumerate(payload["tests"]):
        try:
            run(test, f"<test_{index}>", namespace)
            emit({"event": "test", "index": index, "passed": True})
        except BaseException as exc:
            emit({"event": "test", "index": index, "passed": False, "error": _describe(exc)})
    emit({"event": "done"})


if __name__ == "__main__":
    main()
