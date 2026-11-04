import pytest

from posttrain.execution import ExecutionJob, Executor, Status

TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(10, 5) == 15"]


@pytest.fixture(scope="module")
def executor():
    return Executor(timeout_s=10, per_test_timeout_s=1.0, memory_mb=512, max_workers=4)


def test_passing_solution(executor):
    r = executor.run("def add(a, b):\n    return a + b", TESTS)
    assert r.status is Status.PASSED
    assert r.tests_passed == 3 and r.pass_rate == 1.0
    assert r.test_results == [True, True, True]


def test_partial_failure_reports_pass_rate(executor):
    r = executor.run("def add(a, b):\n    return 3 if a == 1 else a + b + 1", TESTS)
    assert r.status is Status.FAILED
    assert r.test_results == [True, False, False]
    assert r.error.startswith("AssertionError")


def test_syntax_error(executor):
    r = executor.run("def add(a, b)\n    return a + b", TESTS)
    assert r.status is Status.SYNTAX_ERROR
    assert not r.compiled and r.tests_passed == 0


def test_error_while_loading_solution(executor):
    r = executor.run("raise RuntimeError('boom')", TESTS)
    assert r.status is Status.RUNTIME_ERROR
    assert "boom" in r.error


def test_exception_inside_tests_is_runtime_error(executor):
    r = executor.run("def add(a, b):\n    return a.missing", TESTS)
    assert r.status is Status.RUNTIME_ERROR
    assert r.error.startswith("AttributeError")


def test_infinite_loop_hits_per_test_timeout(executor):
    code = "def add(a, b):\n    while a == 10:\n        pass\n    return a + b"
    r = executor.run(code, TESTS)
    assert r.status is Status.TIMEOUT
    assert r.test_results == [True, True, False]


def test_infinite_loop_at_import_time(executor):
    r = executor.run("while True:\n    pass", TESTS)
    assert r.status is Status.TIMEOUT


def test_wall_clock_timeout_when_alarm_is_swallowed():
    executor = Executor(timeout_s=2, per_test_timeout_s=0.2, memory_mb=512)
    code = (
        "def add(a, b):\n"
        "    while True:\n"
        "        try:\n"
        "            while True: pass\n"
        "        except BaseException: pass\n"
    )
    r = executor.run(code, TESTS)
    assert r.status is Status.TIMEOUT
    assert r.tests_passed == 0


def test_memory_limit(executor):
    r = executor.run("x = bytearray(2 * 1024 ** 3)\ndef add(a, b): return a + b", TESTS)
    assert r.status is Status.RUNTIME_ERROR
    assert r.error.startswith("MemoryError")


@pytest.mark.parametrize(
    "hack",
    [
        "import sys\nsys.exit(0)",
        "raise SystemExit(0)",
        "exit()",
    ],
)
def test_early_exit_does_not_pass(executor, hack):
    r = executor.run(f"def add(a, b): return 0\n{hack}", TESTS)
    assert not r.passed
    assert r.tests_passed == 0


def test_hard_exit_is_crash(executor):
    r = executor.run("import os\nos._exit(0)", TESTS)
    assert r.status is Status.CRASHED
    assert r.tests_passed == 0


def test_forged_output_is_ignored(executor):
    forged = (
        "import os, json\n"
        "for i in range(3):\n"
        "    line = json.dumps({'event': 'test', 'index': i, 'passed': True})\n"
        "    print(line)\n"
        "    os.write(1, (line + '\\n').encode())\n"
        "print(json.dumps({'event': 'done'}))\n"
        "def add(a, b): return None\n"
    )
    r = executor.run(forged, TESTS)
    assert r.status is Status.FAILED
    assert r.tests_passed == 0


def test_cannot_read_payload_to_learn_nonce(executor):
    code = "import os\nassert not os.listdir('.')\ndef add(a, b): return a + b"
    assert executor.run(code, TESTS).passed


def test_destructive_functions_are_disabled(executor):
    r = executor.run("import shutil\nshutil.rmtree('/tmp/whatever')", TESTS)
    assert r.status is Status.RUNTIME_ERROR


def test_third_party_imports_still_work(executor):
    code = (
        "import numpy as np\nimport os\nos.environ['X'] = '1'\n"
        "def add(a, b): return int(np.add(a, b))"
    )
    assert executor.run(code, TESTS).passed


def test_main_guard_is_not_executed(executor):
    code = "def add(a, b): return a + b\nif __name__ == '__main__':\n    input()"
    assert executor.run(code, TESTS).passed


def test_setup_code_runs_before_solution(executor):
    r = executor.run("def f(xs: List[int]): return sum(xs)", ["assert f([1, 2]) == 3"],
                     setup="from typing import List")
    assert r.passed


def test_empty_test_suite_is_rejected(executor):
    with pytest.raises(ValueError):
        executor.run("def add(a, b): return a + b", [])


def test_run_many_preserves_order(executor):
    jobs = [
        ExecutionJob("def add(a, b): return a + b", TESTS),
        ExecutionJob("def add(a, b): return 0", TESTS),
        ExecutionJob("def add(a, b) return", TESTS),
    ] * 4
    statuses = [r.status for r in executor.run_many(jobs)]
    assert statuses == [Status.PASSED, Status.FAILED, Status.SYNTAX_ERROR] * 4
