import json
import os
import shutil

import pytest

from ast_agent.agent import Agent, AgentConfig
from ast_agent.edits import apply_edits, parse_edits, revert
from ast_agent.eval import load_tasks, mcnemar_exact, run_one, wilson
from ast_agent.indexer import RepoIndex
from ast_agent.models import ScriptedClient
from ast_agent.mutate import apply_bug, candidate_mutations
from ast_agent.retrieval import make_retriever, traceback_symbols

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "benchmarks", "sample")
REPO = os.path.join(SAMPLE, "repo")


@pytest.fixture
def repo(tmp_path):
    dst = tmp_path / "repo"
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return str(dst)


# ---------------------------------------------------------------- indexer
def test_index_symbols_and_graph():
    idx = RepoIndex.build(REPO)
    ids = {s.id for s in idx.symbols}
    assert "minilib/stats.py::median" in ids
    assert "minilib/shapes.py::Rectangle.contains" in ids
    assert "minilib/shapes.py::Rectangle" in ids                 # class outline
    assert not any(s.path.startswith("tests/") for s in idx.symbols)  # tests excluded
    median = idx.by_id["minilib/stats.py::median"]
    assert "_sorted_copy" in {s.short_name for s in idx.callees(median)}
    mean = idx.by_id["minilib/stats.py::mean"]
    assert "variance" in {s.short_name for s in idx.callers(mean)}
    assert idx.symbol_at("minilib/stats.py", median.start + 2).id == median.id


# ---------------------------------------------------------------- retrieval
def test_ast_retrieval_finds_named_symbol():
    idx = RepoIndex.build(REPO)
    chunks = make_retriever("ast", idx).retrieve("truncate() returns the wrong length", 200)
    assert any(c.symbol_id == "minilib/text.py::truncate" for c in chunks)
    assert sum(c.tokens for c in chunks) <= 200


def test_graph_expansion_adds_callee():
    idx = RepoIndex.build(REPO)
    chunks = make_retriever("ast_graph", idx).retrieve("median gives wrong result", 2000)
    got = {c.symbol_id for c in chunks}
    assert "minilib/stats.py::median" in got and "minilib/stats.py::_sorted_copy" in got


def test_traceback_symbols_maps_frames():
    idx = RepoIndex.build(REPO)
    med = idx.by_id["minilib/stats.py::median"]
    out = f'File "{idx.root}/minilib/stats.py", line {med.start + 3}, in median\n'
    assert [s.id for s in traceback_symbols(idx, out)] == [med.id]
    out2 = f"minilib/stats.py:{med.start + 3}: in median"
    assert [s.id for s in traceback_symbols(idx, out2)] == [med.id]


# ---------------------------------------------------------------- edits
BLOCK = """Root cause: off by one.

```python
minilib/stats.py
<<<<<<< SEARCH
    mid = n // 2
=======
    mid = n // 2  # middle index
>>>>>>> REPLACE
```
"""


def test_parse_and_apply_exact(repo):
    edits = parse_edits(BLOCK)
    assert len(edits) == 1 and edits[0].path == "minilib/stats.py"
    res = apply_edits(repo, edits)
    assert res.ok, res.errors
    assert "# middle index" in open(os.path.join(repo, "minilib/stats.py")).read()
    assert "+    mid = n // 2  # middle index" in res.diff()
    revert(repo, res)
    assert "# middle index" not in open(os.path.join(repo, "minilib/stats.py")).read()


def test_apply_fuzzy_indentation(repo):
    text = ("minilib/stats.py\n<<<<<<< SEARCH\nmid = n // 2\nif n % 2 == 1:\n=======\n"
            "mid = n // 2\nif n % 2 != 0:\n>>>>>>> REPLACE\n")
    res = apply_edits(repo, parse_edits(text))
    assert res.ok, res.errors
    assert "    if n % 2 != 0:" in open(os.path.join(repo, "minilib/stats.py")).read()


def test_apply_rejects_bad_edits_atomically(repo):
    before = open(os.path.join(repo, "minilib/stats.py")).read()
    text = ("minilib/stats.py\n<<<<<<< SEARCH\n    mid = n // 2\n=======\n    mid = n // 3\n>>>>>>> REPLACE\n"
            "minilib/stats.py\n<<<<<<< SEARCH\nthis line does not exist\n=======\nx\n>>>>>>> REPLACE\n")
    res = apply_edits(repo, parse_edits(text))
    assert not res.ok and any("not found" in e for e in res.errors)
    assert open(os.path.join(repo, "minilib/stats.py")).read() == before   # nothing written


def test_apply_rejects_tests_syntax_and_escape(repo):
    edit = lambda p, s, r: parse_edits(f"{p}\n<<<<<<< SEARCH\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n")
    assert "read-only" in apply_edits(repo, edit("tests/test_minilib.py", "import pytest", "x = 1")).errors[0]
    assert "SyntaxError" in apply_edits(repo, edit("minilib/stats.py", "    mid = n // 2", "    mid = (")).errors[0]
    assert "inside the repository" in apply_edits(repo, edit("../evil.py", "", "x = 1")).errors[0]
    assert "no edit blocks" in apply_edits(repo, parse_edits("I think it is fine.")).errors[0]


# ---------------------------------------------------------------- mutation + harness
def test_candidate_mutations_cover_operator_kinds():
    text = open(os.path.join(REPO, "minilib/stats.py")).read()
    kinds = {m.kind for m in candidate_mutations("minilib/stats.py", text)}
    assert {"compare", "arith", "constant"} <= kinds


def _task(task_id):
    return next(t for t in load_tasks(os.path.join(SAMPLE, "tasks.jsonl")) if t["id"] == task_id)


def test_gold_patch_resolves_every_sample_task():
    for t in load_tasks(os.path.join(SAMPLE, "tasks.jsonl")):
        r = run_one(t, "full", "gold", AgentConfig(max_attempts=1))
        assert r["resolved"], (t["id"], r["attempts"][-1]["apply_errors"], r["validation_output"])


def _fix_block(bug):
    return (f"{bug['path']}\n<<<<<<< SEARCH\n{bug['buggy_snippet']}\n=======\n"
            f"{bug['original_snippet']}\n>>>>>>> REPLACE\n")


def test_self_correction_recovers_after_wrong_patch(repo):
    t = _task("minilib-003")          # bug planted in stats.mean
    apply_bug(repo, t["bug"])
    bug = t["bug"]
    wrong = f"{bug['path']}\n<<<<<<< SEARCH\n{bug['buggy_snippet']}\n=======\n{bug['buggy_snippet']}  # noop\n>>>>>>> REPLACE\n"
    model = ScriptedClient([wrong, _fix_block(bug)])
    res = Agent(model, AgentConfig(retriever="ast_graph", max_attempts=3)).solve(
        repo, t["issue"], t["test_cmd"], gold=t["gold"], task_id=t["id"])
    assert res.resolved and res.attempts_used == 2
    assert res.attempts[0].tests_passed is False
    assert res.attempts[1].tests_passed is True and model.calls == 2
    assert "+" in res.patch and json.dumps(res.to_dict())          # serializable for results.jsonl


def test_no_self_correction_stops_after_one_attempt(repo):
    t = _task("minilib-003")
    apply_bug(repo, t["bug"])
    model = ScriptedClient(["no idea"])
    res = Agent(model, AgentConfig(max_attempts=1)).solve(repo, t["issue"], t["test_cmd"], gold=t["gold"])
    assert not res.resolved and res.attempts_used == 1 and model.calls == 1


# ---------------------------------------------------------------- model backend
def test_openai_compatible_client_against_fake_server():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from ast_agent.models import OpenAICompatibleClient

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({"choices": [{"message": {"content": "patched"}}],
                                  "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}/v1", "gemma-test")
        gen = client.generate([{"role": "user", "content": "hi"}], max_tokens=5)
    finally:
        server.shutdown()
    assert gen.text == "patched" and gen.prompt_tokens == 11 and gen.completion_tokens == 3
    assert seen["path"] == "/v1/chat/completions" and seen["body"]["model"] == "gemma-test"


# ---------------------------------------------------------------- runner
def test_timeout_kills_hanging_grandchild(tmp_path):
    import time

    from ast_agent.runner import run_tests

    (tmp_path / "hang.py").write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'while True: pass'])\n")
    t0 = time.time()
    res = run_tests("python hang.py", str(tmp_path), timeout=2)
    assert res.timed_out and not res.passed
    assert time.time() - t0 < 15          # the old version waited forever here


def test_src_layout_imports_working_copy(tmp_path):
    from ast_agent.runner import run_tests

    pkg = tmp_path / "src" / "srcpkg_demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 41\n")
    (tmp_path / "check.py").write_text("import srcpkg_demo, sys\nsys.exit(0 if srcpkg_demo.VALUE == 41 else 1)\n")
    assert run_tests("python check.py", str(tmp_path), timeout=30).passed
    (pkg / "__init__.py").write_text("VALUE = 42\n")          # a planted change must be seen
    assert not run_tests("python check.py", str(tmp_path), timeout=30).passed


# ---------------------------------------------------------------- stats
def test_statistics():
    lo, hi = wilson(17, 17)
    assert 0.8 < lo < 0.82 and hi == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
