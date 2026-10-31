"""Benchmark generator: plant single-operator bugs into a real repo and keep the ones tests catch.

    python -m ast_agent.mutate --repo path/to/clean_repo \
        --test-cmd "python -m pytest -q -x --tb=short" --out benchmarks/mylib --max-tasks 50

Each kept mutant becomes a task whose issue text is derived only from the failing test
output (never from the mutation), and whose gold symbol is the mutated function.
The clean repo is not copied per task: each task stores a small `bug` record that the
eval harness applies to a fresh copy.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass

from .indexer import RepoIndex, is_test_path
from .runner import run_tests

COMPARE_SWAP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
                ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}
BINOP_SWAP = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.FloorDiv: ("//", "/")}
BOOL_SWAP = {ast.And: ("and", "or"), ast.Or: ("or", "and")}
IGNORE_DIRS = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc", ".venv", "venv")


@dataclass
class Mutation:
    path: str
    start: int        # char offsets into the file text
    end: int
    replacement: str
    lineno: int
    kind: str


def _offsets(text: str):
    lines = text.splitlines(keepends=True)
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)

    def to_idx(lineno: int, col_bytes: int) -> int:
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col_bytes].decode("utf-8", "ignore"))
    return to_idx


def _top_functions(tree: ast.AST):
    """Functions/methods not nested inside another function (matches the indexer's symbols)."""
    stack = list(getattr(tree, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            stack.extend(node.body)


def candidate_mutations(rel: str, text: str) -> list[Mutation]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    to_idx = _offsets(text)
    out: list[Mutation] = []

    def between(a: ast.AST, b: ast.AST, expected: str, new: str, kind: str, node: ast.AST):
        if a.end_lineno != b.lineno:
            return
        s, e = to_idx(a.end_lineno, a.end_col_offset), to_idx(b.lineno, b.col_offset)
        seg = text[s:e]
        if seg.strip() != expected:
            return
        out.append(Mutation(rel, s, e, seg.replace(expected, new, 1), node.lineno, kind))

    for fn in _top_functions(tree):
        body_nodes = [n for stmt in fn.body for n in ast.walk(stmt)]
        for node in body_nodes:
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE_SWAP:
                old, new = COMPARE_SWAP[type(node.ops[0])]
                between(node.left, node.comparators[0], old, new, "compare", node)
            elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAP:
                old, new = BINOP_SWAP[type(node.op)]
                between(node.left, node.right, old, new, "arith", node)
            elif isinstance(node, ast.BoolOp) and len(node.values) == 2 and type(node.op) in BOOL_SWAP:
                old, new = BOOL_SWAP[type(node.op)]
                between(node.values[0], node.values[1], old, new, "boolean", node)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) \
                    and node.lineno == node.operand.lineno:
                s, e = to_idx(node.lineno, node.col_offset), to_idx(node.operand.lineno, node.operand.col_offset)
                if text[s:e].strip() == "not":
                    out.append(Mutation(rel, s, e, "", node.lineno, "negation"))
            elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 2) \
                    and node.lineno == node.end_lineno:
                s, e = to_idx(node.lineno, node.col_offset), to_idx(node.end_lineno, node.end_col_offset)
                if text[s:e] == str(node.value):
                    out.append(Mutation(rel, s, e, str(node.value + 1), node.lineno, "constant"))
    return out


def _snippets(original: str, mutated: str, lineno: int, max_extra: int = 6) -> tuple[str, str] | None:
    """Smallest window around `lineno` that is unique in both versions (for gold patches)."""
    o_lines, m_lines = original.splitlines(), mutated.splitlines()
    if len(o_lines) != len(m_lines):
        return None
    for extra in range(max_extra + 1):
        lo, hi = max(0, lineno - 1 - extra), min(len(o_lines), lineno + extra)
        o_snip, m_snip = "\n".join(o_lines[lo:hi]), "\n".join(m_lines[lo:hi])
        if original.count(o_snip) == 1 and mutated.count(m_snip) == 1:
            return o_snip, m_snip
    return None


def issue_from_output(output: str) -> str:
    failed = [ln.strip() for ln in output.splitlines() if ln.startswith(("FAILED ", "ERROR "))]
    if failed:
        body = "\n".join(f"- {ln[ln.index(' ') + 1:]}" for ln in failed[:5])
        return ("The following tests fail on the current code:\n" + body +
                "\n\nFind and fix the bug in the library code. The tests are correct.")
    tail = "\n".join(output.strip().splitlines()[-15:])
    return f"The test suite fails with:\n```\n{tail}\n```\nFind and fix the bug in the library code."


def apply_bug(repo_dir: str, bug: dict) -> None:
    """Plant a recorded bug into a working copy (used by the eval harness)."""
    path = os.path.join(repo_dir, bug["path"])
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(bug["original_snippet"]) != 1:
        raise RuntimeError(f"bug anchor not unique/found in {bug['path']}; was the clean repo changed?")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(bug["original_snippet"], bug["buggy_snippet"], 1))


def generate(repo: str, test_cmd: str, out_dir: str, max_tasks: int, seed: int,
             per_symbol: int = 2, timeout: int = 120, prefix: str | None = None) -> list[dict]:
    repo = os.path.abspath(repo)
    os.makedirs(out_dir, exist_ok=True)
    index = RepoIndex.build(repo)
    prefix = prefix or os.path.basename(repo.rstrip("/"))
    rng = random.Random(seed)

    work = tempfile.mkdtemp(prefix="mutate_")
    shutil.copytree(repo, work, dirs_exist_ok=True, ignore=IGNORE_DIRS)
    try:
        clean = run_tests(test_cmd, work, timeout)
        if not clean.passed:
            raise SystemExit(f"Clean repo must pass its tests first. Output:\n{clean.output[-2000:]}")

        cands = []
        for rel, text in index.files.items():
            if not is_test_path(rel):
                cands.extend(candidate_mutations(rel, text))
        rng.shuffle(cands)
        print(f"{len(cands)} candidate mutations; testing until {max_tasks} are kept...")

        out_path = os.path.join(out_dir, "tasks.jsonl")
        open(out_path, "w").close()          # tasks are appended as they are found
        tasks, per_sym = [], {}
        for tried, m in enumerate(cands, 1):
            if len(tasks) >= max_tasks:
                break
            if tried % 25 == 0:
                print(f"  ...tried {tried} mutations, kept {len(tasks)}", flush=True)
            sym = index.symbol_at(m.path, m.lineno)
            if sym is None or per_sym.get(sym.id, 0) >= per_symbol:
                continue
            original = index.files[m.path]
            mutated = original[:m.start] + m.replacement + original[m.end:]
            try:
                ast.parse(mutated)
            except SyntaxError:
                continue
            snips = _snippets(original, mutated, m.lineno)
            if snips is None:
                continue
            target = os.path.join(work, m.path)
            with open(target, "w", encoding="utf-8") as f:
                f.write(mutated)
            res = run_tests(test_cmd, work, timeout)
            with open(target, "w", encoding="utf-8") as f:
                f.write(original)
            if res.passed or res.timed_out or re.search(r"error(s)? (during|while) collect", res.output, re.I):
                continue
            per_sym[sym.id] = per_sym.get(sym.id, 0) + 1
            tasks.append({
                "id": f"{prefix}-{len(tasks) + 1:03d}",
                "repo": os.path.relpath(repo, os.path.abspath(out_dir)).replace(os.sep, "/"),
                "issue": issue_from_output(res.output),
                "test_cmd": test_cmd,
                "gold": [sym.id],
                "bug": {"path": m.path, "line": m.lineno, "kind": m.kind,
                        "original_snippet": snips[0], "buggy_snippet": snips[1]},
            })
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tasks[-1]) + "\n")
            print(f"  kept {tasks[-1]['id']}: {m.kind} in {sym.id}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"Wrote {len(tasks)} tasks to {out_path}")
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="clean repository whose tests pass")
    ap.add_argument("--test-cmd", default="python -m pytest -q -x --tb=short")
    ap.add_argument("--out", required=True, help="directory for tasks.jsonl")
    ap.add_argument("--max-tasks", type=int, default=50)
    ap.add_argument("--per-symbol", type=int, default=2, help="max bugs per function")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--prefix", default=None, help="task id prefix (default: repo folder name)")
    a = ap.parse_args(argv)
    generate(a.repo, a.test_cmd, a.out, a.max_tasks, a.seed, a.per_symbol, a.timeout, a.prefix)


if __name__ == "__main__":
    main()
