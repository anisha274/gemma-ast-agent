"""Run the experiment grid (tasks x conditions) and produce the paper's results table.

    python -m ast_agent.eval --tasks benchmarks/sample/tasks.jsonl \
        --conditions file,ast,ast_graph,ast_graph_sc,full,oracle \
        --backend openai --base-url http://localhost:8000/v1 --model google/gemma-4-e4b-it \
        --out results/e4b

Results are appended to <out>/results.jsonl as each task finishes, so an interrupted
Kaggle/Colab session can simply be re-run with the same command and it resumes.
<out>/summary.md holds the results table; <out>/run_meta.json records the setup.

    python -m ast_agent.eval --summarize results/e4b      # re-render the table only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import fields

from .agent import Agent, AgentConfig
from .models import GoldPatchClient, OpenAICompatibleClient
from .mutate import IGNORE_DIRS, apply_bug

# Ablation grid. Each row isolates one component; budgets are identical across rows.
CONDITIONS: dict[str, dict] = {
    "file":         dict(retriever="file", max_attempts=1),
    "ast":          dict(retriever="ast", max_attempts=1),
    "ast_graph":    dict(retriever="ast_graph", max_attempts=1),
    "file_sc":      dict(retriever="file", max_attempts=3, traceback_retrieval=False),
    "ast_graph_sc": dict(retriever="ast_graph", max_attempts=3, traceback_retrieval=False),
    "full":         dict(retriever="ast_graph", max_attempts=3, traceback_retrieval=True),
    "oracle":       dict(retriever="oracle", max_attempts=1),
}
DESCRIPTIONS = {
    "file": "Whole-file BM25 retrieval (baseline)",
    "ast": "AST symbol retrieval",
    "ast_graph": "AST + call-graph expansion",
    "file_sc": "Whole-file + self-correction (3 attempts)",
    "ast_graph_sc": "AST + graph + self-correction",
    "full": "AST + graph + self-correction + traceback retrieval",
    "oracle": "Gold symbols in context (upper bound)",
}


def load_tasks(path: str) -> list[dict]:
    base = os.path.dirname(os.path.abspath(path))
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                t["repo_abs"] = os.path.normpath(os.path.join(base, t["repo"]))
                tasks.append(t)
    return tasks


def done_keys(results_path: str) -> set[tuple[str, str]]:
    keys = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["task_id"], r["condition"]))
    return keys


def run_one(task: dict, condition: str, model, cfg: AgentConfig) -> dict:
    work = tempfile.mkdtemp(prefix=f"{task['id']}_")
    try:
        shutil.copytree(task["repo_abs"], work, dirs_exist_ok=True, ignore=IGNORE_DIRS)
        if task.get("bug"):
            apply_bug(work, task["bug"])
        if model == "gold":
            model = GoldPatchClient(task["bug"])
        agent = Agent(model, cfg)
        res = agent.solve(work, task["issue"], task["test_cmd"], task.get("feedback_cmd"),
                          task.get("gold"), task_id=task["id"], condition=condition)
        return res.to_dict()
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(out_dir: str, baseline: str = "file") -> str:
    rows = []
    with open(os.path.join(out_dir, "results.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_cond: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], {})[r["task_id"]] = r
    order = [c for c in CONDITIONS if c in by_cond] + sorted(c for c in by_cond if c not in CONDITIONS)

    def pct(x):
        return f"{100 * x:.1f}"

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    head = ("| Condition | Resolved | Resolve % [95% CI] | p vs baseline | File hit % | Symbol hit % "
            "| Context tok | Total tok / task | Attempts | Model s / task |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    summary_json = {}
    base = by_cond.get(baseline, {})
    for c in order:
        rs = list(by_cond[c].values())
        n, k = len(rs), sum(r["resolved"] for r in rs)
        lo, hi = wilson(k, n)
        common = [t for t in by_cond[c] if t in base]
        if c != baseline and common:
            b_only = sum(base[t]["resolved"] and not by_cond[c][t]["resolved"] for t in common)
            c_only = sum(by_cond[c][t]["resolved"] and not base[t]["resolved"] for t in common)
            p = f"{mcnemar_exact(b_only, c_only):.3f}"
        else:
            p = "-"
        fh = mean([float(r["file_hit"]) if r["file_hit"] is not None else None for r in rs])
        sh = mean([float(r["symbol_hit"]) if r["symbol_hit"] is not None else None for r in rs])
        ctx = mean([r["context_tokens"] for r in rs])
        tot = mean([r["prompt_tokens"] + r["completion_tokens"] for r in rs])
        att = mean([r["attempts_used"] for r in rs])
        sec = mean([r["model_time_s"] for r in rs])
        lines.append(f"| {c} | {k}/{n} | {pct(k / n if n else 0)} [{pct(lo)}-{pct(hi)}] | {p} | "
                     f"{pct(fh) if fh == fh else '-'} | {pct(sh) if sh == sh else '-'} | {ctx:.0f} | "
                     f"{tot:.0f} | {att:.2f} | {sec:.1f} |")
        summary_json[c] = dict(n=n, resolved=k, rate=k / n if n else 0, ci95=[lo, hi], p_vs_baseline=p,
                               file_hit=fh, symbol_hit=sh, context_tokens=ctx, total_tokens=tot,
                               attempts=att, model_seconds=sec)
    legend = "\n".join(f"- **{c}**: {DESCRIPTIONS.get(c, c)}" for c in order)
    md = ("\n".join(lines) + f"\n\nBaseline for p-values: `{baseline}` (exact McNemar test on paired tasks). "
          "Token counts are model-reported when available, otherwise ~4 chars/token estimates.\n\n" + legend + "\n")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    return md


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(__file__)).stdout.strip()
    except OSError:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks")
    ap.add_argument("--conditions", default="file,ast,ast_graph,ast_graph_sc,full,oracle")
    ap.add_argument("--backend", choices=["openai", "hf", "gold"], default="openai",
                    help="gold = apply the known fix (checks tasks + harness, no model needed)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="google/gemma-4-e4b-it")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    ap.add_argument("--dtype", default="auto", help="hf backend: auto | float16 | bfloat16")
    ap.add_argument("--load-in-4bit", action="store_true", help="hf backend: 4-bit via bitsandbytes")
    ap.add_argument("--context-budget", type=int, default=6000)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    ap.add_argument("--baseline", default="file")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--summarize", metavar="OUT_DIR", help="only rebuild summary.md from results.jsonl")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.summarize:
        print(summarize(a.summarize, a.baseline))
        return
    if not a.tasks:
        ap.error("--tasks is required")

    tasks = load_tasks(a.tasks)[: a.limit or None]
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f"unknown conditions {unknown}; choose from {list(CONDITIONS)}")

    if a.backend == "openai":
        model = OpenAICompatibleClient(a.base_url, a.model, a.api_key)
    elif a.backend == "hf":
        from .models import TransformersClient
        model = TransformersClient(a.model, dtype=a.dtype, load_in_4bit=a.load_in_4bit)
    else:
        model = "gold"

    os.makedirs(a.out, exist_ok=True)
    meta = dict(model=a.model, backend=a.backend, dtype=a.dtype, load_in_4bit=a.load_in_4bit, base_url=a.base_url if a.backend == "openai" else None,
                tasks_file=a.tasks, n_tasks=len(tasks), conditions=conditions,
                context_budget=a.context_budget, max_new_tokens=a.max_new_tokens,
                temperature=a.temperature, python=sys.version.split()[0], platform=platform.platform(),
                git_commit=_git_commit(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(a.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results_path = os.path.join(a.out, "results.jsonl")
    done = done_keys(results_path)
    cfg_fields = {f.name for f in fields(AgentConfig)}
    total = len(tasks) * len(conditions)
    i = 0
    for cond in conditions:
        overrides = {k: v for k, v in CONDITIONS[cond].items() if k in cfg_fields}
        cfg = AgentConfig(context_budget=a.context_budget, max_new_tokens=a.max_new_tokens,
                          temperature=a.temperature, **overrides)
        for t in tasks:
            i += 1
            if (t["id"], cond) in done:
                continue
            r = run_one(t, cond, model, cfg)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{i}/{total}] {cond:13s} {t['id']}: {'RESOLVED' if r['resolved'] else 'failed'}"
                  f" (attempts={r['attempts_used']}, ctx={r['context_tokens']})", flush=True)
    print(summarize(a.out, a.baseline))


if __name__ == "__main__":
    main()
