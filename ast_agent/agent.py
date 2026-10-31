"""The agent loop: Retrieve -> Generate patch -> Apply -> Test -> (Self-correct) -> Validate.

Self-correction resets the repo after each failed attempt and asks for a fresh patch,
feeding back the previous diff, the test output, and (optionally) the code of any
repository symbols that appear in the failing traceback but were not shown before.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from . import prompts
from .edits import apply_edits, parse_edits, revert
from .indexer import RepoIndex
from .models import ModelClient
from .retrieval import make_retriever, pack_symbols, traceback_symbols
from .runner import run_tests

log = logging.getLogger("ast_agent")


@dataclass
class AgentConfig:
    retriever: str = "ast_graph"      # file | ast | ast_graph | oracle
    context_budget: int = 6000        # approx tokens of retrieved code in the first prompt
    max_attempts: int = 3             # 1 = no self-correction
    traceback_retrieval: bool = True  # add traceback symbols to self-correction feedback
    traceback_budget: int = 1500
    max_new_tokens: int = 2048
    temperature: float = 0.0
    test_timeout: int = 300


@dataclass
class AttemptLog:
    attempt: int
    n_edits: int
    apply_errors: list[str]
    tests_passed: bool | None
    test_output_tail: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    response: str


@dataclass
class AgentResult:
    task_id: str
    condition: str
    resolved: bool
    attempts_used: int
    patch: str
    context_tokens: int
    retrieved: list[str]
    file_hit: bool | None
    symbol_hit: bool | None
    prompt_tokens: int
    completion_tokens: int
    model_time_s: float
    wall_time_s: float
    validation_output: str = ""
    error: str = ""
    attempts: list[AttemptLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _localization(chunks, index: RepoIndex, gold: list[str]) -> tuple[bool | None, bool | None]:
    if not gold:
        return None, None
    gold_syms = [index.by_id[g] for g in gold if g in index.by_id]
    gold_files = {g.split("::")[0] for g in gold}
    file_hit = any(c.path in gold_files for c in chunks)
    symbol_hit = bool(gold_syms) and all(
        any(c.symbol_id == g.id or c.covers(g.path, g.start, g.end) for c in chunks) for g in gold_syms
    )
    return file_hit, symbol_hit


class Agent:
    def __init__(self, model: ModelClient, config: AgentConfig | None = None):
        self.model = model
        self.cfg = config or AgentConfig()

    def solve(self, repo_dir: str, issue: str, test_cmd: str, feedback_cmd: str | None = None,
              gold: list[str] | None = None, task_id: str = "task", condition: str = "") -> AgentResult:
        cfg, t_start = self.cfg, time.time()
        feedback_cmd = feedback_cmd or test_cmd
        index = RepoIndex.build(repo_dir)
        retriever = make_retriever(cfg.retriever, index, gold)
        chunks = retriever.retrieve(issue, cfg.context_budget)
        file_hit, symbol_hit = _localization(chunks, index, gold or [])
        shown = {c.symbol_id for c in chunks if c.symbol_id}

        messages = [{"role": "user", "content": prompts.initial_prompt(issue, chunks, sorted(index.files))}]
        res = AgentResult(
            task_id=task_id, condition=condition or cfg.retriever, resolved=False, attempts_used=0,
            patch="", context_tokens=sum(c.tokens for c in chunks),
            retrieved=[c.symbol_id or f"{c.path}:{c.start}-{c.end}" for c in chunks],
            file_hit=file_hit, symbol_hit=symbol_hit, prompt_tokens=0, completion_tokens=0,
            model_time_s=0.0, wall_time_s=0.0,
        )

        final_apply = None
        for attempt in range(1, cfg.max_attempts + 1):
            res.attempts_used = attempt
            try:
                gen = self.model.generate(messages, cfg.max_new_tokens, cfg.temperature)
            except Exception as exc:  # network / OOM: record and stop, don't crash the whole eval
                res.error = f"model error: {exc!r}"
                log.exception("model call failed")
                break
            res.prompt_tokens += gen.prompt_tokens
            res.completion_tokens += gen.completion_tokens
            res.model_time_s += gen.latency_s
            messages.append({"role": "assistant", "content": gen.text})

            edits = parse_edits(gen.text)
            applied = apply_edits(repo_dir, edits)
            alog = AttemptLog(attempt, len(edits), applied.errors, None, "",
                              gen.prompt_tokens, gen.completion_tokens, gen.latency_s, gen.text)
            res.attempts.append(alog)

            if not applied.ok:
                log.info("[%s] attempt %d: apply failed: %s", task_id, attempt, applied.errors)
                fb = prompts.no_edits_feedback() if not edits else prompts.apply_error_feedback(applied.errors)
                messages.append({"role": "user", "content": fb})
                continue

            tests = run_tests(feedback_cmd, repo_dir, cfg.test_timeout)
            alog.tests_passed, alog.test_output_tail = tests.passed, tests.output[-1500:]
            log.info("[%s] attempt %d: tests %s", task_id, attempt, "PASS" if tests.passed else "FAIL")
            if tests.passed:
                final_apply = applied
                break
            if attempt == cfg.max_attempts:
                final_apply = applied      # keep the last patch for validation/reporting
                break

            revert(repo_dir, applied)
            extra = []
            if cfg.traceback_retrieval:
                tb = [(1.0, s) for s in traceback_symbols(index, tests.output) if s.id not in shown]
                extra = pack_symbols(tb, cfg.traceback_budget)
                shown.update(c.symbol_id for c in extra if c.symbol_id)
            messages.append({"role": "user", "content": prompts.test_failure_feedback(
                applied.diff(), tests.output[-2500:], extra)})

        if final_apply is not None:
            res.patch = final_apply.diff()
            val = run_tests(test_cmd, repo_dir, cfg.test_timeout)
            res.resolved, res.validation_output = val.passed, val.output[-1500:]
        res.wall_time_s = time.time() - t_start
        return res
