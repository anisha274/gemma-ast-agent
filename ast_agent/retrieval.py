"""Context retrieval strategies compared in the experiments.

    file       - BM25 over whole files, packed into the token budget (baseline)
    ast        - BM25 over AST symbols (functions / methods / class outlines)
    ast_graph  - `ast` plus call-graph expansion (callers + callees of top hits)
    oracle     - gold symbols first, then `ast_graph` (upper bound on localization)

All retrievers return `ContextChunk`s and respect the same approximate token budget,
so differences in resolve rate come from *what* is shown, not *how much*.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .indexer import RepoIndex, Symbol


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token for code)."""
    return max(1, len(text) // 4)


_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "on", "for", "it",
    "this", "that", "with", "be", "as", "at", "by", "from", "not", "when", "should",
    "self", "def", "return", "none", "true", "false", "import", "if", "else", "py",
}


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: keeps `parse_date` and also emits `parse`, `date`."""
    out = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        low = word.lower()
        parts = [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word.replace("_", " "))]
        for tok in {low, *parts}:
            if len(tok) > 1 and tok not in _STOP:
                out.append(tok)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / len(docs)) if docs else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t, qf in q.items():
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


@dataclass
class ContextChunk:
    path: str
    code: str
    label: str                 # e.g. "function mean" or "file (truncated)"
    start: int = 0             # 1-based; 0 for synthetic chunks
    end: int = 0
    symbol_id: str | None = None
    score: float = 0.0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.code)

    def covers(self, path: str, start: int, end: int) -> bool:
        return self.path == path and self.start > 0 and self.start <= start and self.end >= end


# Traceback frames: standard Python and pytest --tb=short styles.
_TB_PATTERNS = [
    re.compile(r'File "([^"]+\.py)", line (\d+)'),
    re.compile(r"^([^\s:]+\.py):(\d+):", re.MULTILINE),
]


def traceback_frames(text: str) -> list[tuple[str, int]]:
    frames = []
    for pat in _TB_PATTERNS:
        for m in pat.finditer(text):
            frames.append((m.group(1), int(m.group(2))))
    return frames


def traceback_symbols(index: RepoIndex, text: str) -> list[Symbol]:
    """Map traceback frames (absolute or relative paths) to indexed symbols, deepest frame last."""
    root = index.root.replace("\\", "/").rstrip("/") + "/"
    out: list[Symbol] = []
    for path, line in traceback_frames(text):
        p = path.replace("\\", "/")
        if p.startswith(root):
            p = p[len(root):]
        while p.startswith("./"):
            p = p[2:]
        sym = index.symbol_at(p, line)
        if sym and sym not in out:
            out.append(sym)
    return out


class Retriever:
    name = "base"

    def __init__(self, index: RepoIndex):
        self.index = index

    def retrieve(self, query: str, budget: int) -> list[ContextChunk]:
        raise NotImplementedError


class FileRetriever(Retriever):
    """Baseline: rank whole files with BM25 and pack them (truncating the last one)."""
    name = "file"

    def retrieve(self, query: str, budget: int) -> list[ContextChunk]:
        paths = list(self.index.files)
        if not paths:
            return []
        bm = BM25([tokenize(p + " " + self.index.files[p]) for p in paths])
        ranked = sorted(zip(bm.scores(tokenize(query)), paths), key=lambda x: -x[0])
        chunks, used = [], 0
        for score, path in ranked:
            if score <= 0 or used >= budget:
                break
            text = self.index.files[path]
            lines = text.splitlines()
            remaining_chars = (budget - used) * 4
            if len(text) > remaining_chars:
                kept, size = [], 0
                for ln in lines:
                    if size + len(ln) + 1 > remaining_chars:
                        break
                    kept.append(ln)
                    size += len(ln) + 1
                if not kept:
                    break
                chunk = ContextChunk(path, "\n".join(kept), "file (truncated)", 1, len(kept), score=score)
            else:
                chunk = ContextChunk(path, text, "file", 1, len(lines), score=score)
            chunks.append(chunk)
            used += chunk.tokens
        return chunks


class ASTRetriever(Retriever):
    """BM25 over AST symbols, with optional call-graph expansion and traceback boosting."""

    def __init__(self, index: RepoIndex, graph: bool = False, graph_seeds: int = 3,
                 graph_decay: float = 0.5):
        super().__init__(index)
        self.graph = graph
        self.graph_seeds = graph_seeds
        self.graph_decay = graph_decay
        self.name = "ast_graph" if graph else "ast"
        docs = []
        for s in index.symbols:
            name_toks = tokenize(s.name) * 3          # names matter most
            docs.append(name_toks + tokenize(s.path) + tokenize(s.docstring) + tokenize(s.code))
        self.bm25 = BM25(docs) if docs else None

    def rank(self, query: str) -> list[tuple[float, Symbol]]:
        if not self.bm25:
            return []
        scores = self.bm25.scores(tokenize(query))
        q_idents = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)}
        tb_ids = {s.id for s in traceback_symbols(self.index, query)}
        top = max(scores) if scores else 0.0
        ranked = []
        for s, sc in zip(self.index.symbols, scores):
            if s.short_name.lower() in q_idents and s.kind != "module":
                sc += 0.5 * top              # issue mentions the symbol by name
            if s.id in tb_ids:
                sc += 1.0 * top              # symbol appears in a traceback frame
            if sc > 0:
                ranked.append((sc, s))
        ranked.sort(key=lambda x: -x[0])
        return ranked

    def retrieve(self, query: str, budget: int) -> list[ContextChunk]:
        ranked = self.rank(query)
        scored: dict[str, tuple[float, Symbol]] = {s.id: (sc, s) for sc, s in ranked}
        if self.graph:
            for sc, seed in ranked[: self.graph_seeds]:
                for nb in self.index.callees(seed) + self.index.callers(seed):
                    boosted = sc * self.graph_decay
                    if nb.id not in scored or scored[nb.id][0] < boosted:
                        scored[nb.id] = (boosted, nb)
        return pack_symbols(sorted(scored.values(), key=lambda x: -x[0]), budget)


def symbol_chunk(sym: Symbol, score: float = 0.0) -> ContextChunk:
    label = {"class": "class outline", "module": "module header"}.get(sym.kind, sym.kind)
    return ContextChunk(
        path=sym.path, code=sym.code, label=f"{label} {sym.name}",
        start=0 if sym.synthetic else sym.start, end=0 if sym.synthetic else sym.end,
        symbol_id=sym.id, score=score,
    )


def pack_symbols(ranked: list[tuple[float, Symbol]], budget: int,
                 already: set[str] | None = None) -> list[ContextChunk]:
    """Greedy knapsack by score; skips symbols that don't fit and keeps trying smaller ones."""
    chunks, used, seen = [], 0, set(already or ())
    for sc, sym in ranked:
        if sym.id in seen:
            continue
        chunk = symbol_chunk(sym, sc)
        if used + chunk.tokens > budget:
            continue
        chunks.append(chunk)
        seen.add(sym.id)
        used += chunk.tokens
    # Present in file order so the model reads code naturally.
    chunks.sort(key=lambda c: (c.path, c.start if c.start else 10**9))
    return chunks


class OracleRetriever(ASTRetriever):
    """Upper bound: gold symbols are always included, the rest filled by ast_graph."""

    def __init__(self, index: RepoIndex, gold_ids: list[str]):
        super().__init__(index, graph=True)
        self.name = "oracle"
        self.gold = [index.by_id[g] for g in gold_ids if g in index.by_id]

    def retrieve(self, query: str, budget: int) -> list[ContextChunk]:
        top = 1e9
        ranked = [(top, g) for g in self.gold] + self.rank(query)
        return pack_symbols(ranked, budget)


def make_retriever(name: str, index: RepoIndex, gold_ids: list[str] | None = None) -> Retriever:
    if name == "file":
        return FileRetriever(index)
    if name == "ast":
        return ASTRetriever(index, graph=False)
    if name == "ast_graph":
        return ASTRetriever(index, graph=True)
    if name == "oracle":
        return OracleRetriever(index, gold_ids or [])
    raise ValueError(f"unknown retriever: {name}")
