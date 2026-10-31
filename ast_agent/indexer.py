"""AST indexing: turn a Python repository into retrievable symbols plus a call graph.

Each function, method and class becomes a `Symbol` with exact line ranges, so the
agent can show the model precise, self-contained code units instead of whole files.
"""
from __future__ import annotations

import ast
import fnmatch
import os
from dataclasses import dataclass, field

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".tox",
    "build", "dist", ".eggs", ".mypy_cache", ".pytest_cache", "site-packages",
}
TEST_GLOBS = ("tests/*", "*/tests/*", "test/*", "*/test/*", "test_*.py", "*/test_*.py",
              "*_test.py", "conftest.py", "*/conftest.py")
MAX_FILE_BYTES = 1_000_000


def is_test_path(rel_path: str) -> bool:
    rel_path = rel_path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel_path, g) for g in TEST_GLOBS)


@dataclass
class Symbol:
    id: str              # "pkg/mod.py::Class.method"
    path: str            # repo-relative path, forward slashes
    name: str            # qualified name, e.g. "Class.method"
    kind: str            # "function" | "method" | "class" | "module"
    start: int           # 1-based first line (includes decorators)
    end: int             # 1-based last line
    code: str            # source shown to the model
    docstring: str = ""
    calls: set[str] = field(default_factory=set)   # bare names this symbol calls
    synthetic: bool = False  # True for class outlines / module headers (not a contiguous range)

    @property
    def short_name(self) -> str:
        return self.name.split(".")[-1]


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def _start_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decos])


def _signature_lines(node: ast.AST, lines: list[str]) -> str:
    """Decorators + `def ...:` header of a function (handles multi-line signatures)."""
    start = _start_line(node)
    body_first = node.body[0].lineno if node.body else node.lineno
    # docstring/body may share the def line in one-liners
    end = max(node.lineno, body_first - 1)
    return "\n".join(lines[start - 1:end])


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, lines: list[str]):
        self.rel_path = rel_path
        self.lines = lines
        self.symbols: list[Symbol] = []
        self.class_stack: list[str] = []

    def _segment(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1:end])

    def _add_function(self, node):
        qual = ".".join(self.class_stack + [node.name])
        kind = "method" if self.class_stack else "function"
        start, end = _start_line(node), node.end_lineno or node.lineno
        self.symbols.append(Symbol(
            id=f"{self.rel_path}::{qual}", path=self.rel_path, name=qual, kind=kind,
            start=start, end=end, code=self._segment(start, end),
            docstring=ast.get_docstring(node) or "", calls=_called_names(node),
        ))
        # Nested functions stay inside their parent's code; don't index separately.

    def visit_FunctionDef(self, node):
        self._add_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._add_function(node)

    def visit_ClassDef(self, node):
        qual = ".".join(self.class_stack + [node.name])
        start, end = _start_line(node), node.end_lineno or node.lineno
        # Outline: class header, class-level statements, and method signatures only.
        parts = [self._segment(start, node.lineno)]
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(_signature_lines(stmt, self.lines) + " ...")
            elif not isinstance(stmt, ast.ClassDef):
                parts.append(self._segment(_start_line(stmt), stmt.end_lineno or stmt.lineno))
        self.symbols.append(Symbol(
            id=f"{self.rel_path}::{qual}", path=self.rel_path, name=qual, kind="class",
            start=start, end=end, code="\n".join(parts),
            docstring=ast.get_docstring(node) or "", calls=set(), synthetic=True,
        ))
        self.class_stack.append(node.name)
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(stmt)
        self.class_stack.pop()


class RepoIndex:
    """All symbols in a repository plus lookup helpers for call-graph expansion."""

    def __init__(self, root: str, symbols: list[Symbol], files: dict[str, str]):
        self.root = os.path.abspath(root)
        self.symbols = symbols
        self.files = files
        self.by_id = {s.id: s for s in symbols}
        self.by_short_name: dict[str, list[Symbol]] = {}
        for s in symbols:
            self.by_short_name.setdefault(s.short_name, []).append(s)

    @classmethod
    def build(cls, root: str, include_tests: bool = False) -> "RepoIndex":
        root = os.path.abspath(root)
        symbols: list[Symbol] = []
        files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                if not include_tests and is_test_path(rel):
                    continue
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                files[rel] = text
                symbols.extend(cls._parse(rel, text))
        return cls(root, symbols, files)

    @staticmethod
    def _parse(rel: str, text: str) -> list[Symbol]:
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            return []
        lines = text.splitlines()
        visitor = _Visitor(rel, lines)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visitor.visit(node)
        out = visitor.symbols
        # Module header: imports, constants and other top-level statements.
        header = [
            "\n".join(lines[_start_line(n) - 1:(n.end_lineno or n.lineno)])
            for n in tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if header:
            out.append(Symbol(
                id=f"{rel}::<module>", path=rel, name="<module>", kind="module",
                start=1, end=len(lines), code="\n".join(header),
                docstring=ast.get_docstring(tree) or "", synthetic=True,
            ))
        return out

    # ---- graph helpers -------------------------------------------------
    def callees(self, sym: Symbol) -> list[Symbol]:
        out = []
        for name in sorted(sym.calls):
            for cand in self.by_short_name.get(name, []):
                if cand.id != sym.id and cand.kind in ("function", "method"):
                    out.append(cand)
        return out

    def callers(self, sym: Symbol) -> list[Symbol]:
        if sym.kind not in ("function", "method"):
            return []
        return [s for s in self.symbols if s.id != sym.id and sym.short_name in s.calls]

    def symbol_at(self, rel_path: str, line: int) -> Symbol | None:
        """Innermost non-synthetic symbol containing `line` in `rel_path`."""
        best = None
        for s in self.symbols:
            if s.path == rel_path and not s.synthetic and s.start <= line <= s.end:
                if best is None or (s.end - s.start) < (best.end - best.start):
                    best = s
        return best
