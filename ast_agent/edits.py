"""Parse SEARCH/REPLACE edit blocks from model output and apply them safely.

Format the model is asked to produce:

    path/to/file.py
    <<<<<<< SEARCH
    exact old lines
    =======
    new lines
    >>>>>>> REPLACE

Application is atomic: if any block fails (not found, ambiguous, protected file,
or the result doesn't parse), nothing is written and every error is reported back
to the model for self-correction.
"""
from __future__ import annotations

import ast
import difflib
import fnmatch
import os
import re
from dataclasses import dataclass, field

from .indexer import TEST_GLOBS

_SEARCH = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^\s*={5,9}\s*$")
_REPLACE = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$")
_FENCE = re.compile(r"^\s*```")


@dataclass
class Edit:
    path: str
    search: str
    replace: str


@dataclass
class ApplyResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)  # path -> (old, new)

    def diff(self) -> str:
        out = []
        for path, (old, new) in sorted(self.changed.items()):
            out.extend(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}",
            ))
        return "".join(out)


def _clean_path(line: str) -> str:
    p = line.strip().strip("`*#:").strip()
    p = re.sub(r"^(file|path)\s*:\s*", "", p, flags=re.IGNORECASE).strip("`'\" ")
    return p


def parse_edits(text: str) -> list[Edit]:
    # Drop any reasoning section some chat templates emit before the answer.
    for tag in ("</think>", "</thinking>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    lines = text.splitlines()
    edits: list[Edit] = []
    i = 0
    while i < len(lines):
        if not _SEARCH.match(lines[i]):
            i += 1
            continue
        # Path = nearest previous non-empty, non-fence line.
        path, j = "", i - 1
        while j >= 0:
            if lines[j].strip() and not _FENCE.match(lines[j]):
                path = _clean_path(lines[j])
                break
            j -= 1
        search, replace, k = [], [], i + 1
        while k < len(lines) and not _DIVIDER.match(lines[k]):
            search.append(lines[k])
            k += 1
        k += 1
        while k < len(lines) and not _REPLACE.match(lines[k]):
            replace.append(lines[k])
            k += 1
        if k < len(lines) and path:
            edits.append(Edit(path, "\n".join(search), "\n".join(replace)))
        i = k + 1
    return edits


def _find_unique(haystack: str, needle: str) -> tuple[int, str | None]:
    count = haystack.count(needle)
    if count == 1:
        return haystack.index(needle), None
    return -1, ("not found" if count == 0 else f"matches {count} places; add more surrounding lines")


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _fuzzy_replace(content: str, search: str, replace: str) -> tuple[str | None, str | None]:
    """Match ignoring indentation/trailing spaces; re-indent the replacement to fit."""
    f_lines = content.split("\n")
    s_lines = search.split("\n")
    while s_lines and not s_lines[0].strip():
        s_lines.pop(0)
    while s_lines and not s_lines[-1].strip():
        s_lines.pop()
    if not s_lines:
        return None, "empty SEARCH block"
    key = [ln.strip() for ln in s_lines]
    hits = [i for i in range(len(f_lines) - len(key) + 1)
            if [ln.strip() for ln in f_lines[i:i + len(key)]] == key]
    if len(hits) != 1:
        return None, "not found" if not hits else f"matches {len(hits)} places; add more surrounding lines"
    i = hits[0]
    file_ind, search_ind = _indent(f_lines[i]), _indent(s_lines[0])
    new_lines = []
    for ln in replace.split("\n"):
        if not ln.strip():
            new_lines.append("")
        elif ln.startswith(search_ind):
            new_lines.append(file_ind + ln[len(search_ind):])
        else:
            new_lines.append(file_ind + ln.lstrip())
    return "\n".join(f_lines[:i] + new_lines + f_lines[i + len(key):]), None


def _is_protected(rel: str, protected: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel, g) for g in protected)


def apply_edits(root: str, edits: list[Edit], protected: tuple[str, ...] = TEST_GLOBS,
                write: bool = True) -> ApplyResult:
    root = os.path.abspath(root)
    result = ApplyResult(ok=False)
    pending: dict[str, str] = {}     # rel path -> new content
    originals: dict[str, str] = {}
    if not edits:
        result.errors.append("no edit blocks found in the response")
        return result

    for n, e in enumerate(edits, 1):
        rel = e.path.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        full = os.path.abspath(os.path.join(root, rel))
        tag = f"block {n} ({rel})"
        if os.path.isabs(rel) or not full.startswith(root + os.sep):
            result.errors.append(f"{tag}: path must be relative and inside the repository")
            continue
        if _is_protected(rel, protected):
            result.errors.append(f"{tag}: test files are read-only; fix the library code instead")
            continue
        if rel in pending:
            content = pending[rel]
        elif os.path.exists(full):
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            originals[rel] = content
        elif not e.search.strip():
            originals[rel] = ""
            pending[rel] = e.replace + ("\n" if e.replace and not e.replace.endswith("\n") else "")
            continue
        else:
            result.errors.append(f"{tag}: file does not exist")
            continue

        if not e.search.strip():
            result.errors.append(f"{tag}: SEARCH is empty but the file already exists")
            continue
        pos, err = _find_unique(content, e.search)
        if err is None:
            new = content[:pos] + e.replace + content[pos + len(e.search):]
        else:
            new, err2 = _fuzzy_replace(content, e.search, e.replace)
            if new is None:
                result.errors.append(f"{tag}: SEARCH text {err2}; copy the exact current lines")
                continue
        pending[rel] = new

    for rel, new in pending.items():
        if rel.endswith(".py"):
            try:
                ast.parse(new)
            except SyntaxError as exc:
                result.errors.append(f"{rel}: patch produces a SyntaxError at line {exc.lineno}: {exc.msg}")

    if result.errors:
        return result
    for rel, new in pending.items():
        if new == originals.get(rel):
            continue
        result.changed[rel] = (originals.get(rel, ""), new)
        if write:
            full = os.path.join(root, rel)
            os.makedirs(os.path.dirname(full) or root, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(new)
    if not result.changed:
        result.errors.append("edits made no change to any file")
        return result
    result.ok = True
    return result


def revert(root: str, result: ApplyResult) -> None:
    """Undo a successful apply_edits (restores originals, removes newly created files)."""
    for rel, (old, _new) in result.changed.items():
        full = os.path.join(root, rel)
        if old == "" and os.path.exists(full):
            os.remove(full)
        else:
            with open(full, "w", encoding="utf-8") as f:
                f.write(old)
