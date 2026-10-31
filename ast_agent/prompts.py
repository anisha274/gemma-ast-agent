"""Prompt templates. Kept in one place so the paper can quote them exactly."""
from __future__ import annotations

from .retrieval import ContextChunk

INSTRUCTIONS = """You are a software engineering agent. Fix the issue below by editing the repository.

Reply with a short root-cause explanation (at most 3 sentences), then one or more edit blocks in exactly this format:

path/to/file.py
<<<<<<< SEARCH
exact lines copied from the current code
=======
replacement lines
>>>>>>> REPLACE

Rules:
- SEARCH must match the current file exactly (including indentation) and occur only once; include 1-3 surrounding lines if needed.
- Edit library code only. Test files are read-only.
- Make the smallest change that fixes the root cause."""


def format_chunks(chunks: list[ContextChunk]) -> str:
    if not chunks:
        return "(no code retrieved)"
    parts = []
    for c in chunks:
        where = f"lines {c.start}-{c.end}" if c.start else "outline, not contiguous"
        parts.append(f"### {c.path} ({where}) - {c.label}\n```python\n{c.code}\n```")
    return "\n\n".join(parts)


def format_file_list(paths: list[str], limit: int = 80) -> str:
    shown = paths[:limit]
    more = f"\n... and {len(paths) - limit} more" if len(paths) > limit else ""
    return "\n".join(shown) + more


def initial_prompt(issue: str, chunks: list[ContextChunk], file_paths: list[str]) -> str:
    return (f"{INSTRUCTIONS}\n\n## Issue\n{issue.strip()}\n\n"
            f"## Repository files\n{format_file_list(file_paths)}\n\n"
            f"## Relevant code\n{format_chunks(chunks)}")


def no_edits_feedback() -> str:
    return ("Your reply contained no valid edit blocks. Reply again using exactly the "
            "path / <<<<<<< SEARCH / ======= / >>>>>>> REPLACE format.")


def apply_error_feedback(errors: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errors)
    return (f"Your edits could not be applied, so the repository is unchanged:\n{bullet}\n\n"
            "Reply with corrected edit blocks. Copy SEARCH lines exactly from the code shown earlier.")


def test_failure_feedback(diff: str, test_output: str, extra: list[ContextChunk]) -> str:
    extra_txt = ""
    if extra:
        extra_txt = ("\n\n## Code from the failing traceback (not shown before)\n"
                     + format_chunks(extra))
    return (f"Your patch applied cleanly but the tests still fail.\n\n"
            f"## Your previous patch\n```diff\n{diff.strip()}\n```\n\n"
            f"## Test output (tail)\n```\n{test_output.strip()}\n```{extra_txt}\n\n"
            "The repository has been reset to its ORIGINAL state. Reply with a complete, corrected "
            "set of edit blocks written against the original code.")
