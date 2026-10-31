# Localizing Gemma 4 Agents via AST-Guided Retrieval and Self-Correction

Code for our entry to the Google – Gemma 4 Developer Agent Paper Track (Kaggle, 2026).

A small, dependency-free bug-fixing agent for **local** Gemma 4 models:

1. **AST indexing** – every function, method and class in the repo becomes a retrievable symbol with exact line ranges and call edges.
2. **AST-guided retrieval** – BM25 over symbols (not whole files), boosted by symbol names in the issue and frames in tracebacks, optionally expanded along the call graph.
3. **Patch generation** – the model answers with SEARCH/REPLACE blocks, applied atomically (syntax-checked, tests read-only).
4. **Self-correction** – on failing tests the repo is reset and the model gets its previous diff, the test output, and the code of traceback symbols it hadn't seen.

Every component can be switched off, so each one gets its own row in the results table.

**Paper:** [`paper/writeup.md`](paper/writeup.md)

## Results

50 bugs planted in [boltons](https://github.com/mahmoud/boltons) 26.2.0, identical ~6,000-token context budget, Kaggle 2× T4:

| Condition | Gemma 4 E2B | Gemma 4 E4B | Buggy function in context |
|---|---|---|---|
| file (whole-file BM25 baseline) | 4% | 28% | 74% |
| ast_graph (AST + call graph) | 2% | 26% | 88% |
| ast_graph_sc (+ self-correction) | 8% | 38% | 88% |
| **full** (+ traceback retrieval) | **10%** | **42%** | 88% |
| oracle (gold function shown, 1 attempt) | 6% | 34% | 100% |

AST retrieval localizes better, but test-driven self-correction is what raises the fix rate (E4B: 26% → 42%, exact McNemar p = 0.008).
Every task outcome is in [`results/kaggle_runs.json`](results/kaggle_runs.json); `python results/analyze.py` regenerates all numbers ([`results/analysis.md`](results/analysis.md)).

## Layout

```
ast_agent/
  indexer.py    AST symbols + call graph
  retrieval.py  file / ast / ast_graph / oracle retrievers (same token budget)
  edits.py      SEARCH/REPLACE parsing and atomic application
  agent.py      retrieve -> patch -> test -> self-correct loop
  models.py     OpenAI-compatible (vLLM, Ollama, LM Studio), transformers, scripted, gold-patch
  mutate.py     benchmark generator: plants bugs in real repos, keeps those the tests catch
  eval.py       runs tasks x conditions, writes results.jsonl + summary.md table
benchmarks/sample/   tiny library + 17 generated tasks (smoke test)
tests/               unit + end-to-end tests (no GPU needed)
```

## Quick start (no GPU)

```bash
pip install pytest
python -m pytest -q tests
# Harness check: apply the known fix to every task; all must resolve.
python -m ast_agent.eval --tasks benchmarks/sample/tasks.jsonl --backend gold --conditions full --out results/gold
```

## Running with Gemma 4

**Kaggle notebook (what the paper used).** Import `notebooks/run_kaggle.ipynb`, set Accelerator to GPU T4 x2 and Internet On, add the Gemma 4 model (framework Transformers, `gemma-4-e2b-it` or `gemma-4-e4b-it`) as an input, and choose Save & Run All. The experiment cell runs:

```bash
python -m ast_agent.eval --tasks benchmarks/boltons/tasks.jsonl --backend hf --model "$MODEL_PATH" \
  --dtype float16 --conditions file,ast_graph,ast_graph_sc,full,oracle --max-new-tokens 1024 --out results/e4b
```

**Any OpenAI-compatible server** (vLLM, Ollama, LM Studio, llama.cpp): `--backend openai --base-url <url>/v1 --model <model-id>`.

**Mac / laptop development with Ollama (small models)**

```bash
ollama pull <gemma-4-small-tag>
python -m ast_agent.eval --tasks benchmarks/sample/tasks.jsonl --backend openai \
  --base-url http://localhost:11434/v1 --model <gemma-4-small-tag> --limit 3 --out results/dev
```

**In-process transformers (Colab)**: `--backend hf --model <hf-model-id>` (needs `torch`, `transformers`, `accelerate`).

Runs are resumable: if the session dies, re-run the same command and finished (task, condition) pairs are skipped.

## Building a real benchmark

The sample library is tiny: the whole repo fits in the context budget, so every retriever looks perfect. Real results need real repositories that are larger than the budget.

The paper's benchmark is rebuilt deterministically with:

```bash
git clone -q --depth 1 --branch 26.2.0 https://github.com/mahmoud/boltons.git benchmarks/repos/boltons
pip install -e benchmarks/repos/boltons
python -m ast_agent.mutate --repo benchmarks/repos/boltons \
  --test-cmd "python -m pytest -q --tb=short -p no:cacheprovider" \
  --out benchmarks/boltons --max-tasks 50 --seed 0
```

The same commands work for any pure-Python library with a fast pytest suite.

Mutations are single-operator changes (comparison, arithmetic, boolean, `not`, small constants) inside functions. A mutant is kept only if the test suite fails. The issue text is built from the failing test names and messages only, so it never reveals the mutation. The gold symbol is the mutated function, which lets us measure localization separately from repair.

## Conditions

| Condition | Retrieval | Attempts | Traceback retrieval |
|---|---|---|---|
| `file` (baseline) | whole files, BM25 | 1 | – |
| `ast` | AST symbols | 1 | – |
| `ast_graph` | AST + call graph | 1 | – |
| `file_sc` | whole files | 3 | no |
| `ast_graph_sc` | AST + call graph | 3 | no |
| `full` | AST + call graph | 3 | yes |
| `oracle` | gold symbol + AST + graph | 1 | – |

All conditions share the same context budget (`--context-budget`, default 6000 approx. tokens).

## Output

- `results.jsonl` – one record per (task, condition): resolved, patch, retrieved symbols, file/symbol hit, tokens, attempts, timings, model responses.
- `summary.md` – the results table (resolve rate with Wilson 95% CI, exact McNemar p-value vs baseline, localization, cost).
- `run_meta.json` – model, backend, budgets, Python version, git commit.

## Limitations

- Mutation bugs are simpler than real issues; report results on them as such.
- The agent sees the same test command used for validation (as with "fail-to-pass" feedback). Set a separate `feedback_cmd` per task to hide validation tests.
- Token counts fall back to a ~4 chars/token estimate when the server reports no usage.

## License

Apache License 2.0 – see `LICENSE`.
