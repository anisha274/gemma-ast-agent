# Localizing Gemma 4 Agents via AST-Guided Retrieval and Self-Correction

*Retrieval finds the bug; test feedback fixes it.*

**Code:** https://github.com/anisha274/gemma-ast-agent (Apache-2.0)

**Runs:** Kaggle notebook *Localizing Gemma 4 Agents via AST-Guided Retrieval*: Version 3 (Gemma 4 E2B) and Version 4 (Gemma 4 E4B); second library: notebook *Gemma AST Agent - Second Library Experiment*, Version 1

## Abstract

Small open models can run coding agents on free or local hardware, but their short context makes it hard to show them the right code. We study two inexpensive aids for local Gemma 4 agents: AST-guided retrieval, which ranks functions, methods and class outlines (BM25 plus call-graph expansion) instead of whole files, and test-driven self-correction, which resets the repository after a failed patch and returns the failing test output with the code of traceback frames the model had not yet seen. We evaluate on 50 bugs planted in each of two libraries, boltons and more-itertools, and caught by their own test suites, under an identical ~6,000-token context budget. On boltons, AST retrieval put the buggy function in context for 88% of tasks versus 74% for whole files, but did not by itself raise single-attempt fix rates (Gemma 4 E4B: 26% vs 28%). Self-correction did: with the same retrieval, E4B rose from 26% to 42% (exact McNemar p = 0.008), above a single-attempt oracle that is always shown the buggy function (34%). On more-itertools, whose main module is 5,644 lines long, whole files showed the buggy function in only 4% of tasks and E4B fixed 2%; AST retrieval raised this to 22% (p = 0.002) and self-correction to 36% (p = 0.016), significantly above the oracle (22%). Gemma 4 E2B followed the boltons pattern at a lower level (2% to 10%). We release the agent, the bug generator and all run logs.

## 1. Introduction

Coding agents usually rely on large hosted models. An agent that runs offline, cheaply, or on private code needs a small open model, and the bottleneck is context: a few thousand tokens cannot hold a repository, so the agent must choose what to show.

A common answer is better retrieval. If the model sees the right function, the reasoning goes, it will fix it. We test that assumption with Gemma 4 E2B and E4B on Kaggle's free 2× T4 GPUs, comparing whole-file and structure-aware retrieval at the *same* token budget against an oracle that is always shown the buggy function. We then add cheap information that does not depend on retrieval: the test suite.

Our contributions:

1. **AST-guided retrieval for small context budgets:** functions, methods, class outlines and module headers ranked with identifier-aware BM25, boosted by issue names and traceback frames, and expanded along the call graph.
2. **Test-driven self-correction with traceback retrieval:** after a failed patch the model receives its previous diff, the test output, and unseen source from traceback frames.
3. **A reusable bug-generation and evaluation harness** for any pytest-tested Python repository, with paired statistics. One script regenerates every number here from the released logs.
4. **An empirical finding, replicated on two libraries.** Retrieval decides the fix rate only when files outgrow the budget; test feedback adds a significant gain on both libraries and beats perfect localization.

## 2. System

The agent is about 1,500 lines of dependency-free Python (plus `transformers` for inference): retrieve, generate a patch, apply it, run the tests, and optionally self-correct.

### 2.1 AST index

Every non-test `.py` file is parsed with Python's `ast` module. Each top-level function, method and class becomes a *symbol* with exact line ranges, its docstring, and the set of names it calls. Classes are represented by an *outline* (header, class-level statements, and method signatures), and each module by a *header* (imports and constants), so the model can see structure without whole bodies. Nested functions stay inside their parent.

### 2.2 Retrieval

All retrievers return code chunks and pack them greedily by score into the same budget (about 6,000 tokens, estimated at 4 characters per token), then present them in file order.

- **file (baseline):** BM25 [5] over whole files; the last file is truncated to fit.
- **ast_graph:** BM25 over symbols. Tokenization keeps identifiers and also splits them (`parse_url` gives `parse`, `url`), and symbol names are weighted three times. A symbol gets a bonus if the issue names it, and a larger bonus if it appears in a traceback frame. The top three symbols are then expanded to their callers and callees at half their score.
- **oracle:** the gold (mutated) function first, with the rest filled by `ast_graph`. This is an upper bound on localization, not on repair.

The prompt also lists the repository's file paths (up to 80).

### 2.3 Patch format and application

The model answers with a short root-cause note and one or more SEARCH/REPLACE blocks (as popularized by Aider [7]). Blocks are applied atomically: an exact unique match first, then a whitespace-insensitive match with re-indentation. Nothing is written if any block fails, edits a test file, escapes the repository, or produces a `SyntaxError`; each failure is explained to the model.

### 2.4 Self-correction with traceback retrieval

If the patch applies but tests fail, the agent reverts the repository and sends a new turn containing (a) the previous diff, (b) the last 2,500 characters of test output, and (c) when traceback retrieval is on, up to 1,500 tokens of source for repository symbols that appear in traceback frames and were not shown before. The model writes a new patch against the original code, for up to three attempts. Decoding is greedy with thinking disabled, so attempt 1 equals the single-attempt run with the same retriever, as our logs confirm.

## 3. Benchmark

We needed many bugs with known locations in a codebase larger than the context budget, so we built a mutation-based generator in the spirit of mutation testing [6].

**Repositories.** The main benchmark is boltons 26.2.0, a pure-Python utility library of about 18,000 lines across many modules. The second is more-itertools 11.2.1, whose code sits in two large modules (`more.py`, 5,644 lines; `recipes.py`, 1,617 lines). The second-library notebook tried click 8.5.0 and dateutil 3.0.0 first, in a fixed order, but pytest failed to start for both on Kaggle.

**Mutations.** Inside function bodies we apply one operator per bug: comparison flips (`<` to `<=`, `==` to `!=`, ...), arithmetic swaps (`+`/`-`, `//` to `/`), boolean swaps (`and`/`or`), removing `not`, and small integer constants (0, 1, 2 to n+1). Candidates are shuffled with seed 0. A mutant is kept only if the test suite fails without timing out or failing collection, with at most two bugs per function. The first 50 kept form each benchmark. boltons (from 1,482 candidates): 15 constant, 13 arithmetic, 9 comparison, 8 negation and 5 boolean bugs across 17 modules. more-itertools: 23, 15, 7, 3 and 2 (38 in `more.py`).

**Issue text.** Each issue lists only the failing test names and their short error messages from pytest, so it never reveals the mutation. The gold symbol is the mutated function, which lets us measure localization separately from repair.

**Check.** The clean repository passes before mutating, so reverting a mutation restores a passing state. Applying each known fix resolves every task on the sample library and all 50 more-itertools tasks, confirming that patch application and scoring work end to end.

## 4. Experimental setup

- **Models:** Gemma 4 E2B-it and E4B-it via Hugging Face `transformers`, float16, split across two T4 GPUs on Kaggle (free tier), greedy decoding, at most 1,024 new tokens per turn.
- **Conditions** (identical context budget):

| Condition | Retrieval | Attempts | Traceback retrieval |
|---|---|---|---|
| file | whole files | 1 | – |
| ast_graph | AST + call graph | 1 | – |
| ast_graph_sc | AST + call graph | 3 | no |
| full | AST + call graph | 3 | yes |
| oracle | gold symbol + AST | 1 | – |

- **Metrics:** *resolved* means the library's full test suite passes after the final patch. *File hit* and *symbol hit* mean the gold file, or the whole gold function, is in the initial context. We also record tokens and model time.
- **Statistics:** Wilson 95% intervals [9]; exact two-sided McNemar tests [8] on paired tasks.
- **Runtime:** 5.3 h (E2B) and 7.2 h (E4B) of wall-clock on 2× T4 for 250 runs each, and 8.6 h for four E4B conditions on more-itertools (200 runs), including setup.

## 5. Results

**Table 1. Resolve rate on 50 boltons bugs (95% CI).**

| Condition | Gemma 4 E2B | Gemma 4 E4B | Symbol hit | Tokens / task (E4B) |
|---|---|---|---|---|
| file | 4% [1–13] | 28% [17–42] | 74% | 7,765 |
| ast_graph | 2% [0–10] | 26% [16–40] | 88% | 8,561 |
| ast_graph_sc | 8% [3–19] | 38% [26–52] | 88% | 23,112 |
| **full** | **10%** [4–21] | **42%** [29–56] | 88% | 22,952 |
| oracle | 6% [2–16] | 34% [22–48] | 100% | 8,423 |

**Table 2. Paired comparisons, boltons, Gemma 4 E4B** ("only A/B" = tasks solved by one condition and not the other).

| Comparison | only A | only B | p |
|---|---|---|---|
| file vs ast_graph | 7 | 6 | 1.00 |
| ast_graph vs ast_graph_sc | 0 | 6 | **0.031** |
| ast_graph vs full | 0 | 8 | **0.008** |
| ast_graph_sc vs full | 1 | 3 | 0.63 |
| oracle vs full | 5 | 9 | 0.42 |
| file vs full | 6 | 13 | 0.17 |

**Finding 1: On boltons, structure-aware retrieval localizes better but does not fix more on its own.** At the same budget, ast_graph puts the whole buggy function in context for 88% of tasks versus 74% for whole files (file hit 100% vs 94%). Yet single-attempt fix rates are indistinguishable (E4B 26% vs 28%; E2B 2% vs 4%), and the two solve different tasks (7 vs 6). Even the oracle gains only six points over whole files (34% vs 28%, p = 0.61). When whole files fit, getting the right code into the prompt is not the binding constraint.

**Finding 2: Test-driven self-correction is what moves the fix rate.** With retrieval held fixed, two extra attempts with test feedback raise E4B from 26% to 38% (p = 0.031). Adding traceback retrieval reaches 42% (p = 0.008 vs single-attempt), and no task solved in one attempt was lost. Of the 21 tasks the full system solves, 8 needed a second or third attempt. Traceback retrieval adds a net two tasks over plain self-correction (p = 0.63).

**Finding 3: Feedback beats perfect localization.** The full system (42%) outperforms the single-attempt oracle (34%). The difference is not significant here (p = 0.42) but is on more-itertools (Finding 7): a failing assertion with concrete values tells a small model more than the correct function alone.

**Finding 4: On boltons, the end-to-end gain over the baseline is not yet significant** (42% vs 28%, p = 0.17); the whole-file baseline solved six tasks the full system did not. On more-itertools it is (36% vs 2%).

**Finding 5: Scale shifts the level, not the pattern.** E2B follows the same ordering (full 10% > ast_graph_sc 8% > oracle 6% > file 4% > ast_graph 2%), but no E2B difference is significant. E4B solves 31/50 tasks in some condition versus 8/50 for E2B.

**Cost.** Self-correction costs about 3× the tokens (22,952 vs 7,765 per task for E4B) and 2.7× the model time (143 s vs 54 s), similar on more-itertools. Because attempt 1 is unchanged, the extra cost is paid only on tasks that would otherwise fail.

### Second library: more-itertools

To test whether these findings depend on boltons, we reran four E4B conditions on more-itertools with the same generator, seed and budget.

**Table 3. Resolve rate on 50 more-itertools bugs, Gemma 4 E4B (95% CI).**

| Condition | Resolved | Symbol hit | Tokens / task |
|---|---|---|---|
| file | 2% [0–10] | 4% | 8,029 |
| ast_graph | 22% [13–35] | 90% | 8,941 |
| **full** | **36%** [24–50] | 90% | 24,773 |
| oracle | 22% [13–35] | 100% | 8,848 |

Paired tests (only A / only B): file vs ast_graph 0/10 (p = 0.002); ast_graph vs full 0/7 (p = 0.016); oracle vs full 1/8 (p = 0.039); file vs full 0/17 (p < 0.001).

**Finding 6: Retrieval matters when files outgrow the budget.** Three in four bugs here sit in a 5,644-line module, so whole-file retrieval reaches the right file (88%) but truncates it before the buggy function (symbol hit 4%) and solves one task. AST retrieval finds the function 90% of the time and matches the oracle (22% each). Finding 1 holds only when whole files fit: retrieval is a precondition for repair, not a lever once the code is in view.

**Finding 7: The self-correction result replicates.** With retrieval fixed, test feedback again adds tasks without losing any (22% to 36%); 7 of 18 solved tasks needed a later attempt. The full system now significantly beats the oracle, so Findings 2 and 3 hold on both libraries.

## 6. Analysis (boltons, E4B)

**Why single attempts fail.** Most failures are wrong fixes, not formatting. Of the 36 whole-file failures for E4B, 25 were patches that applied cleanly but failed the tests, 7 could not be applied, and 4 contained no edit block (AST retrieval: 30, 3 and 4 of 37). Across self-correcting runs, about three in four failed attempts were clean but wrong patches, exactly what a test result can correct.

**By mutation operator.** Self-correction gains are concentrated in wrong constants (whole-file 2/15, full 6/15) and arithmetic swaps (3/13 vs 6/13), while comparison flips do not improve (5/9 vs 4/9). A wrong constant rarely looks wrong in isolation, but a failing assertion with expected and actual values points straight at it.

**Unsolved tasks.** 19 of 50 tasks are solved by no condition; nine sit in stateful container and stream classes (`IndexedSet`, `SpooledStringIO`, `BufferedSocket`), where a wrong constant changes internal bookkeeping rather than an obvious return value.

**A failure pattern: "fixing" toward the bug.** In our smoke test, a bug changed one comparison in `Rectangle.overlaps` from `<` to `<=`. Both models noticed the inconsistency, then made the *other* comparisons match the bug: small models treat local consistency as a proxy for correctness, which test feedback can correct.

## 7. Limitations

- **Planted bugs are simpler than real issues.** Each is a single-operator change inside one function, caught by an existing test. The results describe controlled localization and repair, not issue resolution.
- **Two repositories, 50 tasks each.** Several boltons differences are not significant. more-itertools confirms Findings 2 and 3 but used one model and four conditions.
- **Feedback comes from the validation suite.** Self-correction runs the test command that decides success, as in fail-to-pass setups; the harness supports a separate feedback command, unused here.
- **Approximate token budget:** 4 characters per token, identical for every retriever.
- **Two small models, one setting.** Greedy decoding, float16 and a 1,024-token output limit were fixed.

## 8. Related work

SWE-bench [1] established repository-level issue resolution as an agent benchmark. Agentless [2] showed that a fixed localize-then-repair pipeline competes with open-ended agents; we follow that structure at a much smaller scale. RepoCoder [10] retrieves repository context iteratively for code completion; we instead retrieve syntax-tree units under a fixed budget and measure localization directly. Self-Debugging [3] and Reflexion [4] showed that execution feedback and self-reflection improve code generation; we separate the gains from test feedback and retrieval for small models. Our benchmark construction borrows from mutation testing [6], and our retriever applies BM25 [5] to syntax-tree units rather than files or fixed-size chunks.

## References

[1] C. E. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.  
[2] C. S. Xia et al. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489, 2024.  
[3] X. Chen et al. Teaching Large Language Models to Self-Debug. ICLR 2024.  
[4] N. Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.  
[5] S. Robertson, H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3(4), 2009.  
[6] R. A. DeMillo, R. J. Lipton, F. G. Sayward. Hints on Test Data Selection: Help for the Practicing Programmer. IEEE Computer 11(4), 1978.  
[7] P. Gauthier. Aider: AI pair programming in your terminal. https://aider.chat  
[8] Q. McNemar. Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. Psychometrika 12(2), 1947.  
[9] E. B. Wilson. Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association 22(158), 1927.  
[10] F. Zhang et al. RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation. EMNLP 2023.  

## 9. Reproducibility

Everything runs on free Kaggle GPUs with `notebooks/run_kaggle.ipynb`:

1. 16 unit and end-to-end tests run without a GPU.
2. A known-fix harness check must resolve every task.
3. `python -m ast_agent.mutate` rebuilds the 50-task benchmark deterministically (boltons 26.2.0, seed 0).
4. `python -m ast_agent.eval` runs the five conditions. Results append to `results.jsonl` per task, so interrupted runs resume.

`notebooks/run_kaggle_second_library.ipynb` repeats steps 2–4 on more-itertools. `results/kaggle_runs.json` records every task outcome from all three runs, and `python results/analyze.py` regenerates every number here. Per-attempt records, including model responses and patches, are in each Kaggle notebook's Output tab.

## 10. Conclusion

For small local Gemma 4 agents with a few thousand tokens of context, structure-aware retrieval reliably gets the buggy code in front of the model. That is decisive when files are larger than the budget (more-itertools: 2% to 22%) and neutral when they fit (boltons). Running the tests, showing the model what failed, and letting it try again adds a significant gain on both libraries (26% to 42%, 22% to 36%), beating a single-attempt oracle. The practical lesson for offline coding agents: retrieve functions, not files, then spend tokens on a short test-feedback loop. Our harness makes it cheap to test that lesson on other repositories, models and bug types.
