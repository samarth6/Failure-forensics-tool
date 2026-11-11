# ffx commands

Run everything from the project root with the virtualenv active. Add `--help` after any
command to see its options, e.g. `python -m ffx eval run --help`.

The commands are listed in the order you would normally use them.

## Setup and health

| Command | What it does |
|---|---|
| `python -m ffx doctor` | Checks that Ollama is up and the pipeline model is pulled, then does a real round-trip to each configured cloud judge (Groq, Gemini). Exits with code 1 if anything is broken. |

## Corpus (the synthetic benchmark)

| Command | What it does |
|---|---|
| `python -m ffx corpus generate [--seed 7]` | Writes the 220 synthetic claim documents, each with an answer key, to `data/corpus.json`. The same seed always gives the same corpus. |
| `python -m ffx corpus stats` | Shows distributions: failure conditions, designed pressure points, categories, channels and document lengths. |

## Phases 1–2: run the traced pipeline

| Command | What it does |
|---|---|
| `python -m ffx run [--limit N] [--doc doc_005] [--mode ocr_garble] [--notes "..."]` | Runs documents through intake → extraction → classification → summarization and saves one trace per document under a new `run_id`. Repeat `--doc` or `--mode` to select several. Re-runs are answered from the replay cache. |
| `python -m ffx show <trace_id>` | Prints one trace step by step: status, tokens, attempts, output and signals. |

## Phase 3: diagnose

| Command | What it does |
|---|---|
| `python -m ffx diagnose [--run run_...] [--doc doc_005] [--no-counterfactual] [--no-judge]` | Finds each trace's root cause, verifies it with a counterfactual re-run, and scores the verdicts against corpus ground truth (precision, recall, root-step accuracy). Defaults to the latest run. |
| `python -m ffx explain <trace_id>` | Prints the full diagnosis for one trace: span assessments, ranked candidates, evidence chain, and each counterfactual's before/after checks. |

## Phase 4: the trace explorer

| Command | What it does |
|---|---|
| `python -m ffx serve [--host 127.0.0.1] [--port 8000]` | Starts the API and the explorer at http://127.0.0.1:8000, with API docs at `/docs`. |

Build the frontend once before serving, and again after any change under `web/`:

```
cd web
npm install
npm run build
```

## Phase 5: evals and analytics

| Command | What it does |
|---|---|
| `python -m ffx eval generate` | Rebuilds eval cases from every flagged trace's latest human verdict. Saving a review in the explorer already does this for that one trace. |
| `python -m ffx eval run [--label "what changed"] [--model llama3.1:8b]` | Re-runs each case's step on its frozen input with the current prompts and model. Unchanged prompts replay from cache; edited prompts call the model. |
| `python -m ffx eval matrix` | Shows each case's pass/fail history across eval runs, with its status: fixed, regressed, failing or passing. |
| `python -m ffx eval show <case_id>` | Shows one case: its assertions, bad output and expected output. |
| `python -m ffx analytics report [--run run_...]` | Shows category mix, per-step failure rates, time to root cause, accuracy against ground truth and human review, judge accuracy, and confidence calibration. |
| `python -m ffx analytics baseline <minutes> [--trace trace_...] [--note "..."]` | Records how long root-causing a trace took you by hand: the manual side of the MTTR comparison. |

## Tests

| Command | What it does |
|---|---|
| `python -m pytest -q` | Runs the full test suite. It needs no model and no network. |
