# Demo: one failure, from symptom to regression test

About 5 minutes. It follows corpus document **doc_005** through every layer of the tool,
using the traces stored in run `run_20260925_191446`.

Those traces were recorded with the *original* intake prompt, the one that made the
mistake. The prompt has since been fixed, so a fresh run of doc_005 would no longer fail.
That's why the demo uses the stored run.

## Before you start

1. **Plug in the charger.** Model commands refuse to run on battery.
2. Terminal 1: `ollama serve`
3. Terminal 2: `python -m ffx doctor` should print `OK` for Ollama. Then run
   `python -m ffx serve` and open http://127.0.0.1:8000.
4. In the top bar, pick run **run_20260925_191446**.

---

## 1. The symptom (Traces page)

**Do:** open **doc_005** in the list. Expand **Source document** in the banner.

**Show:** the source is a *home* burglary (jewellery, a laptop, two bicycles) reported by
**Nadia Osei**. Open the summarization step: the final summary says *"John Doe reports
the theft of his vehicle"*. Wrong person, wrong kind of claim.

**Say:** "You only see the problem at the end. The question is where it started."

## 2. Where it started, not where it showed (graph)

**Show:** the four-node graph.
- **Intake** is red: root cause.
- **Extraction** and **classification** are orange: they also failed on their own.
- **Summarization** is purple: propagation.

The quality numbers on each node show extraction had the *bigger* drop.

**Say:** "A 'blame the worst step' rule would pick extraction. But extraction invented
'John Doe' *because* intake had already thrown the name away. The root cause is the
earliest step that failed its own checks."

## 3. The evidence (intake step panel)

**Do:** click the **intake** node.
1. **Signals** tab: `intake_retention` failed, lost "Nadia Osei".
2. **Diff** tab: what intake received, what it produced, and what it should have
   produced. The subject line and sign-off, the only places the name appeared, are gone.

**Say:** "This check is plain code, not an LLM opinion. Every step is checked against
the input *it* received, so the evidence is reproducible."

## 4. The proof (evidence chain and counterfactual)

**Show:**
- The **Evidence chain**: origin → independent (a separate failure) → propagation →
  symptom → verification.
- The **Counterfactual** panel: the tool spliced in a corrected intake output, re-ran
  every later step, and the final summary's fidelity rose from 0.78 to 0.89 (**PARTIAL**).
  The facts intake's fix can reach got fixed; classification's separate mistake
  remained.

**Say:** "The diagnosis isn't asserted, it's tested. If fixing intake had changed
nothing downstream, intake would have been refuted and the next candidate tested."

## 5. Human review becomes a test (review panel)

**Do:** choose **Confirm diagnosis** and click **Save review**.

**Show:** the message *"Eval case … for intake"*. The confirmation is now a step-level
regression test.

## 6. The regression matrix (Evals page)

**Do:** open **Evals** and click the **doc_005 · intake** row.

**Show:**
- The matrix: **FAIL** at baseline, **ok** after the run labelled *"intake keeps subject
  + sign-off"*, with status **↑ fixed**.
- doc_009 stays **failing**: the prompt change fixed what it should and touched nothing
  else.
- In the case detail: the assertions, and the bad output beside the output the step
  produces now.

**Say:** "One line in the intake prompt fixed it, and the matrix proves it on the exact
input that broke. If a later change broke it again, this cell would turn red and the case
would read *regressed*."

### Optional live moment: catch a regression

Both prompt versions are in the replay cache, so this takes seconds.

1. In `ffx/pipeline/steps.py`, inside `intake_prompt`, comment out the two lines that
   begin `"   Keep the subject line and the sign-off` and `"they are often the only place`.
2. Run `python -m ffx eval run --label "revert intake fix"` and refresh **Evals**.
   doc_005 now reads **↓ regressed**.
3. Restore the two lines and run `python -m ffx eval run --label "restore intake fix"`.
   It reads **↑ fixed** again.

## 7. The bigger picture (Analytics page)

**Show:**
- Failure categories, and where failures start versus where they spread.
- Accuracy against ground truth and against human review.
- **Calibration:** whether the model's own 1–5 confidence predicts its failures.
- **Mean time to root cause**, automated versus your recorded manual timing.

**Say:** "Every number here says what it's measured against: corpus ground truth, the
human reviewer, or the diagnosis itself."

---

## The same story from the terminal

```
python -m ffx explain <doc_005 trace_id>   # diagnosis, candidates, evidence, counterfactual checks
python -m ffx eval matrix                  # the regression matrix
python -m ffx eval show <case_id>          # one eval case in full
python -m ffx analytics report --run run_20260925_191446
```

The trace id is shown under the banner on the Traces page.
