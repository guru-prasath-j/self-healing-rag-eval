# The Eval Layer, Explained Simply

> A plain-language guide to the evaluation + observability layer added on top of
> the Self-Healing RAG pipeline. Written so you can explain it confidently in an
> interview even to someone outside AI. Read this top-to-bottom once and you'll
> be able to answer "what did you build and why does it matter?"

---

## 1. The one-sentence version

> "On top of a RAG system that already fixes its own answers at runtime, I added
> a **report card** that automatically grades the system's answers, and a
> **gate** that refuses to let new code ship if the grades get worse."

If you only remember one thing, remember that.

---

## 2. A story without any jargon

Imagine you run a small help desk. A new employee (the AI) answers customer
questions by first looking things up in a binder of company documents.

The original project — *self-healing RAG* — is like teaching that employee to
**double-check their own answer**: "Wait, did I actually find that in the binder,
or did I make it up? If I'm not sure, let me search again with better keywords."
That's great. But it only judges **one answer at a time, in the moment**.

Here's the problem that this new layer solves: every time you "improve" the
employee's training, how do you *know* you actually made them better and not
worse? Maybe you sped them up but now they guess more. You can't eyeball every
answer forever.

So you do what a good manager does:

1. **You write an exam.** A fixed list of questions where you already know the
   correct answers. (We call this the *golden set*.)
2. **After any change, the employee re-sits the exam.** You grade every answer.
3. **You keep a scorecard over time** so you can see if today's version is
   better or worse than last week's.
4. **You make a rule:** "No change goes live if the exam score drops." If a new
   idea makes the score fall, it's automatically rejected.

That manager's exam-and-rule system is exactly what this layer is. In industry
language it's called **eval-driven development** and **observability**.

---

## 3. The four pieces, in order

### Piece 1 — The golden set (the exam)
A file, `eval/golden_set.json`, holding ~10 questions and the *reference answer*
for each (the answer we consider correct, taken straight from the source docs).
One question is deliberately **unanswerable** — to check the system honestly says
"I don't know" instead of inventing something.

### Piece 2 — The grader (the metrics)
`eval/metrics.py` grades each answer on three things, each scored 0 to 1:

| Score | Plain meaning | Why it matters |
|---|---|---|
| **answer_similarity** | How close is the answer to the correct one? | Is it *right*? |
| **faithfulness** | Is the answer actually backed by the documents it pulled? | Did it *hallucinate*? |
| **context_relevance** | Were the documents it pulled even related to the question? | Is *retrieval* working? |

The clever, cheap trick: instead of paying another AI to be the judge, we turn
both texts into number-fingerprints (embeddings) using the **same model the
pipeline already uses** and measure how close those fingerprints are. Free,
fast, deterministic. (You *can* bolt on heavier "LLM-as-judge" graders like
RAGAS or DeepEval — that's in `requirements-eval.txt` — but it's optional.)

We also record three operational numbers per question: how many self-healing
**retries** it needed, how many **milliseconds** it took, and whether the
pipeline said it was **grounded**.

### Piece 3 — The gate (the rule)
`eval/thresholds.json` lists the minimum acceptable scores. `eval/evaluate.py`
runs the whole exam, averages the grades, and compares them to those minimums.
If anything is below the line, the script **exits with an error code**. That
error is the "no, you can't ship this" signal.

### Piece 4 — The CI workflow (the automatic enforcer)
`.github/workflows/eval.yml` tells GitHub: every time someone proposes a code
change, rebuild everything, run the exam, and **if the gate fails, mark the
whole pull request as failed** so it can't be merged. Nobody has to remember to
run it — it's automatic. That's "CI" (continuous integration).

### Bonus — Observability (the security camera)
`src/tracing.py` optionally connects to **Langfuse**. When switched on (just add
keys), every single LLM call — generate, critique, reformulate — is recorded as
a timeline you can click through later: what prompt went in, what came out, how
long it took, what it cost. It's a no-op if you don't add keys, so it never gets
in the way.

---

## 4. How a single change flows through it

```
You tweak a prompt  ──►  open a Pull Request
                              │
                              ▼
                 GitHub Actions wakes up automatically
                              │
                 rebuild index → run unit tests → run the exam
                              │
                    grade every answer, average the scores
                              │
              ┌───────────────┴───────────────┐
        scores OK                         scores dropped
              │                                │
        ✅ PR can merge                  ❌ PR blocked, build red
```

---

## 5. Why this is the impressive part (say this in interviews)

Most people's portfolios stop at "I built a RAG demo." This says something much
stronger: **"I treat answer quality as something you measure and protect, like
a test suite."** The three things that signal seniority:

1. **Regression-gated evals in CI** — you can't merge a change that lowers
   quality. This is rare in junior portfolios.
2. **Cheap, local metrics** — you got a real quality signal without an expensive
   LLM judge, and you know the trade-off (lexical/semantic proxy vs. a true
   judge) and how to upgrade.
3. **Observability + cost/latency tracking** — you think about running this in
   production, not just making it work once.

---

## 6. Interview Q&A (rehearse these)

**Q: What does the eval layer actually do?**
It grades the pipeline's answers against a fixed golden set on every code change,
and fails the build if average quality drops below set thresholds. So quality can
only stay flat or go up over time, never silently degrade.

**Q: How do you grade an answer without a human?**
For each answer I compute three embedding-based scores: similarity to the
reference answer (correctness proxy), similarity to the retrieved context
(faithfulness / anti-hallucination), and question-to-context similarity
(retrieval quality). I use the same MiniLM embedder the pipeline uses, so it's
free and deterministic. Heavier LLM-judge metrics (RAGAS, DeepEval) are an
optional add-on.

**Q: Why not just use RAGAS/DeepEval for everything?**
They need an LLM judge, which costs money and adds latency and nondeterminism to
CI. I wanted the gate to be fast, free, and repeatable, so the default metrics
are local. The heavier graders are available when I want a deeper read.

**Q: What's the difference between this and the self-healing loop?**
The self-healing loop is *runtime* — it fixes one answer while serving it. The
eval layer is *development-time* — it proves, across many questions and across
versions, that my changes don't make things worse. One is resilience; the other
is measurement and regression prevention. They complement each other: the eval
layer is what tells me whether the self-healing loop is actually helping and
what it costs in extra retries and latency.

**Q: How does the CI gate physically block a bad change?**
`eval/evaluate.py` exits with a non-zero status code when a threshold is
breached. GitHub Actions treats a non-zero exit as a failed job, which marks the
pull request red so it can't be merged.

**Q: What do you track besides correctness?**
Per question I log the number of self-healing retries, the latency in
milliseconds, and the grounded/not-grounded verdict; I aggregate average and p95
latency and the average retry count. That's how I'd catch a change that improves
accuracy but doubles cost or latency.

**Q: What is Langfuse doing here?**
It's optional observability. If keys are set, every LLM call becomes a traced
span — prompt, output, timing, cost — and the eval harness attaches the quality
scores to each run, so I can watch quality and cost trends over time in a
dashboard. Without keys it's a complete no-op.

**Q: How would you extend this?**
Grow the golden set, add LLM-judge metrics for nuance, track per-document
retrieval recall, store historical results to plot trends, and add a "canary"
comparison that diffs the PR's scores against main rather than fixed thresholds.

---

## 7. Words to own

eval-driven development, golden set, regression gate, CI/CD, faithfulness,
hallucination, retrieval relevance, embeddings, cosine similarity, observability,
tracing, p95 latency, threshold, non-zero exit code, LLM-as-judge.
