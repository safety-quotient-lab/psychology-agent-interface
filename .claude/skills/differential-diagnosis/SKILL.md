---
name: differential-diagnosis
description: Structured hypothesis evaluation for debugging. Score competing hypotheses on 5 dimensions (empirical support, parsimony, predictive power, chain integrity, falsifiability) to identify the most likely root cause. Modelled on the discriminator methodology from unratified (plan.md.decisions).
disable-model-invocation: true
argument-hint: "[symptom or bug description]"
---

## Problem

$ARGUMENTS

## Differential diagnosis protocol

Generate 3–6 competing hypotheses for the root cause. Score each on 5 dimensions (0–5):

| Dimension | What it measures |
|-----------|-----------------|
| **Empirical support** | Evidence in code, logs, or behaviour that supports this hypothesis |
| **Parsimony** | Simplicity — does it explain the symptom without unnecessary assumptions? |
| **Predictive power** | Does it correctly predict other observable symptoms or failure modes? |
| **Chain integrity** | Is the causal chain from root cause → symptom complete and unbroken? |
| **Falsifiability** | Can this hypothesis be definitively ruled out by a specific test? |

Score each 0–5. Total /25. Threshold for a surviving hypothesis: ≥ 20/25.

## Output format

For each hypothesis:
```
H[N]: [one-line description]
  Empirical support:  N/5 — [evidence]
  Parsimony:          N/5 — [reasoning]
  Predictive power:   N/5 — [what else it predicts]
  Chain integrity:    N/5 — [causal chain]
  Falsifiability:     N/5 — [the decisive test]
  TOTAL: N/25
```

After scoring all hypotheses:

**Surviving hypotheses (≥ 20/25):** list them
**Eliminated hypotheses:** list with reason
**Recommended investigation:** one concrete next action for each survivor

If no hypothesis scores ≥ 20/25, identify the highest-scoring one and flag what evidence is missing to confirm or eliminate it.
