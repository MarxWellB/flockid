# Risk Engine v2 -- a real Bayesian network (pgmpy)

## Why this, and why now

The architecture document always said v1 (weighted sum) was a starting
point for when there weren't enough evidence sources yet, and that it
should migrate to a Bayesian network once there were several signals to
combine in a principled way. Three now exist (Behavior, Audio,
Environmental) -- this is the right moment to do it, as planned from the
start, not an afterthought.

## The real difference (not just "more sophisticated")

**v1** combines evidence with a hand-tuned weighted sum, and needs special
branching (`if audio_report is not None: ...`) to handle partial evidence
at every point in the code.

**v2** uses actual Bayes' theorem: each signal has a conditional
probability given a latent health state, and inference
(`VariableElimination` from pgmpy) combines everything with consistent
probability math. Missing evidence is **marginalized automatically** -- a
structural property of the network, not a code patch.

Network structure:
```
EnvironmentalStress -> Health -> Isolation
                               -> LowActivity
                               -> AudioDistress
```
Environment remains CONTEXT (it modifies `Health`'s prior), not direct
evidence -- the same design decision as v1, now expressed as network
structure instead of an ad-hoc multiplier.

## Verification (same risk progression, both engines)

| Scenario | v1 (risk_score) | v2 (P(at_risk)) |
|---|---|---|
| Healthy, no audio/environment | 3.0 | 4.2% |
| Healthy + normal audio | 2.8 | 1.9% |
| At-risk, no audio/environment | 52.3 | 88.9% |
| At-risk + elevated audio | 64.1 | 98.4% |
| At-risk + elevated audio + bad environment | 84.7 | 99.4% |
| At-risk + bad environment ONLY | 69.1 | 95.4% |

Both engines rise monotonically and consistently with more risk evidence
-- v2 doesn't give a "better" number, it reaches the same kind of
conclusion through a mathematically correct mechanism instead of
invented weights.

## Honest scope (this doesn't change just by using a Bayesian network)

The conditional probability tables are still **engineering judgment, not
calibrated with real data** -- exactly the same limitation as v1's
weights. Changing the combination mechanism does not solve the lack of
labeled data; that's still pending real confirmed batches, as noted for
every module in this project. v2's values look more "confident" (88.9%,
98.4%) than v1's on a 0-100 scale -- that's an artifact of how strong
conditional probabilities combine (multiplying likelihood ratios), not
evidence that v2 is more accurate.

## How to run it

```bash
pip install pgmpy
python risk_v1_vs_v2_demo.py
```
