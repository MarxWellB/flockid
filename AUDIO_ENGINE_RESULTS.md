# Audio Engine -- honest scope and the real bug it fixed

## What was found before building anything

`RiskEngineV1.score()` already accepted `audio_report` as a parameter
from the original design phase, and added it to `evidence_sources` -- but
never used it to compute anything. A dashboard showing
`evidence_sources: ["behavior", "audio"]` would have been claiming
multimodal fusion that didn't exist. Fixed: `audio_report` now actually
changes `health_score`, through an explicit, traceable term.

## Honest scope of the Audio Engine

No real poultry-house audio is available. Simulating a waveform and then
a realistic spectrogram would fabricate sophistication that doesn't
exist -- instead, this simulates directly at the **acoustic feature**
level (energy, a synthetic "spectral centroid," duration), the
representation any real audio pipeline produces after its first
processing stage. Same principle as the color histogram used as a
placeholder for a real embedding elsewhere in the project.

**Architectural difference kept on purpose:** audio is a ZONE-level
signal, not per-individual -- a microphone cannot isolate which bird
coughed among thousands. `AudioEngine` therefore produces zone/session
evidence and feeds the Risk Engine at the POPULATION level, not per bird,
consistent with how the rest of the engine already works.

## Classifier validation (honest, with the caveat it deserves)

```
Precision: 100.0%   Recall: 98.5%   (trained seed=1, validated seed=999)
```

This precision **should not be read as "this is how it would perform with
real audio"** -- the synthetic problem is separable by design (two
Gaussian distributions with different parameters). Real poultry-house
audio has much harder fan and motor noise, exactly the caveat already
flagged before building this. What's validated here is the PIPELINE
(features -> classifier -> event -> evidence -> score), not the
feasibility of real-world cough detection -- that requires real data,
like everything else in this project.

## Verifying the fix actually works

| Scenario | health_score | cough_rate |
|---|---|---|
| No audio connected | 0.0 | -- |
| Normal audio (1.1/min, below reference) | 0.0 | correctly triggers nothing |
| Elevated audio (7.6/min, above reference) | **30.4** | rises, with `audio_distress` as the dominant factor (25.9%) |

The score only rises when there is real distress evidence (cough rate
above the reference), not with any audio -- correct behavior, not
"blindly monotonic."

## How to run it

```bash
python audio_demo.py
```

## What's missing (honestly)

- The classifier is a 3-feature logistic regression over separable
  synthetic data -- there is no evidence this works with real audio. The
  real next step is a spectrogram classifier (CNN over a mel-spectrogram)
  trained on real field data.
- `cough_rate_reference=2.0/min` is a placeholder, not a clinically
  calibrated value.
- There is no handling of industrial background noise (fans, motors),
  already flagged as this module's real risk.
