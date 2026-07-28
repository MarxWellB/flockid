# Environmental Engine -- context, not evidence (and why that matters)

## The same gap Audio had, found the same way

`environmental_report` was also accepted and declared in
`evidence_sources` without being used in the calculation. Fixed -- but
with a mechanism **deliberately different** from Audio's.

## Why it's different from Audio (this is the important part)

The original architecture document was explicit: environment **is not
direct health evidence** -- an ammonia spike is not a disease event. It
is context that changes how concerning another signal is. So:

- **Audio**: an ADDITIVE term on `health_score` -- a cough event is direct
  evidence, added like any other signal.
- **Environment**: a MULTIPLIER on the already-computed `risk_score` --
  the same observed behavior is more concerning in a house with
  ammonia/heat out of range, not because the environment itself is the
  problem, but because it changes how the behavior should be interpreted.

This architectural distinction was in the original design from the start
-- implementing it the same way as Audio would have been simpler but
technically incorrect relative to what the design called for.

## Verification (same behavior, different environment)

| Scenario | risk_score | stress_index |
|---|---|---|
| No environment connected | 18.0 | -- |
| Optimal environment | 18.0 | 0.0 |
| Stressed environment (27C, NH3=26ppm, CO2=4048ppm) | **21.4** | 0.479 |

The score rises purely from environmental context, with identical
behavior -- confirms the multiplier genuinely works, not just declared.

## Honest scope

- No real sensors are connected -- a time series is simulated within
  general poultry-management "optimal" reference ranges, not values
  clinically calibrated for a specific client.
- The multiplier has a deliberate cap (+40% max) -- environment modulates,
  it cannot dominate the score on its own, consistent with not being
  direct evidence.
- Per-variable weights (ammonia and temperature weighted higher than
  humidity/CO2) are engineering judgment based on general management
  literature, not a specific cited clinical source -- this would need
  validation with a real poultry veterinarian/nutritionist before
  production.

## How to run it

```python
from vision.environmental.environmental_engine import EnvironmentalSimulator, EnvironmentalEngine
from vision.risk.engine import RiskEngineV1
# see the verification block in this document for the full example
```
