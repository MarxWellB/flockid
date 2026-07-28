# Real benchmark: our tracker vs. real ByteTrack/BoT-SORT/OC-SORT/DeepSORT

## Finally, the full table the architecture document promised

The architecture document compared 5 trackers in a theoretical
advantages/disadvantages table. This is that comparison, run for real --
actual implementations (the `ultralytics` and `deep-sort-realtime`
libraries, not custom reimplementations), same synthetic frames, same
detector, same seed.

## Full result

| Scenario | Metric | Ours | ByteTrack | BoT-SORT | OC-SORT | DeepSORT |
|---|---|---|---|---|---|---|
| Conveyor | ID switches | 97 | **76** | **76** | 92 | 109 |
| Conveyor | Coverage | 89.5% | 86.7% | 86.7% | 86.7% | **89.0%** |
| Conveyor | IDF1 | **81.5%** | 80.0% | 80.0% | 78.0% | 77.2% |
| Random motion | ID switches | **61** | 67 | 67 | 69 | 160 |
| Random motion | Coverage | **99.3%** | 97.0% | 97.0% | 97.0% | 99.1% |
| Random motion | IDF1 | **81.6%** | 77.5% | 77.5% | 78.1% | 58.5% |

**Our tracker has the highest IDF1 in both scenarios.** DeepSORT collapses
under random motion (58.5% IDF1, more than double the ID switches of
anything else) -- consistent with its own design: it relies heavily on
appearance to resolve ambiguity, and has none available here.

## Three honest findings about why, not just the numbers

**1. BoT-SORT and ByteTrack produced IDENTICAL results.** BoT-SORT needs
real images for its two distinguishing features (appearance ReID and
camera motion compensation). Without images, it degrades to exactly
ByteTrack.

**2. DeepSORT without real appearance is DeepSORT without its main
advantage.** It was given constant embeddings (not zeros -- that caused
NaN in cosine normalization, the first bug found and fixed) so as not to
invent an appearance advantage it doesn't have here. The result (worst
IDF1 in the group, by a wide margin) is consistent with its design
depending on that more than the others.

**3. None of the 4 reference trackers received graduated confidence
scores** -- the blob detector outputs a fixed 1.0 or 0.5. ByteTrack in
particular is designed around a real confidence gradient that doesn't
exist here.

## Honest conclusion

This does not say "our tracker beats the state of the art" -- it says
that, under these specific conditions (a detector with no graduated
confidence, no images for appearance/ReID), the custom tracker competes
well and wins on the most relevant metric (IDF1) in both tested
scenarios. The pending, more valuable comparison is repeating this with
the real YOLO11 detector (which does provide graduated confidence) and
passing real frames so BoT-SORT/DeepSORT can use their design advantages
-- only then would this be conclusive in both directions.

## How to run it

```bash
pip install deep-sort-realtime setuptools
python real_tracker_benchmark.py
```
