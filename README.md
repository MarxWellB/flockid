# FlockID
### Persistent visual identity + multimodal fusion for precision poultry production

> **What this is:** a computer vision platform that tracks each bird
> individually, links that identity to real production data (via nest-box
> RFID), evaluates risk/health with traceable evidence, and brings that
> information into the field through an AR interface, so a worker,
> looking at a specific bird, knows whether it should be culled, moved to
> observation, or left to keep producing.
>
> This document summarizes several weeks of design, prototyping, and
> validation work. It is not a polished demo hiding what doesn't work,
> it is, deliberately, the evidence of how it was built, including what
> failed and why.

---

## The idea, in one sentence

Fixed cameras in a poultry house can already maintain a persistent
identity per bird and link it to real production data. A worker wearing
AR glasses (or using a phone as a viewer) walks through the house and
sees, over each bird, who it is and how well it is producing, without
depending on a dashboard in an office.

## Why now, and why cage-free layers as the primary target

The full stack described above, persistent identity linked to *individual
egg production* to decide which specific bird to cull, has no business
case in broiler production, since broilers are slaughtered together at
6-7 weeks regardless of individual performance. It also has no case in
traditional small-cage systems, where egg production is already
attributed by cage position mechanically; that specific problem has been
solved for decades without vision or AI.

That said, a large part of this platform is **not** specific to egg
production and applies directly to broiler operations as well: the
Behavior Engine and Risk Engine track general activity, detect sustained
immobility, and flag disease risk regardless of housing type or
production purpose. A bird that stops moving and is not promptly removed
is a real biosecurity problem in any poultry house, a decomposing
carcass left in the litter is a disease vector and an ammonia source, and
faster detection directly improves welfare compliance and health outcomes.
That part of the system (movement/health monitoring, not individual
production attribution) is broadly applicable, broilers included, and is
a legitimate value proposition on its own even without the RFID/egg/AR
identity-fusion story.

The part that specifically requires cage-free layers is the *production
attribution* piece, linking a persistent visual identity to real egg
output via nest-box RFID, which only matters where individual birds live
long enough and produce a trackable output (eggs) worth attributing. That
narrower piece is where the market is moving: in the US, close to **50%
of egg production is already cage-free** (146.4 million cage-free hens as
of March 2026, +16% year over year, USDA), driven by state regulation
and corporate commitments (Walmart, Kroger, McDonald's). In the EU,
conventional battery cages have been banned since 2012. See
`COMPETITIVE_LANDSCAPE.md` for full sources.

## Competitive landscape (summary, see `COMPETITIVE_LANDSCAPE.md`)

| Who | What they do | What they don't do |
|---|---|---|
| FLOX (UK/US/Poland) | Real-scale vision (60M+ birds/year) | Population-level weight/uniformity, not individual identity |
| Faromatics/ChickenBoy (AGCO) | Robot + environmental sensors | No visual tracking of individuals |
| ChickTrack (Neethirajan, 2022) | YOLO+Kalman, persistent identity, close to our Phases 1-3 | Academic research, never commercialized |
| -- | -- | **No one combines visual identity + real production (RFID) + multi-camera + AR** |

---

## What's built, honestly, by validation level

### Validated with real data and real tools

| Piece | Measured result |
|---|---|
| Chicken detector (YOLO11n) | **95.6% mAP50** on dates never seen during training (real dataset, 917 images / 13 dates / 18 cameras) |
| Custom tracker (Kalman+Mahalanobis+prior) | Benchmarked against **real ByteTrack, BoT-SORT, OC-SORT, and DeepSORT**, best IDF1 in 2/2 tested scenarios, with documented caveats |
| Database + API (FastAPI/SQLite) | Running and tested with `curl` against real data from the full pipeline |
| Full pipeline on real video | Run on **9 real chicken videos** (not just synthetic), found and fixed 2 real bugs (permissive NMS, fixed confidence threshold) with measured improvement (-29% to -30% identity fragmentation) |

### Real-world tracker demo

▶️ **[Watch the tracker running on unseen video](https://drive.google.com/file/d/1RnW71MNvqV99i1gwcBb9M9LY3187QQtd/view?usp=sharing)**

This video is intentionally shown as an honest out-of-domain test, not as a
polished best-case demo. The detector was trained on approximately 1,000 images
from a different dataset and camera environment, none taken from the video shown
here.

Despite the domain mismatch, the model detects and tracks a meaningful portion
of the birds. The main failures occur during dense grouping, heavy occlusion,
scale changes, and visual overlap between nearly identical birds.

These limitations are the reason the next stage is not simply more tracker
tuning, but the creation of a farm-specific dataset through FlockTrack Copilot:
automatic proposals for simple scenes, manual bounding-box correction for dense
groups, multimodal review, active learning, and iterative retraining.

### Validated in simulation, with real documented findings

| Module | What was validated | Document |
|---|---|---|
| Identity Fusion (RFID+vision) | Engine confidence predicts accuracy: >=50% confidence -> 100% correct | `IDENTITY_FUSION_RESULTS.md` |
| Multi-camera consensus | Confirmed in a large area: coverage 57%->96%, IDF1 2.5x, **after 2 failed attempts, diagnosed** | `MULTICAM_CONSENSUS_RESULTS.md` |
| Behavior Engine | 6 behavioral signals, 2 real bugs found and fixed (event flooding, trend artifact) | `BEHAVIOR_ENGINE.md` |
| Risk Engine v1 -> v2 | From weighted sum to a real Bayesian network, handles partial evidence natively | `RISK_ENGINE_V2_RESULTS.md` |
| Audio Engine | Found and fixed a real gap: the engine accepted audio without using it | `AUDIO_ENGINE_RESULTS.md` |
| Environmental Engine | Context multiplier (not additive evidence), a deliberate architectural decision | `ENVIRONMENTAL_ENGINE_RESULTS.md` |

### Attempted, measured, and honestly discarded (this is a strength, not an omission)

- **Raw multi-camera detection fusion**: failed on the first and second attempt; the third (correct architecture + fair comparison) worked. Documented step by step.
- **RFID-based identity auto-correction** (alias merging): implemented, measured, did not improve results, code stays disabled by default, not presented as functional.
- **Adaptive Kalman tuning via NIS** (Phase 2): implemented, mathematically diagnosed for why it doesn't address the real cause of the problem.
- **Generalizing a detector from 26 images of a single clip**: 0% mAP on unseen data, the lesson that shaped how everything after it was validated.

---

## Architecture

```
Camera(s) -> Detector (YOLO11) -> Tracker (Kalman+Mahalanobis+prior)
                                        |
                    +-------------------+-------------------+
              Behavior Engine     Identity Fusion       Multicam
              (behavior)          (RFID + production)   Consensus
                    +-------------------+-------------------+
                                        |
                    Audio Engine  ->  Risk Engine (v1 + Bayesian v2)  <-  Environmental Engine
                                        |
                          Database + API (FastAPI/SQLite)
                                        |
                    Dashboard (FlockID) <-> Field app (AR/glasses/phone)
```

Full architecture document (roadmap, database schema, tracker comparison,
MLOps): see `ARCHITECTURE.md`.

---

## Three technical decisions worth asking about in an interview

1. **Why Mahalanobis distance instead of simple Euclidean distance for track association**: the Kalman filter's own uncertainty grows with time lost; Mahalanobis weights distance by that uncertainty instead of using an arbitrary fixed threshold. Measured result: 301->90 ID switches from this change alone.
2. **Why environment multiplies the risk score and audio adds to it**: environment is context (it changes how concerning another signal is), audio is direct evidence -- these are deliberately different mechanisms, not an oversimplification.
3. **Why Union-Find instead of a hand-rolled alias dictionary for cross-camera identity fusion**: a hand-rolled dictionary has real alias-chain bugs (we had them, diagnosed them); Union-Find with path compression is the correct structure for "these two things are the same, with incrementally accumulated evidence."

---

## What's missing and the real ask of this proposal

Everything above is limited by the one thing that can't be solved with
more code: **real production data at scale**. The current detector
generalizes well within its training domain (a Swiss broiler study) and
degrades, in a diagnosed way, outside of it,the same reason "person"
comes solved out of the box in any vision model (~250,000 COCO images)
and "chicken" doesn't.

**There is already a dedicated follow-up project addressing exactly this
bottleneck: FlockTrack Copilot**, a human-in-the-loop pipeline for turning
real poultry-house video into specialized detection/tracking training
data (auto-labeling, multimodal LLM review, manual correction, active
learning, iterative retraining). Building that pipeline is the concrete
next step already planned to train a real bird-identification network
under the same rigor documented throughout this project, not a vague
intention, but the direct continuation of the lesson from
`GENERALIZATION_TEST_RESULTS.md`: a detector is only as good as the
diversity of the data it was trained on, and that diversity has to come
from a real, repeatable data pipeline, not one-off datasets.

What an operating company can provide that an independent developer
cannot: **access to multiple farms, multiple batches, multiple camera
conditions, in real cage-free systems**. No need to start from zero, the
labeling pipeline (`LABELING_WORKFLOW.md`, `DATA_COLLECTION_PROTOCOL.md`)
and the base model already exist; what's missing is the data scale only a
real operation can provide.

---

## How to run it

```bash
git clone <this repo>
pip install -r requirements.txt

python main.py --frames 400 --compare              # tracker on the simulator
python real_tracker_benchmark.py                    # vs real ByteTrack/BoT-SORT/OC-SORT/DeepSORT
python fusion_demo.py                                # RFID+vision identity
python multicam_demo_v2.py                           # multi-camera consensus
python risk_v1_vs_v2_demo.py                         # weighted scoring vs Bayesian network
cd backend/database && python3 ingest_pipeline.py && cd .. && uvicorn api.main:app --reload
```

Each script prints its own metrics -- nothing here needs to be taken on
faith, it can be run and checked.

---

## About this project

Built as a demonstration of technical capability and product vision, not
as a finished product. The core idea (visual identity fusion + real
production data + field consumption via AR) is original, the base
tracking approach has academic precedent (ChickTrack), the full
combination was not found in any source researched.

Each results document (`*_RESULTS.md`) follows the same format: what was
tried, what was measured, what failed and why, what's next. That is
deliberate -- the metric for this project is not "everything works
perfectly," it is "the process for getting to the truth is rigorous and
repeatable."
