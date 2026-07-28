# Competitive landscape -- who does what (with sources)

Researched to position the proposal precisely: what already exists, what
is academic research never commercialized, and what does not exist
anywhere yet.

## Established commercial platforms

**Big Dutchman, Cargill, MTech Systems** -- established poultry technology
providers, generally environment+dashboard first, not vision-tracking
first. Originally mentioned as target-market reference points.

## Commercial computer vision in real production

**FLOX** (UK, US, Poland) -- monitors 60M+ birds per year, covers 10% of
UK poultry production, targeting 1 billion birds within 3 years. Cameras
scan the house, estimate weight without handling birds, add environmental
sensors, web dashboard covering up to 6 houses at once. Focus: **flock
weight and uniformity at the population level**, not persistent individual
identity with attributed production.
Source: https://www.wattagnet.com/poultry-future/poultry-tech-summit-news/news/15769370/computer-vision-could-move-poultry-monitoring-onto-desktop and https://flox.ai/

**Faromatics / ChickenBoy** (Spain, acquired by AGCO in 2021) -- a
ceiling-mounted robot that travels on rails, with temperature, air
quality, light, and sound sensors plus AI for health/welfare risk. **Does
not do visual tracking of individuals** -- a mobile-robot-with-sensors
approach, not fixed cameras with persistent identity.
Source: https://investors.agcocorp.com/news-releases/news-release-details/agco-acquires-faromatics-precision-livestock-farming-company

**TARGAN** -- computer vision for feather-based sexing (gender
identification), a different use case from this project.
Source: https://www.wattagnet.com/poultry-future/new-technologies/news/15828097/top-10-poultry-technology-trends-of-2026-so-far

## Academic research (not commercialized) -- the closest technical precedent

**ChickTrack** (Neethirajan, 2022, published in ScienceDirect/Computers
and Electronics in Agriculture) -- YOLOv5 + Kalman filter for persistent
individual chicken identity; "the same number stays with the bird even
out of camera view or under poor lighting." Technically, this is close to
exactly this project's Phases 1-3. **It is academic research, never
commercialized** ("yet to be made commercially available" per 2021 press
coverage).
Sources: https://www.sciencedirect.com/science/article/pii/S0263224122001154 and
https://www.wattagnet.com/poultry-future/new-technologies/article/15534419/new-ai-monitoring-system-offers-promise-to-poultry-sector

**"Chicken Tracking and Individual Bird Activity Monitoring Using the
BoT-SORT Algorithm"** (MDPI, 2023) -- 98.5% mAP on detection, BoT-SORT to
maintain identity during tracking. Confirms that benchmarking against
BoT-SORT (already done in this project, see
`REAL_TRACKER_BENCHMARK_RESULTS.md`) is exactly the methodology the
research community uses.
Source: https://www.mdpi.com/2624-7402/5/4/104

**AR smart glasses in livestock farming (GlassUp F4)** (MDPI, 2019) -- a
lab and field study of AR glasses as a Precision Livestock Farming tool.
Generic to livestock, not chicken-specific, not connected to individual
production data.
Source: https://www.mdpi.com/2076-2615/9/11/903

## What was not found anywhere -- the real differentiator

1. **RFID+vision fusion with calibrated confidence** to attribute real
   production (eggs) to a persistent visual identity -- not found in any
   source researched.
2. **Multi-camera consensus explicitly validated for large-area coverage**
   with this project's methodology (independent per-camera tracker +
   trajectory-based association) -- FLOX uses multiple cameras but without
   this documented mechanism.
3. **AR/wearable field interface connected to the same individual
   identity + production backend** -- not found combined anywhere; the
   GlassUp study is generic livestock with no such connection.

## Honest conclusion for the pitch

Base tracking (Phases 1-3) has direct academic precedent (ChickTrack) --
it should not be presented as unprecedented. What is genuinely
differentiated is the **combination**: multimodal fusion with calibrated
confidence (RFID+audio+environment+vision), validated multi-camera
consensus, and the field-consumption layer (AR) connected to all of it.
That is the defensible argument, backed by sources.
