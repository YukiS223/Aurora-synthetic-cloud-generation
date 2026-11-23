# Synthetic clouds for all-sky aurora footage

Adds realistic, temporally consistent, physically lit clouds to clear all-sky
aurora clips (KHO Svalbard, 608x608 fisheye, about 1 real second per frame,
played at 25 fps). Each run produces a (clear, cloudy, cloud-opacity mask)
triplet for training cloud-removal models.

The current pipeline is `cloudvol/`: a volumetric cloud renderer whose
parameters are measured from the 87 real cloudy clips in `Cloud_footgae/`.

## Current result

Frame 2500 of `outputs_cloudvol/22112017_195302_s1_demo.mp4` — clear input on
the left, clouded output on the right:

![cloudvol demo, clear | cloudy](docs/images/demo_cloudvol_s1.jpg)

For reference, frame 700 of the previous pipeline's last demo,
`outputs_v1_old/demo_progress_round10.mp4` (a different clear clip,
`18122017_000058`):

![round 10 demo, clear | cloudy](docs/images/demo_round10.jpg)

And the same clear frame through both, so the difference is like for like
(`18122017_000058`, source frame 2100; cloudvol with its `overcast` regime,
seed 3):

![same frame: clear | round 10 | cloudvol](docs/images/same_frame_round10_vs_cloudvol.jpg)

Round 10 reads as a milky filter over the lens: one smooth sheet, one tint,
terrain and instrument dome painted over. The current output has cloud bodies
with gaps and stars between them, aurora diffusing through thinner parts,
colour that changes with what is lighting the cloud, and an untouched
foreground.

## How the current pipeline works

The earlier pipeline drew a 2-D opacity map and blended the frame toward a
glow colour, so opacity and colour were the same quantity and "more cloud"
could only mean "more of the same tint". Real clouds in the reference clips do
not behave that way: thin cloud is often *brighter* than the clear sky beside
it while stars still show; thick cloud hides the aurora but glows in the colour
of the whole sky; the same cloud is yellow-green over the ground lights and
green under the aurora; gaps close toward the horizon.

`cloudvol` therefore models the cloud as a 3-D volume and computes light
transport through it.

1. **Scene statics** (`scene.py`). The sky mask comes from temporally
   persistent edges pooled over same-camera clips, so terrain, the instrument
   dome and the lens rim are excluded and clouds never paint over the
   foreground. Sensor hot pixels and the grain level are also measured.
2. **Cloud volume** (`render.py::density`, `textures.py`). Each layer is a
   thresholded 3-D Perlin-Worley-style field gated by an evolving coverage map
   and a height profile, with detail noise eroding the edges — the density
   modelling scheme of real-time volumetric cloudscapes [3][4], built on
   cellular noise [5] — defined in world kilometres and advected by wind. Every
   camera ray samples it on a stack of altitude planes through the measured
   fisheye geometry (curved Earth), with distance-based detail fading so the
   horizon does not shimmer.
3. **Light propagation** (`render.py::transport`). Optical depth is accumulated
   through the volume along three families of paths — down from the sky, up
   from the snow, and slanted from each ground light — each as a chain of
   image warps between neighbouring planes. Every sample gets its own
   illumination and shadowing.
4. **The clear frame is the light source** (`render.py::sky_light`). What a
   cloud scatters toward the camera is the clear frame itself, blurred more
   widely the deeper the light travelled through cloud (forward-scattering halo
   at first, isotropic glow when thick). Clipped aurora highlights get their
   hue restored first. Cloud colour therefore follows the aurora frame by
   frame.
5. **Compositing.** `out = clear x direct transmittance + in-scattered light`,
   in linear light, inside the sky mask only. Lost sensor grain is re-added and
   hot pixels are kept.

Temporal consistency is structural: frames are samples of one continuous 4-D
field (wind advection, two coverage fields drifting through each other, and
vertical "boiling" of the shape noise), not per-frame draws.

Brightness is not monotonic in thickness, by construction. Thin cloud adds
forward-scattered aurora light on top of an almost untouched background; thick
cloud removes the background and returns only what diffuses through, plus
whatever the ground lights put on its base.

### What is learned from the cloudy clips

The 87 reference clips are measured, not copied, and no neural model is
trained (`fit_refs.py`, `fit_motion.py`, `fit_prior.py` -> `cloudvol/data/prior.json`):

| quantity | measured value | used for |
|---|---|---|
| zenith drift, px/frame (p10 / p50 / p90) | 0.21 / 0.52 / 0.97 | winds are sampled from the empirical list |
| cloud pattern surviving 60 s (p10 / p50 / p90) | 0.61 / 0.76 / 0.88 | evolution rate |
| cloud pattern surviving 300 s | 0.43 / 0.56 / 0.70 | evolution rate |
| cloudy-sky colour, 2015-18 camera (zenith / mid / NW rim) | (66,86,72) / (75,96,79) / (91,113,89) | ground-light colour and direction |
| cloudy-sky colour, 2019+ camera (zenith / mid / NW rim) | (100,92,74) / (118,105,82) / (147,140,107) | ground-light colour and direction |
| dome luminance, structure contrast, star visibility | envelopes per camera era | `validate.py` scores synthetic clips with the same analysers |

Cloud shapes are procedural and were tuned by eye against the references.

### Variety

`weather.py::sample_scenario(seed, era)` draws a regime — broken deck, overcast
with holes, thin high veil, scattered fragments, cloud streets, two layers with
different winds, or a passing front — then its geometry, optical thickness,
texture scales, wind, evolution rates, a cover timeline (steady, trending or
pulsing) and ground-light strengths. The scenario is saved next to every output
and reproduces the render exactly. `outputs_cloudvol/variety_*.jpg` show all
seven regimes on two clips.

### How the demo scores against the references

| measure | demo (seed 1, `broken`) | reference clips |
|---|---|---|
| drift at zenith, px/frame | 0.27 | 0.21-0.97, median 0.52 |
| pattern kept after 1 min | 0.79 | median 0.76 |
| pattern kept after 5 min | 0.50 | median 0.56 |
| dome luminance | 94 | median 91 |
| mid-scale structure contrast | 0.121 | 95th percentile 0.114 |

Contrast is slightly above the reference range; the clear clip itself measures
0.15, so much of that is its own bright aurora showing through.
Frame-to-frame change under thick cloud is lower than in the clear input, so no
flicker is added.

## Usage

```
python -m cloudvol.fit_refs   Cloud_footgae cloudvol/data/ref_stats.json    # once
python -m cloudvol.fit_motion Cloud_footgae cloudvol/data/ref_motion.json   # once
python -m cloudvol.fit_prior                                                # once
python -m cloudvol.run Clear_sky_footage/22112017_195302.avi --seed 7 --demo
python -m cloudvol.validate outputs_cloudvol/22112017_195302_s7_cloudy.mp4 --era A
```

`run` writes `<clip>_s<seed>_cloudy.mp4`, `_mask.mp4` (cloud opacity = 1 -
direct transmittance), `_demo.mp4` (clear | cloudy, with `--demo`) and
`<clip>_s<seed>.json` (the scenario). `--regime` forces a regime; `--scenario`
replays a saved one. Rendering alone takes about 0.3 s per frame on an M3
(MPS); a full 3600-frame clip with decoding and three encoders takes roughly 45
minutes.

Requires python3 with torch, numpy, scipy and opencv, and ffmpeg on PATH.

## Repository layout

```
cloudvol/             current pipeline (see above); cloudvol/data/ holds the fitted reference measurements
outputs_cloudvol/     scenario of the current demo and the regime variety sheets
docs/images/          pictures used in this README
docs/RESEARCH_cloud_realism.md   literature notes written during the previous pipeline (round 7)
```

Kept locally and not part of this repository:

```
Clear_sky_footage/    clear clips to cloud (KHO all-sky camera footage)
Cloud_footgae/        87 real cloudy reference clips ("Test *.avi", KHO all-sky camera footage)
outputs_cloudvol/*.mp4   current demo, cloudy clip and mask videos
outputs_v1_old/       previous pipeline's demo videos
outputs/              previous pipeline's round-6 output video
```

The raw footage belongs to the observatory (see References) and is not
redistributed here; rendered videos are left out for size. Per-clip sky-mask
caches are rebuilt automatically on first run.

## Known limits (initial version)

* Moonlit and twilight skies are not modelled: adding a moon would change the
  clear sky itself, not only the clouds. Some reference clips are of that kind
  and are not represented.
* Regimes are not equally convincing. Broken decks, scattered fragments,
  two-layer skies and passing fronts look best; the overcast regime can still
  show an over-smooth deck edge, and cloud streets ("rolls") can look streaky.
* Ground-light positions and colours are set per camera era from zone colour
  statistics and a handful of inspected frames; the 2019+ era has been checked
  less than the 2015-18 era.
* Morphology is procedural, not learned. A learned texture prior for the
  density field is the natural next step if the procedural look proves limiting
  downstream.
* The sky mask stops at the first static outline, which also excludes a thin
  crescent of real sky behind lens-rim reflections.
* Frame time is assumed to be 1 s per frame (true for these clips).

Two lessons about the noise are worth keeping. Any 2-D field that gates the
volume (the coverage map) is extruded vertically, so it must contain no
wavelength shorter than the layer thickness, or it renders as vertical curtains
— radial streaks in the fisheye. And 3-D noise must be isotropic in kilometres,
not in texels. Both produced comb-like artefacts before they were fixed.

## Previous pipeline (benchmark)

The old demo videos are kept locally; the code that made them has been
removed and is not part of this repository. This is how the videos were made.

**The method.** `aurora_cloudsim/` started from a vendored copy of
SatelliteCloudGenerator [2] (a daytime-satellite cloud synthesiser: a per-frame
2-D Perlin transparency mask blended toward white) and rebuilt it for night
all-sky video:

* a single cloud *plane* at 1-3 km seen through the fisheye; a seeded periodic
  3-D value-noise field over (time, y, x), advected by wind, gave one 2-D
  optical-depth map per frame;
* direct transmittance `exp(-tau * sec(zenith))`, a thickness-dependent blur
  of the background, and a scattered-light term built from a heavily blurred
  copy of the frame plus a ground-glow colour measured from the clip;
* from round 6, "two-path" transport in code-value space (downwelling light
  decaying with depth, upwelling light returned with a saturating response),
  and a learned appearance correction: a small U-Net over quarter-resolution
  physical fields adding a bounded, band-limited, gated residual, trained
  adversarially (two-scale PatchGAN plus feature matching) on crops extracted
  from the cloudy clips;
* round 7 was a literature review (`docs/RESEARCH_cloud_realism.md`) after the
  round-6 output failed visual review for fog-filter softness, GAN colour
  mottling and a uniform bright rim;
* round 8 added a Worley erosion chain for the opacity map, grid-searched
  against texture-band statistics, and fitted per-band temporal decorrelation
  on all 87 clips;
* round 9 added pooled deck-light calibration (thick-deck hue, and per-channel
  amplification of cloudy over clear sky of 2.21 / 1.67 / 2.03) and fitted the
  low-band gain of the learned correction — the fit selected 0, switching that
  band off;
* round 10 is the final state of that code. Its sources carry no round-10
  labels, so what changed after round 9 is not recorded here.

**The videos.**

| file | what it is |
|---|---|
| `outputs_v1_old/Third_attempt.mp4` | early full-length single-panel render (19 July, 3598 frames) |
| `outputs/improved_18122017_000058.mp4` | round-6 published output, seed 33, regime `haze_patches`, learned appearance on |
| `outputs_v1_old/demo_progress.mp4` | first of the three 24 July demos (unlabelled; by timestamp, presumably round 8) |
| `outputs_v1_old/demo_progress_round9.mp4` | round 9 |
| `outputs_v1_old/demo_progress_round10.mp4` | round 10 |

The three `demo_progress*` files are clear | cloudy side by side on
`18122017_000058`, using every third source frame (1198 frames).

**Why it was replaced.** It passed its own statistical checks, invariant tests
and audits while still looking like a haze filter. The cloud was a single 2-D
sheet whose opacity and brightness came from the same scalar, so it had no
depth, no gaps closing with perspective, and no way for thick and thin cloud to
be lit differently. Statistics alone did not catch that; looking at frames did.

## References and acknowledgements

This repository contains no code from other projects. It builds on the
following data and published work:

1. **Footage.** All clips are from the all-sky camera of the Kjell Henriksen
   Observatory (KHO), The University Centre in Svalbard (UNIS), Longyearbyen,
   Svalbard — <https://kho.unis.no>. The footage is not included here.
2. **SatelliteCloudGenerator.** Czerkawski, M., Atkinson, R., Michie, C. and
   Tachtatzis, C. (2023). "SatelliteCloudGenerator: Controllable Cloud and
   Shadow Synthesis for Multi-Spectral Optical Satellite Images." *Remote
   Sensing* 15(17), 4138. <https://doi.org/10.3390/rs15174138>. Code:
   <https://github.com/strath-ai/SatelliteCloudGenerator>. This was the
   starting point of the previous pipeline (see above); none of it is used by,
   or included with, the current one.
3. Schneider, A. and Vos, N. (2015). "The Real-time Volumetric Cloudscapes of
   Horizon: Zero Dawn." *Advances in Real-Time Rendering in Games*, SIGGRAPH
   2015. The coverage-gated, height-profiled, detail-eroded density model in
   `cloudvol/render.py` follows this scheme.
4. Schneider, A. (2017). "Nubis: Authoring Real-Time Volumetric Cloudscapes
   with the Decima Engine." *Advances in Real-Time Rendering in Games*,
   SIGGRAPH 2017.
5. Worley, S. (1996). "A Cellular Texture Basis Function." *Proceedings of
   SIGGRAPH '96*, 291-294. Cellular noise used in `cloudvol/textures.py`.

`docs/RESEARCH_cloud_realism.md` lists the wider literature consulted during
the previous pipeline, with links.

Software used: PyTorch, NumPy, SciPy, OpenCV and FFmpeg.
