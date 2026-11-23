# Cloud-realism research (round 7)

Trigger: step-2000 learned model + physics pipeline failed visual QA per user criteria (fog-filter softness; GAN color mottling vs measured hue-pinned luminance-only mottling; uniform bright rim ring). Stats/invariants/null-audit all PASS - the failure is purely in look, which the band-limited D + mean/std feature matching cannot see.

Multi-agent sweep: 66 unique papers across 6 angles, 12 deep-read.

# Prioritized Improvement Plan — Cloudsim Pipeline

## Part A — Verdicts on the four candidate fixes

### A1. Hue-locked learned correction (single-channel G × base-hue unit vector) — **SUPPORTED, with three refinements**
The strongest consensus in the whole corpus. Every synthesis method that improved realism did so by removing per-channel freedom and deriving color from one scalar field: PGCS's single-channel thickness + frozen spectral law [1], SatelliteCloudGenerator's shared mask with scene-tied color, whose ablation shows per-channel *amplitude calibration* — not texture — was the decisive realism factor [2], LCDiff's channel-swap evidence that weather degradation lives almost entirely in Y [7], production volumetrics' gray extinction with color only from light sources [9][10], and the boredealis negative result ("magenta clouds") when color came from anywhere but the scene [12]. Physically, there is no second illuminant: cloud color shift is a single global drift toward the ground-light spectrum [4][5][6].

**Refinements the literature demands:**
1. **The chroma direction must be scene-measured, per-pixel** — the unit color of the local physics-rendered light mix (downwelling aurora vs. upwelling town), not one fixed global vector [3][6][12]. A global vector would tint aurora-lit regions with town color.
2. **Make the discriminator luma-only too.** Feed PatchGAN + feature-matching BT.601 luma crops [7, constrained-gan notes]. With a 1-channel G this is belt-and-suspenders, but it removes the gradient incentive at the source and lets D focus capacity on structure — pix2pix's own rationale says D should never own low-frequency color.
3. **Any real global chroma drift belongs in the physics stage, deterministic**: if measurement of the 30 clips shows a low-frequency chroma shift under thick decks, model it as one monotone per-frame mix toward the measured town color as a function of dome-mean cover (log-linear form from [5], CCT monotonicity from [4][6]) — never in G.

### A2. Worley/curl-noise detail erosion of tau — **SUPPORTED, but the mechanism must be subtractive remap-erosion, not added noise**
The production literature is unanimous [9][10, volumetric-clouds notes]: *adding* Worley octaves to tau gives more fog; crispness comes from a specific chain: (a) Perlin-Worley base (packed cellular billows), (b) **coverage remap** `remap(base, 1-C, 1, 0, 1)*C` producing **exact-zero gaps** and steep deck edges, (c) high-frequency inverted-Worley detail **subtracted only near edges** (erosion strength gated by `1 - shape` so cores stay dense), (d) a re-saturating remap restoring contrast. Curl-noise distortion of the detail sample coordinates shears edges into wisps and is divergence-free (GT-safe) [10]. SCG independently converges on the same ingredients in 2D: tanh output-transform plateaus, product-of-masks locality, `clear_threshold` zero floor [2]; boredealis converges on decoupled coverage-vs-mottle fields [12]. So: yes, but implement the HZD/Nubis chain, not "mix in some Worley."

### A3. Azimuthally-modulated town-sector rim — **SUPPORTED, and the literature substantially extends it**
Measured cloudy skies are azimuthally asymmetric by up to ~an order of magnitude at the same zenith angle (0.15× vs 11.3×) [6]; a uniform ring is off by up to ~75×. Extensions we must adopt:
1. **Fit the azimuth weight w(φ) from each clip's own clear frames** (horizon-annulus luma excess, von Mises smoothed) — the Longyearbyen dome is already visible in the clean footage [3][6].
2. **Place the lobe at the physically correct elevation**, θ_peak = 90° − atan(h_cb/d_town), not on the rim; measure the empirical h_cb distribution from the 30 Test clips via h_CB = d·tan(ε) [6].
3. **The rim must fade, not persist, as decks close**: Ev_scal/Ev_cos falls from 3.43 → 1.90 under overcast+snow — heavy decks render the dome near-isotropic. Multiply clear-sky horizon-glow amplitude by (1 − R(τ_deck)) [4], and make the town-dome zenith-falloff τ-dependent ("creeping") [3].
4. **Away-from-source horizon must DARKEN under thick decks** (3.8–11.3× measured) via slant-path extinction τ_view(θ) = τ_zenith·min(1/cosθ, ~6) — invertible view-path extinction, GT-exact [6].
5. Keep a small isotropic snow floor w_snow > 0 (Svalbard albedo) [4][6].

### A4. Evolution-dominated temporal model (slow advection + detail-phase re-seed) — **SUPPORTED in intent, REFINED into something better: per-scale AR(2) cascade**
The direction (evolution ≫ advection) is right [9][10][11, temporal-dynamics notes], but discrete phase re-seeding "every ~minutes" will pop, and pure 4D-noise time-scrubbing reads as soft morphing — a fog-filter regression (temporal-dynamics notes). STEPS/pysteps [11] provides the exact, *fittable* replacement: bandpass-decompose log-τ into ~7 levels, evolve each with AR(2) whose lag-1/lag-2 autocorrelations are **measured from the 30 clips** (Lagrangian, with sub-pixel flow it's safe to fit at zero advection). Fine bands decorrelate in frame-scale time, large bands over minutes — continuous churn, correct per-scale statistics, deterministic per seed, trivially cheap at 608². Combine with Nubis's split: **coverage field frozen/near-static per clip** (advected at ~0.01 px/frame), a deterministic mean ramp μ(t) for deck closing (the AR process must not be asked to produce the 83→143 trend), and curl-swirl on detail coordinates for shear [10][11].

---

## Part B — Prioritized recommendations

| # | Fix | Failure mode | Digests | Runs in | Effort |
|---|-----|--------------|---------|---------|--------|
| R1 | 1-channel G + per-pixel scene-light chroma direction; PatchGAN+FM on luma-only crops; keep per-annulus zero-mean on the scalar | FM2 | 1,2,3,5,6,7,9,12 | learned correction + training loss | **1 day** (arch + retrain on 5090) |
| R2 | Chroma-envelope QA gate: std(Cb/Cr) and low-pass ΔCb/ΔCr of synthetic clips must fall in the 5–95% envelope measured on the 30 real clips; plus per-annulus/sector R-G, B-G spatial-std check | FM2 (regression guard) | 5,7 | offline QA (Mac) | 0.5 day |
| R3 | Rebuild τ: Perlin-Worley base → coverage remap (exact-zero gaps) → edge-gated HF Worley erosion with curl-distorted coords → re-saturating remap; per-clip deck-type presets; calibrate to band stds (2.6–4.4 / 3.5–5.8 / 7.3–11.3) | FM1 | 9,10,2,12 | procedural physics | **2–3 days** |
| R4 | Monotone CDF-match τ to pooled reference log-extinction distribution (frozen per clip); tunes intermittency/heavy tail | FM1 | 11 | procedural physics | 0.5–1 day |
| R5 | Powder/in-scatter multiplier on upwelling path: R(τ)·[p0 + (1−p0)·(τ_lp/τ_ref)^k] — thin edges/gaps read darker than cores; scalar, hue-pinned, invertible | FM1 (+matches measured luminance-only mottling) | 9,10 | procedural physics | 0.5 day |
| R6 | Azimuthal source-lobed upwelling: w(φ) from clear frames; lobe elevation from h_cb geometry; snow floor; horizon-glow amplitude ×(1−R(τ_deck)); slant-path darkening in dark azimuths | FM3 | 3,4,6 | procedural physics | **2 days** |
| R7 | Rim/aniso acceptance metrics: S/C dome ratio flag (>~2.3 at high coverage), horizon-annulus azimuth std must not drop clear→cloudy, per-wedge amplification profiles vs zenith angle matching real-clip envelopes | FM3 (gate) | 3,4,6 | offline QA (Mac) | 0.5–1 day |
| R8 | STEPS AR(2) cascade on log-τ bands, ρ_j1/ρ_j2 fitted from the 30 clips (Yule-Walker + stationarity clamp; vendor pysteps, MIT/BSD); coverage static + μ(t) ramp; advection 0.012 px/s; curl phase drift | FM4 | 11,9,10 | procedural physics | **2–3 days** incl. fitting |
| R9 | CSM-style calibration: ρ_c = P95(cloudy)/P50(clear) per channel from real clips → fit upwelling gain + color; the single feature that moved SCG's downstream masked SSIM 0.32→0.46 | FM1/FM2 (amplitude realism, the proven dominant factor) | 2, (4,5 for form) | procedural physics (calibration) | 1 day |
| R10 | Steepen R(τ): fast onset (thin veils already amplify 2.8–6×), asymptote allowed well above clear level via two-stream reflectance + snow multi-bounce series r_c/(1−a_snow·r_c); per-clip lognormal jitter of asymptote | FM1-adjacent (brightening realism) | 3,4,5 | procedural physics | 1 day |
| R11 | Distribution-overlap acceptance suite (>80% histogram overlap on band-stds, dome-luma trajectory, R/G) + downstream gate: tiny syn-only-trained segmenter must make visible plausible edits on the 30 real clips | all (protocol) | 1,2 | QA + training loss | 1–2 days |
| R12 | On-the-fly parameterized synthesis during 5090 training (store parameters, not frames) | infrastructure (also fits 8GB Mac) | 12 | training loop | 0.5 day |
| R13 | Optional: τ-gated blur of downwelling aurora detail under thick cloud (blur-pyramid lerp, not unfold). **Constraint-sensitive**: amend forward model explicitly — ramp σ from 0 where transmission is high, store σ map per pair | FM1 (strongest "cloud vs veil" cue) | 2, (12 caveat) | procedural physics | 1–2 days, needs sign-off on forward-model amendment |
| R14 | Optional fallback: Harmonizer-style bounded polar-grid filter-argument head (luma-only chain, per-annulus zero-mean on gain) if the free-field scalar GAN still misbehaves; + self-supervised inverse-filter pretraining; θ-EMA over ~30–120 real seconds | FM2/FM3/FM4 (correction side) | 8 | learned correction | 3–4 days, hold in reserve |
| R15 | Later: StyleGAN2-ADA single-channel texture layer for mid/fine τ bands + latent-walk evolution; hold-out clips + nearest-neighbor memorization checks | FM1/FM4 | 1 | procedural physics (learned texture) | 3–5 days + training; only if R3/R4/R8 fall short |
| R16 | Later: 3-frame zero-init residual deflicker net on the *removal model's* output (proven on identical KHO footage) | FM4 (output side) | 12 | downstream model | 1–2 days, post-hoc |

All of R1–R12 keep GT exact: τ remains the single scalar mask, all remaps/CDF maps are monotone functions of τ applied *before* transport, corrections stay bounded/gated/band-limited, and no reference clip is copied (only pooled statistics). R13 is the one deliberate forward-model amendment and is flagged as such.

---

## Part C — Things the digests reveal that we had NOT considered

1. **Amplitude calibration beats texture realism for downstream transfer** [2]. SCG's no-CSM model *silently ignored real clouds*. Our per-channel ρ_c statistics may matter more than any texture work — and we currently hand-tune the light levels. (→ R9.)
2. **The horizon should DARKEN away from town under thick decks** [6]. We've only been fighting a too-bright rim; the real signature includes 3.8–11.3× *darkening* in dark azimuths via slant-path cutoff of distant skyglow. Absence of this is itself a realism tell.
3. **Heavy decks flatten the dome to near-isotropic** [4]. The rim problem is not just azimuthal — under a closed deck the horizon must not be relatively brighter than mid-dome at all. Gives a one-line per-frame detector (S/C ratio).
4. **Our R(τ) asymptote may be clamped too low.** Snow closes a multi-bounce loop; measured snow-site amplification is 26× linear [4]. The measured 1.7× (83→143) is at the *conservative* end — per-clip jitter of the asymptote (lognormal, [5]'s scatter growth) adds real deck-to-deck diversity we currently lack.
5. **Cloud-base height as a shared latent** [6]: h_cb ties together lobe elevation, angular advection scale (wind/h_cb), and is *measurable* from the Test clips via h_CB = d·tan(ε). One parameter couples glow geometry and motion consistently.
6. **Dual Beer-Lambert transmission floor** [10]: `max(exp(−στ), 0.7·exp(−0.25στ))` keeps visible internal structure in heavy decks instead of clamping to black; still monotone/invertible.
7. **Gray extinction check** [5]: confirm our T(τ) uses one scalar k across channels; all chromatic behavior belongs in the additive upwelling term. Keeps the mask a single scalar field.
8. **Downstream-transfer as the acceptance protocol** [1][2]: rank synthesis variants by removal-model performance, not eyeballing — PGCS showed the ranking transfers.
9. **Static datasets overfit; randomize per draw** [12] — also solves our 8GB storage pressure.
10. **Objective external metrics**: star-dimming vs injected τ, keogram coefficient-of-variation, temporal-decorrelation detectors (allsky-astro / aurora-asi-decloud notes) — validation that never touches reference-clip content.
11. **Flow-warp temporal consistency is the wrong prior for our downstream model** (aurora-asi-decloud notes): evolution-dominated boiling breaks optical-flow assumptions — relevant when we design the removal model, not just synthesis.

---

## Do-first shortlist (max realism gain per day)

1. **R1 + R2 — Hue-lock the correction and blind the discriminator to chroma** (~1.5 days, learned correction). Eliminates FM2 *by construction*; unanimous literature support; cheapest certain win.
2. **R3 (+R5) — Remap-erosion τ chain with coverage zeros, edge-gated Worley erosion, powder term** (~3 days, procedural). The single biggest attack on the fog-filter look; every production system uses exactly this mechanism.
3. **R6 (+R7) — Source-lobed, elevation-correct, coverage-gated horizon treatment with dark-azimuth slant darkening** (~2.5 days, procedural). Kills the artificial ring and adds the measured asymmetry signature that reads as "real sky."
4. **R8 — Fitted per-band AR(2) evolution of τ, coverage frozen, μ(t) ramp for closure** (~2.5 days, procedural). Turns sliding into statistically-correct boiling at 25×, with parameters measured from — not copied from — the 30 reference clips.

Run R9 (ρ_c calibration) immediately after, before any large retrain: it is one day of work and the literature says it is the highest-leverage realism factor for downstream transfer.

---

# Completeness critique

## 1. Failure modes without a strong, evidence-backed fix

- **FM4, structural conflict R3 vs R8 (unaddressed).** R3 builds non-Gaussian cellular structure (remap zeros, edge erosion); R8 evolves bandpassed log-τ with AR(2) Gaussian innovations. AR innovations Gaussianize the field over the correlation time — the remap structure decays back into the exact fog-filter texture R3 was built to kill. STEPS [11] never claims texture-shape preservation (it synthesizes rain-rate fields whose realism is judged by rank statistics, not visual cellular structure). No cited digest shows an AR cascade producing visually crisp cloud texture. Fix candidates exist (evolve the *noise inputs*/Worley seed points, or re-impose R4's CDF map per frame after AR evolution) but the synthesis proposes neither.
- **FM4, coordinate frame of R8.** The codebase samples τ on a world cloud-plane through the fisheye (geometry.py: `world_x/world_y = h·tan(θ)`), correctly. R8's "bandpass-decompose log-τ … fit ρ from the 30 clips" is specified in image space. On an equidistant fisheye a fixed image-space band mixes physical scales by >10× between zenith and rim; STEPS assumes a homogeneous Cartesian grid. Fitting Yule-Walker on raw 608² frames gives wrong per-scale timescales, and evolving image-space bands makes the rim churn at zenith rates. Both fitting and evolution must run on the reprojected cloud plane (requires per-clip h_cb — which is exactly the C5 "shared latent," never wired into R8).
- **FM4/FM2, temporal behavior of the learned correction G.** Nothing in R1–R13 makes dS temporally coherent. A per-frame G on stochastic fields shimmers at 25 fps — low-frequency luminance mottle flicker is a new tell. R14 (θ-EMA) is "held in reserve" and R16 acts on the removal model's output, not synthesis. No digest is cited for temporal regularization of a bounded residual GAN.
- **FM3, terrain/occlusion mask.** KHO frames contain a horizon silhouette (terrain, structures) inside the image circle. If cloud τ, rim glow, or slant darkening composite *over* that silhouette, it is an instant artificial-ring-adjacent tell. No R covers a static scene mask; no digest addresses it.
- **FM1-adjacent, sensor-noise consistency.** The measured brightening (83→143) changes shot-noise statistics. If the noise stage adds fixed-variance noise after brightening, heavy decks look too clean/too noisy vs the real clips. No fix, no QA gate (R2/R7/R11 check none of this).
- **Moonlit clips.** If any of the 30 Test clips are moonlit, the "no second illuminant" premise of A1 fails (white top-lit decks + orange bottom-lit decks coexist). Unhandled and unmeasured.

## 2. Claims not supported by the cited digests

- **"pix2pix's own rationale says D should never own low-frequency color" (A1.2).** pix2pix is not among the 12 digests, and its actual rationale is L1-handles-low-frequencies / PatchGAN-handles-high-frequencies — not a color claim. Uncited and misstated.
- **A1.2's luma-only discriminator cites "[7, constrained-gan notes]".** LCDiff [7] is a diffusion method with no discriminator; "constrained-gan notes," "temporal-dynamics notes," "volumetric-clouds notes," "allsky-astro / aurora-asi-decloud notes" are not in the digest list at all, yet carry load-bearing claims (D-luma-only, "4D-noise reads as soft morphing," "phase re-seed pops," flow-warp-wrong-prior, star-dimming metrics). Either surface those digests or mark these as unsupported.
- **LCDiff "channel-swap evidence that weather degradation lives almost entirely in Y" [7]** is an extrapolation from daytime rain/haze/snow ground photos to night cloud decks; the digest cannot support "hue-pinned cloud mottling," only Y/chroma decoupling for its own restoration task.
- **"curl-noise … is divergence-free (GT-safe)" [10].** Non-sequitur: GT-safety comes from all warps happening to τ before transport; divergence-freeness is irrelevant. Nubis supports curl noise for wisps only.
- **Dual Beer-Lambert floor `max(exp(−στ), 0.7·exp(−0.25στ))` [10] (C6)** is a Nubis *in-cloud lighting* hack for scattered sunlight, not a view-path transmission law. Applying it as T(τ) on the clean frame leaks aurora/star structure through decks that should occlude them, and changes mask semantics. The digest does not support using it on the transmission path.
- **SCG numbers (A?/R9): "masked SSIM 0.32→0.46," "no-CSM model silently ignored real clouds," "amplitude calibration — not texture — was the decisive factor" [2]** — verify the digest actually contains an ablation isolating CSM from texture; if the compared variants differ in more than CSM, "decisive factor" is an overclaim.
- **"PGCS showed the ranking transfers" (C8) [1]** — PGCS shows its synthesis improves downstream removal; a general variant-ranking-transfer result is a stronger claim needing digest confirmation.
- **Specific numerics attributed to [4]/[5]/[6]:** "Ev_scal/Ev_cos 3.43→1.90," "26× linear snow amplification," "log-linear chroma-mix form," "0.15× vs 11.3× at the same zenith angle" (and the derived "~75×") — plausible but each needs a line-item check; the *mechanisms* built on them (horizon-glow × (1−R(τ_deck)), monotone per-frame chroma mix) are invented functional forms presented as literature-backed. Fine to invent them, but label them as such.
- **R16 "proven on identical KHO footage" [12]** — verify boredealis actually used KHO Svalbard data and actually contains a 3-frame residual deflicker net; the digest title says neither.
- **"Physically, there is no second illuminant" [4][5][6]** contradicts the synthesis's own refinement 1 (per-pixel aurora-vs-town mix = two illuminants) and ignores the moon and aurora top-lighting of decks.

## 3. Missing search angles (exact queries)

- **Ground-based sky-imager generation — the single most on-point literature, entirely absent** (every synthesis source is satellite, game, radar, or photometry): `"SkyGPT physics-constrained generative sky images cloud video prediction solar forecasting"` and `"all-sky imager cloud video prediction deep learning benchmark"`.
- **Real nighttime sky-camera cloud texture/segmentation data (free realism envelopes, no KHO copying):** `"SWINSEG SWINySEG nighttime whole sky imager cloud segmentation dataset Dev"`.
- **Physically-grounded per-scale decorrelation to replace/validate R8's rain-derived AR fits:** `"stratocumulus cloud field scale-dependent decorrelation time large eddy simulation lifetime eddy turnover"`.
- **Texture-statistics-preserving synthesis/evolution (fixes the R3↔R8 Gaussianization conflict):** `"Portilla Simoncelli steerable pyramid texture synthesis temporal evolution non-Gaussian"`.
- **Evidence base for R13's blur amount (currently near-uncited):** `"angular spread point spread function light transmitted through cloud layer multiple scattering diffusion approximation"`.
- **Cloud-base height measurement to pin the h_cb latent:** `"cloud base height estimation all-sky camera stereo photogrammetry night"`.
- **Rim artifact vs instrument: separate optics from physics before fixing physics:** `"all-sky camera flat field vignetting correction night sky photometry calibration"`.
- **Noise-stage consistency:** `"heteroscedastic Poisson-Gaussian noise model calibration low light video sensor"`.
- **Temporal coherence of the bounded correction:** `"temporally coherent GAN residual video flicker bounded correction EMA"`.
- **Moonlight check for the Test set:** `"moonlight cloud radiance all-sky photometry lunar skyglow"`.

Code paths referenced: aurora_cloudsim/geometry.py (fisheye world-plane projection already exists — R3/R8 must state they operate in these world coordinates, which the synthesis never does), aurora_cloudsim/noise3d.py, aurora_cloudsim/render.py.

---

## Top papers

- 5★ PGCS: Physical Law Embedded Generative Cloud Synthesis in Remote Sensing Images — https://arxiv.org/abs/2410.16955
- 5★ SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images — https://doi.org/10.3390/rs15174138
- 5★ Imaging and mapping the impact of clouds on skyglow with all-sky photometry — https://arxiv.org/abs/1705.04968
- 5★ Snowglow - The Amplification of Skyglow by Snow and Clouds Can Exceed Full Moon Illuminance in Suburban Areas — https://www.mdpi.com/2313-433X/5/8/69
- 5★ Red is the new black: how the colour of urban skyglow varies with cloud cover — https://academic.oup.com/mnras/article/425/1/701/1003276
- 5★ Using all-sky differential photometry to investigate how nocturnal clouds darken the night sky in rural areas — https://arxiv.org/abs/1807.10593
- 5★ When Color-Space Decoupling Meets Diffusion for Adverse-Weather Image Restoration (LCDiff) — https://arxiv.org/abs/2509.17024
- 5★ Harmonizer: Learning to Perform White-Box Image and Video Harmonization — https://arxiv.org/abs/2207.01322
- 5★ The Real-Time Volumetric Cloudscapes of Horizon: Zero Dawn (Schneider & Vos) — https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- 5★ Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine (Schneider) — https://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
- 5★ Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0) [implements STEPS: Bowler, Pierce & Seed 2006] — https://gmd.copernicus.org/articles/12/4185/2019/
- 5★ Deep Learning-Based Declouding of the Aurora Borealis (Press, Scalzo, Fasel) — https://dl.acm.org/doi/10.1007/978-3-032-14495-9_31
- 4★ Thin cloud removal in optical remote sensing images based on generative adversarial networks and physical model of cloud distortion — https://www.sciencedirect.com/science/article/abs/pii/S0924271620301787
- 4★ U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series — https://arxiv.org/abs/2305.13277
- 4★ Enhancing Visibility in Nighttime Haze Images Using Guided APSF and Gradient Adaptive Convolution — https://arxiv.org/abs/2308.01738
- 4★ Nighttime Haze Removal with Glow and Multiple Light Colors — https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Li_Nighttime_Haze_Removal_ICCV_2015_paper.pdf
- 4★ Quantitative analysis of night skyglow amplification under cloudy conditions — https://academic.oup.com/mnras/article/443/4/3665/2106676
- 4★ Urban night-sky luminance due to different cloud types: A numerical experiment — https://journals.sagepub.com/doi/10.1177/1477153515597732
- 4★ Cloud coverage acts as an amplifier for ecological light pollution in urban ecosystems — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017307
- 4★ Measuring Anthropogenic Sky Glow Using a Natural Sky Brightness Model — https://iopscience.iop.org/article/10.1086/673888
- 4★ Cloud Identification and Reconstruction from All-sky Camera Images Based on Star Photometry Estimation — https://iopscience.iop.org/article/10.1088/1538-3873/ad2867
- 4★ Nighttime Cloud Detection, Tracking and Prediction with All-Sky Cameras — https://arxiv.org/abs/2507.21711
- 4★ LenghuSky-8: An 8-Year All-Sky Cloud Dataset with Star-Aware Masks and Alt-Az Calibration for Segmentation and Nowcasting — https://arxiv.org/abs/2603.16429
- 4★ Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE) — https://arxiv.org/abs/2001.06826
- 4★ Image-to-Image Translation with Conditional Adversarial Networks (pix2pix / PatchGAN) — https://arxiv.org/abs/1611.07004
