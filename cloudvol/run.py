"""Apply synthetic clouds to a clear all-sky clip.

    python -m cloudvol.run Clear_sky_footage/22112017_195302.avi --seed 7
    python -m cloudvol.run Clear_sky_footage/22112017_195302.avi --seed 7 --heavy

Without --heavy the scenario comes from the broken / scattered / veil family
(weather.py); with --heavy from the sky-filling heavy family (heavy.py).

Writes, into --out (default outputs_cloudvol/):
    <stem>_cloudy.mp4      the clouded clip
    <stem>_mask.mp4        cloud opacity, lossless 8-bit:  255 * (1 - T)
    <stem>_thickness.mp4   optical thickness, lossless 8-bit, log-encoded (see below)
    <stem>_demo.mp4        clear | cloudy (| thickness with --demo-mask), with --demo
    <stem>.json            the scenario (fully reproduces the render), mask semantics, per-frame statistics
where <stem> = <clip>_s<seed> (plus _heavy for the heavy family).

Masks.  T is the direct transmittance of the clear frame along each line of
sight, so opacity says how much of the clear sky is hidden.  Under heavy cloud
it saturates to white almost everywhere, so the second map carries the
structure: thickness = 255 * log1p(tau) / log1p(80), with tau the optical depth
along the line of sight (tau = -ln T before clipping).  Invert with
tau = expm1(v / 255 * log1p(80)).  Both are zero outside the sky mask.
"""
import argparse
import json
import math
import os
import time
import numpy as np
import cv2

from . import scene
from .vio import read_frames, probe, Writer, MaskWriter
from .weather import sample_scenario, Scenario, REGIMES
from .heavy import sample_heavy_scenario, HEAVY_REGIMES
from .render import Renderer, TAU_MAX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heavy", action="store_true", help="sample from the heavy (sky-filling) family")
    ap.add_argument("--regime", default=None,
                    help=f"force a regime: {', '.join(REGIMES)}; with --heavy: {', '.join(HEAVY_REGIMES)}")
    ap.add_argument("--scenario", default=None, help="JSON scenario to replay instead of sampling")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs_cloudvol"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="also write clear|cloudy side by side")
    ap.add_argument("--demo-mask", action="store_true", help="add the thickness map as a third demo panel")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refs", nargs="*", default=[os.path.join(ROOT, "Clear_sky_footage"), os.path.join(ROOT, "Cloud_footage")],
                    help="folders searched for same-camera clips when building the sky mask")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.clip).replace(".avi", "")
    info = probe(a.clip)
    sc = scene.build(a.clip, a.refs)
    era = scene.era_of(scene.clip_date(a.clip))
    if a.scenario:
        scn = Scenario.from_dict(json.load(open(a.scenario))["scenario"])
    elif a.heavy:
        scn = sample_heavy_scenario(a.seed, era=era, regime=a.regime)
    else:
        scn = sample_scenario(a.seed, era=era, regime=a.regime)
    heavy = scn.regime.startswith("heavy/")
    print(f"clip {name}: era {era}, regime {scn.regime}, seed {scn.seed}, {len(scn.layers)} layer(s)", flush=True)
    R = Renderer(sc, scn)
    stem = os.path.join(a.out, f"{name}_s{scn.seed}{'_heavy' if heavy else ''}{a.tag}")
    W, H, fps = info["width"], info["height"], info["fps"]
    w_cloud = Writer(stem + "_cloudy.mp4", W, H, fps, crf=12)
    w_mask = MaskWriter(stem + "_mask.mp4", W, H, fps)
    w_thick = MaskWriter(stem + "_thickness.mp4", W, H, fps)
    demo = a.demo or a.demo_mask
    w_demo = Writer(stem + "_demo.mp4", W * (3 if a.demo_mask else 2), H, fps, crf=15) if demo else None
    covers, taus, t0, n = [], [], time.time(), 0
    for i, f in read_frames(a.clip, step=a.step, start=a.start, count=a.count):
        out, aux = R.render(f, i, return_aux=True)
        w_cloud.write(out)
        w_mask.write(aux["opacity"])
        w_thick.write(aux["thickness"])
        if w_demo:
            panels = [f, out]
            if a.demo_mask:
                panels.append(np.repeat(aux["thickness"][..., None], 3, -1))
            w_demo.write(np.ascontiguousarray(np.hstack(panels)))
        covers.append(round(aux["cover"], 3)); taus.append(round(aux["tau_col_median"], 2)); n += 1
        if n % 300 == 0:
            print(f"  frame {i}  {n / (time.time() - t0):.1f} fps  cover {aux['cover']:.2f}  "
                  f"median overhead optical depth {aux['tau_col_median']:.1f}", flush=True)
    for w in (w_cloud, w_mask, w_thick, w_demo):
        if w:
            w.close()
    json.dump(dict(
        clip=name, era=era, frames=n, start=a.start, step=a.step,
        scenario=json.loads(scn.to_json()),
        masks=dict(
            mask="uint8 = 255 * (1 - T); T = direct transmittance of the clear frame along the line of sight; 0 outside the sky mask",
            thickness=f"uint8 = 255 * log1p(tau) / log1p({TAU_MAX:g}); tau = optical depth along the line of sight, "
                      f"clipped at {TAU_MAX:g}; invert: tau = expm1(v / 255 * {math.log1p(TAU_MAX):.6f}); 0 outside the sky mask",
            encoding="lossless full-range H.264 (values are exact)"),
        cover_per_frame=covers, median_overhead_optical_depth_per_frame=taus), open(stem + ".json", "w"))
    # Completeness check.  The container header and ffprobe's own frame count can each be
    # off by a frame or two from what the decoder actually yields, so allow that much.
    if any(w and w.n != n for w in (w_cloud, w_mask, w_thick, w_demo)):
        raise RuntimeError("output streams have different frame counts")
    avail = len(range(a.start, info["nb_frames"], a.step))
    expected = avail if a.count is None else min(a.count, avail)
    if abs(n - expected) > 2:
        raise RuntimeError(f"rendered {n} frames, expected about {expected}")
    print(f"done: {n} frames in {time.time() - t0:.0f}s -> {stem}_cloudy.mp4", flush=True)


if __name__ == "__main__":
    main()
