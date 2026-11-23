"""Apply synthetic clouds to a clear all-sky clip.

    python -m cloudvol.run Clear_sky_footage/22112017_195302.avi --seed 7

Writes, into --out (default outputs_cloudvol/):
    <clip>_s<seed>_cloudy.mp4     the clouded clip
    <clip>_s<seed>_mask.mp4       cloud opacity (1 - direct transmittance), greyscale
    <clip>_s<seed>_demo.mp4       clear | cloudy side by side (with --demo)
    <clip>_s<seed>.json           the scenario (fully reproduces the render) + per-frame cover
"""
import argparse
import json
import os
import time
import numpy as np
import cv2

from . import scene
from .vio import read_frames, probe, count_frames, Writer
from .weather import sample_scenario, Scenario
from .render import Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default=None, help="force a cloud regime (see weather.REGIMES)")
    ap.add_argument("--scenario", default=None, help="JSON scenario to replay instead of sampling")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs_cloudvol"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="also write clear|cloudy side by side")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refs", nargs="*", default=[os.path.join(ROOT, "Clear_sky_footage"), os.path.join(ROOT, "Cloud_footgae")],
                    help="folders searched for same-camera clips when building the sky mask")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.clip).replace(".avi", "")
    info = probe(a.clip)
    sc = scene.build(a.clip, a.refs)
    era = scene.era_of(scene.clip_date(a.clip))
    if a.scenario:
        scn = Scenario.from_dict(json.load(open(a.scenario))["scenario"])
    else:
        scn = sample_scenario(a.seed, era=era, regime=a.regime)
    print(f"clip {name}: era {era}, regime {scn.regime}, seed {scn.seed}, {len(scn.layers)} layer(s)", flush=True)
    R = Renderer(sc, scn)
    stem = os.path.join(a.out, f"{name}_s{scn.seed}{a.tag}")
    w_cloud = Writer(stem + "_cloudy.mp4", info["width"], info["height"], info["fps"], crf=12)
    w_mask = Writer(stem + "_mask.mp4", info["width"], info["height"], info["fps"], crf=16)
    w_demo = Writer(stem + "_demo.mp4", info["width"] * 2, info["height"], info["fps"], crf=15) if a.demo else None
    covers, t0, n = [], time.time(), 0
    for i, f in read_frames(a.clip, step=a.step, start=a.start, count=a.count):
        out, aux = R.render(f, i, return_aux=True)
        w_cloud.write(out)
        w_mask.write(np.repeat(aux["opacity"][..., None], 3, -1))
        if w_demo:
            w_demo.write(np.ascontiguousarray(np.hstack([f, out])))
        covers.append(round(aux["cover"], 3)); n += 1
        if n % 300 == 0:
            print(f"  frame {i}  {n / (time.time() - t0):.1f} fps  cover {aux['cover']:.2f}", flush=True)
    for w in (w_cloud, w_mask, w_demo):
        if w:
            w.close()
    json.dump(dict(clip=name, era=era, frames=n, start=a.start, step=a.step,
                   scenario=json.loads(scn.to_json()), cover_per_frame=covers), open(stem + ".json", "w"))
    # every decodable source frame must have been rendered (headers can overstate the count)
    avail = len(range(a.start, count_frames(a.clip), a.step))
    expected = avail if a.count is None else min(a.count, avail)
    if n != expected:
        raise RuntimeError(f"rendered {n} frames, expected {expected}")
    print(f"done: {n} frames in {time.time() - t0:.0f}s -> {stem}_cloudy.mp4", flush=True)


if __name__ == "__main__":
    main()
