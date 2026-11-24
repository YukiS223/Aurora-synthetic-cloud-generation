"""Score a synthetic cloudy clip with the same analysers used on the real ones.

    python -m cloudvol.validate outputs_cloudvol/<clip>_cloudy.mp4 [--era A]

Prints where the clip sits relative to the reference envelopes: drift at the
zenith, how much of the cloud pattern survives 1 and 5 minutes (evolution),
dome luminance, mid-scale structure contrast, and star visibility.
"""
import argparse
import json
import os
from . import fit_motion, fit_refs

DATA = os.path.join(os.path.dirname(__file__), "data")


def measure(path):
    m = fit_motion.analyse(path)
    s = fit_refs.analyse(path)
    return dict(
        drift_px_per_frame=m.get("drift_px_per_frame"),
        survive_60s=m["lag60"]["peak_med"] if m["lag60"] else None,
        survive_300s=m["lag300"]["peak_med"] if m["lag300"] else None,
        lum_mean=s["lum_mean"], lum_std_time=s["lum_std_time"],
        contrast=s["contrast_median"], stars=s["star_median"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("--era", default="A")
    ap.add_argument("--heavy", action="store_true", help="compare against the heavy reference subset")
    a = ap.parse_args(argv)
    prior = json.load(open(os.path.join(DATA, "prior.json")))
    key = "envelope_A_2015_2018" if a.era == "A" else "envelope_B_2019_2024"
    r = measure(a.clip)
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    if a.heavy:
        hv = prior["heavy"]; env = hv[key]
        print(f"{'metric':<22}{'synthetic':>10}   heavy references (p10 / p50 / p90, n={env['n']} clips of this camera era)")
        d = hv["drift_px_per_frame"]
        import numpy as np
        print(f"{'drift px/frame':<22}{fmt(r['drift_px_per_frame']):>10}   {np.percentile(d, [10, 50, 90]).round(2).tolist()}  (all {len(d)} heavy clips)")
        print(f"{'pattern kept @60s':<22}{fmt(r['survive_60s']):>10}   {hv['survive_60s']}")
        print(f"{'pattern kept @300s':<22}{fmt(r['survive_300s']):>10}   {hv['survive_300s']}")
        for k, lab in (("lum_mean", "dome luminance"), ("lum_std_time", "luminance std (time)"), ("contrast", "structure contrast"), ("stars", "star count")):
            print(f"{lab:<22}{fmt(r[k]):>10}   {env[k]}")
        return r
    env = prior[key]
    print(f"{'metric':<22}{'synthetic':>10}   reference")
    print(f"{'drift px/frame':<22}{fmt(r['drift_px_per_frame']):>10}   p10/p50/p90 = {prior['drift_percentiles']['p10']}/{prior['drift_percentiles']['p50']}/{prior['drift_percentiles']['p90']}")
    print(f"{'pattern kept @60s':<22}{fmt(r['survive_60s']):>10}   p10/p50/p90 = {list(prior['survive_60s'].values())}")
    print(f"{'pattern kept @300s':<22}{fmt(r['survive_300s']):>10}   p10/p50/p90 = {list(prior['survive_300s'].values())}")
    for k, lab in (("lum_mean", "dome luminance"), ("lum_std_time", "luminance std (time)"), ("contrast", "structure contrast"), ("stars", "star count")):
        print(f"{lab:<22}{fmt(r[k]):>10}   p5/p25/p50/p75/p95 = {env[k]}")
    return r


if __name__ == "__main__":
    main()
