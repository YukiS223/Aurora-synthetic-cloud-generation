"""Condense the reference measurements into the scenario prior.

    python -m cloudvol.fit_prior

Reads data/ref_motion.json and data/ref_stats.json, writes data/prior.json:
  * the empirical list of zenith drift speeds / headings (clips where drift
    could be tracked reliably) - winds are sampled from it,
  * evolution statistics (how much of the cloud pattern survives 1 and 5 min),
  * per-era colour of the cloudy sky by zone (used to calibrate ground lights),
  * reference envelopes for the realism checks in validate.py,
  * the same for the heavy subset (stars extinguished all hour), used by heavy.py.
"""
import json
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
HEAVY_STAR_P90 = 12      # a clip is "heavy" if its star count stays at or below this for 90% of the hour


def main():
    mot = json.load(open(os.path.join(DATA, "ref_motion.json")))
    st = json.load(open(os.path.join(DATA, "ref_stats.json")))
    ok = {k: v for k, v in mot.items() if v.get("frac_tracked", 0) >= 0.5 and "drift_px_per_frame" in v}
    drift = [round(v["drift_px_per_frame"], 4) for v in ok.values()]
    head = [round(v["heading_deg"], 1) for v in ok.values()]
    p60 = [v["lag60"]["peak_med"] for v in ok.values() if v["lag60"]]
    p300 = [v["lag300"]["peak_med"] for v in ok.values() if v["lag300"]]
    prior = dict(
        n_clips=len(mot), n_tracked=len(ok),
        drift_px_per_frame=drift, heading_deg=head,
        drift_percentiles=dict(zip(["p10", "p25", "p50", "p75", "p90"], np.percentile(drift, [10, 25, 50, 75, 90]).round(3).tolist())),
        survive_60s=dict(zip(["p10", "p50", "p90"], np.percentile(p60, [10, 50, 90]).round(3).tolist())),
        survive_300s=dict(zip(["p10", "p50", "p90"], np.percentile(p300, [10, 50, 90]).round(3).tolist())),
        evo_scale=1.0,
    )
    for era in ("A_2015_2018", "B_2019_2024"):
        clips = [v for v in st.values() if v["era"] == era]
        if not clips:
            continue
        z = np.array([c["zone_rgb_median"] for c in clips])            # clips x 24 x 3
        prior[f"zone_rgb_{era}"] = dict(n=len(clips), median=np.median(z, 0).round(1).tolist(),
                                        p25=np.percentile(z, 25, axis=0).round(1).tolist(),
                                        p75=np.percentile(z, 75, axis=0).round(1).tolist())
        prior[f"envelope_{era}"] = dict(
            lum_mean=np.percentile([c["lum_mean"] for c in clips], [5, 25, 50, 75, 95]).round(1).tolist(),
            contrast=np.percentile([c["contrast_median"] for c in clips], [5, 25, 50, 75, 95]).round(4).tolist(),
            lum_std_time=np.percentile([c["lum_std_time"] for c in clips], [5, 25, 50, 75, 95]).round(1).tolist(),
            stars=np.percentile([c["star_median"] for c in clips], [5, 25, 50, 75, 95]).round(1).tolist())
    # --- heavy subset: clips whose stars stay extinguished all hour (90th-percentile star count <= 12)
    hv = sorted(k for k, v in st.items() if v["star_p10_p90"][1] <= HEAVY_STAR_P90 and k in ok)
    heavy = dict(clips=hv, drift_px_per_frame=[round(ok[k]["drift_px_per_frame"], 4) for k in hv],
                 heading_deg=[round(ok[k]["heading_deg"], 1) for k in hv],
                 survive_60s=np.percentile([ok[k]["lag60"]["peak_med"] for k in hv], [10, 50, 90]).round(3).tolist(),
                 survive_300s=np.percentile([ok[k]["lag300"]["peak_med"] for k in hv if ok[k]["lag300"]], [10, 50, 90]).round(3).tolist())
    for era in ("A_2015_2018", "B_2019_2024"):
        cl = [st[k] for k in hv if st[k]["era"] == era]
        if cl:
            q = lambda key: np.percentile([c[key] for c in cl], [10, 50, 90]).round(4).tolist()
            z = np.array([c["zone_rgb_median"] for c in cl]).reshape(len(cl), 3, 8, 3)
            heavy[f"envelope_{era}"] = dict(n=len(cl), lum_mean=q("lum_mean"), contrast=q("contrast_median"),
                                            stars=q("star_median"), lum_std_time=q("lum_std_time"),
                                            zenith_rgb=np.median(z[:, 0].mean(1), 0).round(1).tolist(),
                                            rim_rgb=np.median(z[:, 2].mean(1), 0).round(1).tolist())
    prior["heavy"] = heavy
    old = os.path.join(DATA, "prior.json")
    if os.path.exists(old):
        prior["evo_scale"] = json.load(open(old)).get("evo_scale", 1.0)   # keep the calibrated value
    json.dump(prior, open(old, "w"), indent=1)
    print(json.dumps({k: v for k, v in prior.items() if not k.startswith("zone") and k not in ("drift_px_per_frame", "heading_deg", "heavy")}, indent=1))
    print("heavy:", json.dumps({k: v for k, v in heavy.items() if k not in ("heading_deg",)}))


if __name__ == "__main__":
    main()
