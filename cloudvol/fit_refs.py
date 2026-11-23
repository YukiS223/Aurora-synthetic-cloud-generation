"""Measure the real cloudy reference clips ("training" pass).

For every reference clip this extracts, from sparsely sampled frames:
  * cloud drift near the zenith (robust optical flow -> angular speed, heading)
  * colour by sky zone (3 zenith rings x 8 azimuth sectors)
  * mid-scale structure contrast (band-passed luminance std / mean)
  * star visibility proxy (cover) and its variability over the hour
Results go to cloudvol/data/ref_stats.json and drive the scenario prior
(weather.py) and the realism checks (validate.py).
"""
import glob, json, os, sys
import numpy as np
import cv2
from .vio import read_frames
from .geometry import pixel_grid, CX, CY, R90

STEP = 10          # sample every 10th frame (10 s of real time)


def era_of(name):
    year = int(name.split("_")[0][-4:])
    return "A_2015_2018" if year <= 2018 else "B_2019_2024"


def analyse(path):
    r, theta, psi = pixel_grid()
    dome = r < 0.93 * R90
    core = r < 0.42 * R90                      # zenith angle < 38 deg
    ring_id = np.digitize(r / R90, [0.35, 0.7, 0.93])        # 0,1,2 inside dome
    sec_id = ((psi + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    zone = np.where(dome, ring_id * 8 + sec_id, -1)
    small_core = cv2.resize(core.astype(np.uint8), (304, 304)) > 0

    prev = None
    flows, zones, lum, contrast, stars = [], [], [], [], []
    hot_acc, n = np.zeros((608, 608), np.float32), 0
    for i, f in read_frames(path, step=STEP):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        # star / hot-pixel proxy: sharp positive outliers against a 5x5 median
        med = cv2.medianBlur(g, 5)
        pts = (g.astype(np.int16) - med) > 22
        hot_acc += pts; n += 1
        stars.append(pts)
        fl = f.astype(np.float32)
        zones.append([fl[zone == k].mean(0) for k in range(24)])
        lum.append(float(med[dome].mean()))
        # mid-scale contrast: DoG band 6..24 px of the star-free luminance
        m = med.astype(np.float32)
        band = cv2.GaussianBlur(m, (0, 0), 6) - cv2.GaussianBlur(m, (0, 0), 24)
        contrast.append(float(band[core].std() / (m[core].mean() + 1)))
        s = cv2.resize(cv2.GaussianBlur(med, (0, 0), 2), (304, 304), interpolation=cv2.INTER_AREA)
        if prev is not None:
            fw = cv2.calcOpticalFlowFarneback(prev, s, None, 0.5, 4, 25, 3, 7, 1.5, 0)
            # texture weight so flat regions do not vote
            gx = cv2.Sobel(prev, cv2.CV_32F, 1, 0); gy = cv2.Sobel(prev, cv2.CV_32F, 0, 1)
            wgt = np.hypot(gx, gy)[small_core]
            v = fw[small_core]
            keep = wgt > np.percentile(wgt, 60)
            flows.append(np.median(v[keep], 0) * 2.0 / STEP)     # full-res px / frame
        prev = s
    hot = hot_acc / n > 0.5
    star_counts = np.array([int((p & ~hot & core).sum()) for p in stars])
    flows = np.array(flows)
    med_flow = np.median(flows, 0)
    px_per_rad = R90 / (np.pi / 2)
    speed = float(np.hypot(*med_flow))
    # how consistent the heading is across the hour (1 = steady translation)
    unit = flows / (np.hypot(flows[:, 0], flows[:, 1])[:, None] + 1e-6)
    zones = np.array(zones)                                        # T x 24 x 3
    return dict(
        frames=n * STEP,
        drift_px_per_frame=speed,
        drift_mrad_per_s=speed / px_per_rad * 1000.0,
        drift_heading_deg=float(np.degrees(np.arctan2(med_flow[1], med_flow[0]))),
        drift_steadiness=float(np.hypot(*unit.mean(0))),
        flow_speed_p25_p75=[float(x) for x in np.percentile(np.hypot(flows[:, 0], flows[:, 1]), [25, 75])],
        zone_rgb_median=np.median(zones, 0).round(1).tolist(),
        zone_rgb_p10=np.percentile(zones, 10, axis=0).round(1).tolist(),
        zone_rgb_p90=np.percentile(zones, 90, axis=0).round(1).tolist(),
        lum_mean=float(np.mean(lum)), lum_std_time=float(np.std(lum)),
        lum_series=[round(x, 1) for x in lum[::6]],
        contrast_median=float(np.median(contrast)),
        contrast_p10_p90=[float(x) for x in np.percentile(contrast, [10, 90])],
        star_median=float(np.median(star_counts)),
        star_p10_p90=[float(x) for x in np.percentile(star_counts, [10, 90])],
        star_series=[int(x) for x in star_counts[::6]],
    )


def main(folder, out):
    res = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.avi")))
    for k, p in enumerate(paths):
        name = os.path.basename(p).replace("Test ", "").replace(".avi", "")
        try:
            d = analyse(p)
        except Exception as e:      # keep going; report at the end
            print("FAILED", name, e, flush=True)
            continue
        d["era"] = era_of(name)
        res[name] = d
        print(f"[{k+1}/{len(paths)}] {name} drift={d['drift_mrad_per_s']:.2f} mrad/s "
              f"steady={d['drift_steadiness']:.2f} lum={d['lum_mean']:.0f} "
              f"contrast={d['contrast_median']:.3f} stars={d['star_median']:.0f}", flush=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"))
    print("done", len(res), "clips")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
