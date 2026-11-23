"""Measure cloud drift and evolution in the reference clips.

Drift: for frame pairs 60 s apart, the zenith region (static pattern removed,
stars filtered) of the later frame is matched against the earlier one by
normalised cross-correlation over a +-88 px search (cv2.matchTemplate).  The
peak position is the drift; the peak height says how much of the cloud field
survived the minute unchanged (evolution).  The same at 5 min lag.
"""
import glob, json, os, sys
import numpy as np
import cv2
from .vio import read_frames
from .geometry import R90

STEP = 5
HALF, SEARCH = 56, 44            # half-res pixels: template +-112 px, search +-88 px at full res


def analyse(path):
    fr = []
    for _, f in read_frames(path, step=STEP):
        g = cv2.medianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 5)        # stars / hot pixels out
        g = cv2.resize(g, (304, 304), interpolation=cv2.INTER_AREA).astype(np.float32)
        fr.append(cv2.GaussianBlur(g, (0, 0), 1.5))
    fr = np.stack(fr)
    fr -= np.median(fr, 0)                                                # static pattern out
    c = 148
    out = {}
    for lag_frames in (20, 60, 300):
        L = lag_frames // STEP
        sh, pk, zero = [], [], []
        for t in range(0, len(fr) - L, max(1, L // 2)):
            tpl = fr[t, c - HALF:c + HALF, c - HALF:c + HALF]
            win = fr[t + L, c - HALF - SEARCH:c + HALF + SEARCH, c - HALF - SEARCH:c + HALF + SEARCH]
            if tpl.std() < 0.5:                                           # featureless: nothing to track
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            dx, dy = loc[0] - SEARCH, loc[1] - SEARCH
            sh.append((dx * 2 / lag_frames, dy * 2 / lag_frames)); pk.append(mx); zero.append(res[SEARCH, SEARCH])
        if not sh:
            out[f"lag{lag_frames}"] = None
            continue
        sh, pk, zero = np.array(sh), np.array(pk), np.array(zero)
        good = pk > 0.5
        spd = np.hypot(sh[:, 0], sh[:, 1])
        out[f"lag{lag_frames}"] = dict(
            n=len(sh), frac_tracked=float(good.mean()),
            peak_med=float(np.median(pk)), zero_shift_med=float(np.median(zero)),
            shift_med=[float(x) for x in (np.median(sh[good], 0) if good.any() else (0, 0))],
            speed_med=float(np.median(spd[good])) if good.any() else None,
            speed_p10_p90=[float(x) for x in np.percentile(spd[good], [10, 90])] if good.any() else None)
    l60, l20 = out["lag60"], out["lag20"]
    # fast cloud can leave the 60 s search window: fall back to the 20 s lag
    best = l60
    if l20 and l20["speed_med"] is not None and (l20["speed_med"] > 1.1 or not l60 or l60["speed_med"] is None
                                                 or l60["frac_tracked"] < 0.5 * l20["frac_tracked"]):
        best = l20
    if best and best["speed_med"] is not None:
        out["drift_px_per_frame"] = best["speed_med"]
        out["drift_mrad_per_s"] = best["speed_med"] / (R90 / (np.pi / 2)) * 1000
        out["heading_deg"] = float(np.degrees(np.arctan2(best["shift_med"][1], best["shift_med"][0])))
        out["frac_tracked"] = best["frac_tracked"]
    out["texture_std"] = float(np.median(fr[:, c - HALF:c + HALF, c - HALF:c + HALF].std((1, 2))))
    return out


def main(folder, out, only=None):
    res = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.avi")))
    for k, p in enumerate(paths):
        name = os.path.basename(p).replace("Test ", "").replace(".avi", "")
        if only and name not in only:
            continue
        try:
            d = analyse(p)
        except Exception as e:
            print("FAILED", name, repr(e), flush=True)
            continue
        res[name] = d
        l60, l300 = d["lag60"], d["lag300"]
        print(f"[{k+1}/{len(paths)}] {name} drift={d.get('drift_px_per_frame', float('nan')):.3f} px/fr "
              f"tracked={l60['frac_tracked'] if l60 else 0:.2f} peak60={l60['peak_med'] if l60 else 0:.2f} "
              f"peak300={l300['peak_med'] if l300 else 0:.2f}", flush=True)
        if not only:
            json.dump(res, open(out, "w"))
    print("done", len(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None)
