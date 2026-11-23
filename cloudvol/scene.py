"""Per-clip scene statics: where the sky is, which pixels are sensor defects.

Sky mask.  The camera is fixed, so clips recorded within a few weeks share the
same foreground (terrain, instrument dome, masts, lens rim).  Foreground
outlines are edges whose gradient keeps the same sign in every frame, whereas
aurora, stars and real clouds average out.  We pool the signed log-luminance
gradient over the target clip and its "sibling" clips (same camera era, nearby
dates, from both the clear and the cloudy folders); the magnitude of that mean
is a clean map of static outlines.  Walking outwards from the zenith, the
first static outline in each azimuth is the edge of the visible sky.  Clouds
are composited only inside it.

Hot pixels.  Sensor defects are bright in (almost) every frame at the same
pixel and strongly coloured; they belong to the sensor, not the sky, so they
must survive under cloud.
"""
import datetime as dt
import glob
import os
import numpy as np
import cv2
from scipy import ndimage as ndi
from .vio import read_frames
from .geometry import pixel_grid, R90, CX, CY

DATA = os.path.join(os.path.dirname(__file__), "data")
EDGE_THR = 0.15          # persistent-edge strength; sky interior ~0.03, horizon line >= 0.16


def clip_date(path):
    stem = os.path.basename(path).replace("Test ", "").split("_")[0]
    return dt.date(int(stem[4:8]), int(stem[2:4]), int(stem[0:2]))


def era_of(date):
    return "A" if date.year <= 2018 else "B"


def siblings(path, folders, max_days=45):
    d0 = clip_date(path)
    out = []
    for fo in folders:
        for p in sorted(glob.glob(os.path.join(fo, "*.avi"))):
            d = clip_date(p)
            if era_of(d) == era_of(d0) and abs((d - d0).days) <= max_days:
                out.append(p)
    return out


def _grad_sum(path, step):
    gs = np.zeros((608, 608, 2)); n = 0
    for _, f in read_frames(path, step=step):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(np.log1p(g), (0, 0), 1.5)
        gs[..., 0] += cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gs[..., 1] += cv2.Sobel(g, cv2.CV_32F, 0, 1)
        n += 1
    return gs, n


def _own_scan(path, step=3):
    """Hot-pixel statistics, mean frame and noise level of the target clip."""
    n = 0
    out_freq = np.zeros((608, 608)); mean = np.zeros((608, 608, 3)); resid = np.zeros((608, 608, 3))
    prev = None; nacc = []
    for _, f in read_frames(path, step=step):
        med = cv2.medianBlur(f, 5)
        d = f.astype(np.float32) - med
        out_freq += d.max(-1) > 20
        resid += d; mean += f; n += 1
        hp = f.astype(np.float32) - cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5)
        nacc.append(np.median(np.abs(hp[200:400, 200:400]), axis=(0, 1)) * 1.4826)
    return dict(out_freq=out_freq / n, mean=mean / n, resid=resid / n,
                noise_sigma=np.median(np.array(nacc), 0))


def sky_from_edges(pers):
    """Soft sky mask from the persistent-edge map."""
    r, _, psi = pixel_grid()
    e = (pers > EDGE_THR).astype(np.uint8)
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    big = np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] >= 60)[0][1:])     # drops hot-pixel rings
    NAZ, R0 = 720, 140
    az = (np.arange(NAZ) + 0.5) / NAZ * 2 * np.pi - np.pi
    rs = np.arange(R0, int(0.985 * R90))
    px = np.clip(np.round(CX + rs[None] * np.cos(az[:, None])).astype(int), 0, 607)
    py = np.clip(np.round(CY + rs[None] * np.sin(az[:, None])).astype(int), 0, 607)
    hit = big[py, px]
    first = np.where(hit.any(1), hit.argmax(1), len(rs) - 1)
    rh = rs[first].astype(np.float32)
    # circular min-then-median: a thin outline can be missed by single rays
    rh3 = np.r_[rh, rh, rh]
    rh3 = ndi.minimum_filter1d(rh3, 3)
    rh3 = ndi.median_filter(rh3, 9)
    rh = ndi.gaussian_filter1d(rh3, 1.0)[NAZ:2 * NAZ] - 3.0
    k = ((psi + np.pi) / (2 * np.pi) * NAZ).astype(int) % NAZ
    core = (r < rh[k]).astype(np.float32)
    return cv2.GaussianBlur(core, (0, 0), 1.5), rh


def build(path, folders, cache=True, verbose=True):
    name = os.path.basename(path).replace(".avi", "")
    cpath = os.path.join(DATA, f"scene_{name}.npz")
    if cache and os.path.exists(cpath):
        return dict(np.load(cpath))
    sibs = siblings(path, folders)
    if os.path.abspath(path) not in [os.path.abspath(p) for p in sibs]:
        sibs.append(path)
    gs = np.zeros((608, 608, 2)); n = 0
    for p in sibs[:16]:
        if verbose:
            print("  static-outline evidence from", os.path.basename(p), flush=True)
        g, k = _grad_sum(p, step=12)
        gs += g; n += k
    pers = (np.hypot(gs[..., 0], gs[..., 1]) / n).astype(np.float32)
    sky, rh = sky_from_edges(pers)
    own = _own_scan(path)
    res = own["resid"]
    chroma = res.max(-1) - res.min(-1)
    hot = (own["out_freq"] > 0.6) & (chroma > 0.45 * np.maximum(res.max(-1), 1)) & (sky > 0.5)
    hot = cv2.dilate(hot.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = dict(sky=sky.astype(np.float32), hot=hot, edges=pers, horizon_r=rh,
               mean=own["mean"].astype(np.float32),
               noise_sigma=own["noise_sigma"].astype(np.float32),
               n_siblings=np.int32(len(sibs)))
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(cpath, **out)
    return out
