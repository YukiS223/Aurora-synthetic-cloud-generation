"""Tileable procedural noise textures (numpy, generated once per clip seed).

Three textures drive the cloud volume:
  coverage  2-D, smooth, large scale   -> where cloud exists at all
  shape     3-D "Perlin-Worley"        -> billows / cells inside the cloud
  detail    3-D Worley fBm             -> wisps that erode the edges
All are periodic in every axis and rank-equalised to a uniform [0,1]
marginal, so coverage thresholds translate directly into sky fractions.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi


def equalize(x):
    """Rank-transform to a uniform [0,1] marginal."""
    flat = x.ravel()
    order = np.argsort(flat, kind="stable")
    out = np.empty_like(flat, dtype=np.float32)
    out[order] = (np.arange(flat.size, dtype=np.float32) + 0.5) / flat.size
    return out.reshape(x.shape)


def spectral(shape, beta, rng, kmin=1.0, kmax=None, stretch=None):
    """Periodic Gaussian random field with power spectrum ~ k^-beta.

    shape may be 2-D or 3-D.  `stretch` scales each axis' frequency (values
    > 1 make the field vary faster along that axis in texture units).
    """
    freqs = [np.fft.fftfreq(n) * n for n in shape]
    grids = np.meshgrid(*freqs, indexing="ij")
    if stretch is not None:
        grids = [g / s for g, s in zip(grids, stretch)]
    k = np.sqrt(sum(g * g for g in grids))
    amp = np.zeros_like(k)
    ok = k >= kmin
    if kmax is not None:
        ok &= k <= kmax
    amp[ok] = k[ok] ** (-beta / 2.0)
    spec = amp * np.exp(2j * np.pi * rng.random(k.shape))
    f = np.fft.ifftn(spec).real
    return (f / f.std()).astype(np.float32)


def worley(shape, cells, rng):
    """Periodic inverted Worley noise: 1 at feature points, falling to 0."""
    n_pts = int(np.prod(cells))
    # jittered lattice of feature points in the unit torus
    idx = np.stack(np.meshgrid(*[np.arange(c) for c in cells], indexing="ij"), -1).reshape(-1, len(cells))
    pts = (idx + rng.random(idx.shape)) / np.array(cells)
    tree = cKDTree(pts % 1.0, boxsize=1.0)
    axes = [(np.arange(n) + 0.5) / n for n in shape]
    q = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(shape))
    # distances measured in units of the cell size along each axis
    d, _ = tree.query(q, k=1)
    d = d.reshape(shape) * (n_pts ** (1.0 / len(shape)))
    return np.clip(1.0 - d / 0.9, 0.0, 1.0).astype(np.float32)


def warp2d(tex, rng, amp_px, beta=3.5):
    """Periodic domain warp of a 2-D texture by a smooth random vector field."""
    n = tex.shape[0]
    wx = spectral((n, n), beta, rng, kmin=1) * amp_px
    wy = spectral((n, n), beta, rng, kmin=1) * amp_px
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return ndi.map_coordinates(tex, [(yy + wy) % n, (xx + wx) % n], order=1, mode="grid-wrap").astype(np.float32)


def coverage_texture(rng, n=512, beta=3.4, warp=0.06, channels=2, kmax=None):
    """Smooth large-scale fields; two independent channels drift past each
    other at render time so cloud banks change shape instead of only sliding.
    The coverage field is extruded vertically through the layer, so `kmax`
    must keep its finest wavelength above the layer thickness - anything finer
    renders as vertical curtains (radial streaks in the fisheye)."""
    out = []
    for _ in range(channels):
        f = spectral((n, n), beta, rng, kmin=2, kmax=kmax or n / 5)
        if warp > 0:
            f = warp2d(f, rng, warp * n)
        out.append(equalize(f))
    return np.stack(out)


def shape_texture(rng, n=128, nz=32, cellular=0.5, beta=3.0, cells=5):
    """3-D base shape, [nz, n, n].  `cellular` in [0,1] mixes smooth fBm
    (stratiform) with packed Worley billows (cumuliform)."""
    g = spectral((nz, n, n), beta, rng, kmin=2, kmax=n / 5, stretch=(nz / n * 2.0, 1.0, 1.0))
    cz = max(2, int(round(cells * nz / n * 2)))
    w = (0.60 * worley((nz, n, n), (cz, cells, cells), rng)
         + 0.28 * worley((nz, n, n), (cz * 2, cells * 2, cells * 2), rng)
         + 0.12 * worley((nz, n, n), (cz * 4, cells * 4, cells * 4), rng))
    w = (w - w.mean()) / w.std()
    return equalize((1.0 - cellular) * g + cellular * w)


def detail_texture(rng, n=64, nz=64, cells=4):
    """3-D erosion noise, [nz, n, n], cubic tile (isotropic wisps, no vertical curtains).
    Only octaves the grid resolves well: the texture is magnified near the zenith."""
    w = (0.62 * worley((nz, n, n), (cells, cells, cells), rng)
         + 0.38 * worley((nz, n, n), (cells * 2, cells * 2, cells * 2), rng))
    g = spectral((nz, n, n), 2.6, rng, kmin=3, kmax=n / 5)
    return equalize(w / w.std() + 0.5 * g)
