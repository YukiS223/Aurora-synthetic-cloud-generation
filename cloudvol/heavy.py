"""Heavy-cloud scenarios: decks that fill the whole sky like fog.

The regimes in weather.py cover skies where cloud and clear sky alternate.
The heavy family covers the other end, measured on the reference clips whose
stars stay extinguished for the whole hour (data/prior.json -> "heavy"):

  * the sky is covered edge to edge and the opacity mask is white almost
    everywhere, so "how heavy" has to be carried by optical thickness and by
    texture, not by coverage;
  * the dome is *bright* (on the 2019+ camera brighter than broken cloud),
    because light is trapped between the snow and the closed cloud base;
  * texture is soft and large: slow swells of thickness, darker thin "wells"
    that open and close, no hard edges;
  * the aurora survives only as glows tens of degrees wide;
  * the deck still drifts and churns - about as fast as lighter cloud.

A heavy deck here is the same kind of object as any other cloud in cloudvol:
a 3-D density field sampled along every camera ray.  It is deep (0.7-1.6 km)
and its density varies gently in all three dimensions instead of being cut
by a sharp threshold, and it is usually accompanied by a second layer - low
fragments racing underneath, or another deck above - so there is parallax.

    scn = sample_heavy_scenario(seed, era)
"""
import numpy as np

from .weather import (Layer, Scenario, era_lighting, load_prior, PX_PER_RAD)

HEAVY_REGIMES = {
    # name: (weight, description)
    "fog": (0.32, "one deep, soft deck: luminous fog with slow swells of thickness"),
    "wells": (0.26, "heavy deck that thins into soft dark wells, then closes again"),
    "stacked": (0.26, "two heavy decks at different heights, moving differently"),
    "scud": (0.16, "heavy deck with a thin soft mist layer sliding underneath"),
}

_FALLBACK_HEAVY = dict(drift_px_per_frame=[0.35, 0.45, 0.55, 0.6, 0.9, 1.3], heading_deg=[-150, -100, -60, 20, 120])


def mean_density_fraction(L, cover, n=40000, seed=0):
    """Expected density / sigma of a layer, by Monte Carlo over its own density law.

    The noise textures are rank-equalised (uniform marginals), so drawing
    uniform numbers for them reproduces the layer's density statistics.  Used
    to turn a target optical depth into an extinction coefficient.
    """
    r = np.random.default_rng(seed)
    c, shp, det, mod, z = r.random((5, n))
    ss = lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - a) / (b - a), 0, 1))
    w = (c - (1.0 - cover)) / L.edge + 0.5
    w = 1.0 / (1.0 + np.exp(-4.0 * (w - 0.5))) if L.soft_ramp else np.clip(w, 0, 1)
    top = w ** L.thick_pow
    prof = ss(0.0, L.base_soft, z) * (1 - ss(L.top_soft * top, top + 1e-4, z))
    avail = ss(0.0, 1.0, w) * prof
    base = np.clip((shp - (1.0 - avail * L.fill)) / L.fall, 0, 1)
    base = np.clip(base - L.erosion * (1 - det) * (1 - 0.75 * base), 0, 1) / (1 - 0.25 * L.erosion)
    base = np.clip(base, 0, 1)
    dens = np.exp(L.core_var * 2.4 * (mod - 0.5)) * (L.veil * base + (1 - L.veil) * base ** L.dens_pow)
    return float(dens.mean())


def _cover_mean(L):
    ts = np.array([c[0] for c in L.cover]); cs = np.array([c[1] for c in L.cover])
    return float(np.trapz(cs, ts) / (ts[-1] - ts[0]))


SIGMA_MAX = 70.0     # 1/km.  Above this a single ray sample is opaque and sampling planes show as contours.


def set_optical_depth(L, tau, sigma_max=SIGMA_MAX):
    """Choose sigma so the layer's mean vertical optical depth is `tau`, without
    exceeding `sigma_max`: a heavy deck is deep and moderately dense, not a thin
    dense sheet (which the ray sampling could not resolve smoothly)."""
    L.sigma = float(min(tau / (L.thickness * max(mean_density_fraction(L, _cover_mean(L)), 1e-3)), sigma_max))
    return L


def _wind(rng, hp, z, speed_scale=1.0, heading=None):
    w = hp["drift_px_per_frame"][rng.integers(len(hp["drift_px_per_frame"]))] * rng.uniform(0.85, 1.2) / PX_PER_RAD
    if heading is None:
        heading = np.radians(hp["heading_deg"][rng.integers(len(hp["heading_deg"]))] + rng.normal(0, 20))
    w *= speed_scale
    return (float(w * z * np.cos(heading)), float(w * z * np.sin(heading))), float(heading)


def _thickness_timeline(rng, kind, T=3600.0):
    """Keyframes of the 'cover' parameter.  For a heavy deck this is a
    thickness dial rather than a sky fraction: >= ~1.05 means no thin spots
    anywhere, ~0.7 opens soft wells where the deck is thinnest."""
    if kind == "closed":
        ts = np.linspace(0, T, 5)
        cs = rng.uniform(1.0, 1.15) + rng.normal(0, 0.04, 5)
    elif kind == "breathing":
        ts = np.linspace(0, T, 7)
        cs = rng.uniform(1.0, 1.08) + rng.uniform(0.04, 0.09) * np.sin(np.linspace(0, rng.uniform(1.5, 3.5) * np.pi, 7) + rng.uniform(0, 6.28))
    else:  # "wells": thins markedly once or twice, then closes
        ts = np.linspace(0, T, 9)
        cs = np.full(9, rng.uniform(1.0, 1.1))
        for _ in range(rng.integers(1, 3)):
            k = rng.integers(1, 8)
            cs[k] = rng.uniform(0.62, 0.76); cs[k - 1] = min(cs[k - 1], rng.uniform(0.82, 0.94)); cs[k + 1] = min(cs[k + 1], rng.uniform(0.82, 0.94))
    return [(float(t), float(np.clip(c, 0.45, 1.3))) for t, c in zip(ts, cs)]


def _deck(rng, hp, e, z, name="deck", timeline="closed", planes=16, tau=None):
    """A deep, soft deck: density varies gently in 3-D instead of being thresholded."""
    U = rng.uniform
    L = Layer(name=name, z_base=float(z), thickness=float(U(0.7, 1.5)), n_planes=planes)
    L.wind, h = _wind(rng, hp, z + 0.15 * L.thickness)
    L.orient = h
    L.cover = _thickness_timeline(rng, timeline)
    # smooth (sigmoid) thickness ramp: thick over most of the field, thin only in the lowest part, no crease
    L.soft_ramp = True
    L.edge, L.thick_pow = float(U(0.8, 1.1)), float(U(0.3, 0.5))
    # Fog fills its slab: with fill ~1.8 and a gentle slope, density is 40-90% of sigma almost
    # everywhere inside the deck and falls smoothly only where the thickness ramp is low.  (A deck
    # made of thin, very dense sheets would show every sampling plane as a contour line.)
    L.fill, L.fall = float(U(1.6, 2.0)), float(U(1.8, 2.4))
    L.erosion, L.dens_pow, L.veil = float(U(0.0, 0.05)), float(U(1.0, 1.4)), float(U(0.25, 0.45))   # no erosion: it would cut hard edges into thin spots
    L.core_var, L.cellular = float(U(0.5, 0.95)), float(U(0.1, 0.4))
    L.cov_beta = float(U(3.3, 3.8))
    # thickness swells a few km across: the view always holds several, so the deck never thins everywhere at once
    L.cov_period = float(U(3.0, 5.0) * (z + L.thickness))
    L.shape_period, L.shape_vperiod, L.detail_period = float(U(3.0, 6.5)), float(U(1.4, 2.4)), float(U(0.9, 1.6))
    L.aniso = float(U(1.0, 1.8))
    # heavy decks hold their pattern longer than broken cloud (83% vs 76% kept after a minute)
    L.evo_speed = float(U(0.0005, 0.0013) * e)
    L.boil = float(U(0.0002, 0.0005) * e)
    L.shear = (float(U(-0.0007, 0.0007) * e), float(U(-0.0007, 0.0007) * e))
    L.top_soft = float(U(0.6, 0.75))               # broad plateau: the deck is full for most of its depth
    # diffuse base: density builds up over the lower half of the slab, so the first optical depth is
    # spread over several samples.  An abrupt base makes each sampling plane show as a contour line.
    L.base_soft = float(U(0.3, 0.45))
    L.cov_wind_scale = float(U(0.25, 0.6))         # swells of thickness move slower than the texture
    set_optical_depth(L, tau if tau is not None else U(14, 40))
    # residual haze through the slab: about one optical depth overhead (more toward the horizon), so the
    # thinnest spots dim the stars instead of baring them; the "wells" timeline uses less
    hz = U(0.3, 0.9) if timeline == "wells" else U(0.8, 2.0)
    L.haze = float(hz / (L.sigma * L.thickness * 0.667))
    return L


def _scud(rng, hp, e, deck):
    """A thin, soft mist layer under the deck, crossing the sky faster than it.

    It is optically thin (well under one optical depth) and has no edges; what
    it adds is parallax - faint veils sliding across the swells of the deck."""
    U = rng.uniform
    z = float(deck.z_base * U(0.5, 0.8))
    L = Layer(name="mist", z_base=z, thickness=float(U(0.15, 0.3)), n_planes=8)
    h0 = float(np.arctan2(deck.wind[1], deck.wind[0]))
    L.wind, h = _wind(rng, hp, z, speed_scale=U(1.2, 1.7), heading=h0 + rng.normal(0, 0.3))
    L.orient = h
    c0 = U(0.45, 0.75)
    L.cover = [(0.0, float(c0)), (1800.0, float(np.clip(c0 + rng.normal(0, 0.1), 0.3, 0.85))), (3600.0, float(np.clip(c0 + rng.normal(0, 0.1), 0.3, 0.85)))]
    L.soft_ramp = True
    L.edge, L.thick_pow = float(U(0.7, 1.0)), 0.8
    L.fill, L.fall = float(U(1.4, 1.8)), float(U(1.8, 2.6))
    L.erosion, L.dens_pow, L.veil = 0.0, float(U(1.0, 1.3)), float(U(0.35, 0.5))
    L.core_var, L.cellular = float(U(0.3, 0.6)), float(U(0.1, 0.3))
    L.cov_beta = 3.4
    L.cov_period = float(U(5.0, 8.0) * z)
    L.shape_period, L.shape_vperiod, L.detail_period = float(U(1.8, 3.2)), float(U(0.8, 1.2)), float(U(0.6, 1.0))
    L.top_soft = 0.3
    L.base_soft = 0.35
    L.evo_speed = float(U(0.0006, 0.0014) * e)
    L.boil = float(U(0.0003, 0.0006) * e)
    L.shear = (float(U(-0.0006, 0.0006) * e), float(U(-0.0006, 0.0006) * e))
    return set_optical_depth(L, U(0.3, 0.9), sigma_max=40.0)


def sample_heavy_scenario(seed, era="A", regime=None):
    rng = np.random.default_rng([int(seed), 7041])        # independent of the light-cloud stream
    prior = load_prior()
    hp = prior.get("heavy", _FALLBACK_HEAVY)
    e = prior.get("evo_scale", 1.0)
    U = rng.uniform
    names = list(HEAVY_REGIMES)
    if regime is None:
        p = np.array([HEAVY_REGIMES[n][0] for n in names]); regime = str(rng.choice(names, p=p / p.sum()))
    if regime not in HEAVY_REGIMES:
        raise ValueError(f"unknown heavy regime {regime!r}; choose from {names}")
    z = U(0.35, 1.3)
    if regime == "fog":
        layers = [_deck(rng, hp, e, z, timeline=str(rng.choice(["closed", "breathing"])), planes=18)]
    elif regime == "scud":
        deck = _deck(rng, hp, e, U(0.6, 1.4), timeline=str(rng.choice(["closed", "breathing", "wells"])))
        layers = [_scud(rng, hp, e, deck), deck]
    elif regime == "stacked":
        low = _deck(rng, hp, e, z, name="low deck", timeline="breathing", planes=14, tau=U(6, 16))
        low.thickness = float(min(low.thickness, 1.0)); set_optical_depth(low, U(6, 16))
        low.haze = float(U(0.5, 1.2) / (low.sigma * low.thickness * 0.667))
        up = _deck(rng, hp, e, low.z_base + low.thickness + U(0.5, 1.8), name="upper deck", timeline="closed", planes=10, tau=U(8, 22))
        layers = [low, up]
    else:  # wells
        layers = [_deck(rng, hp, e, z, timeline="wells", planes=18, tau=U(12, 26))]
    lighting = era_lighting(era, rng)
    lighting.tau_deep = float(U(14, 26))
    lighting.bounce = float(U(0.6, 0.76))
    if era != "A":
        # the 2019+ camera's heavy decks are uniformly bright (dome luminance 119-143 in the references):
        # draw the upwelling skyglow from the upper part of the era's range instead of all of it
        k3 = lighting.ambient[0] / 0.15
        k3_new = float(U(0.62, 0.95))
        # ... and their colour is creamier than the era's average cloudy sky (zenith 132/125/104)
        lighting.ambient = (0.150 * k3_new, 0.131 * k3_new, 0.087 * k3_new)
    lighting.relight_foreground = True
    return Scenario(seed=int(seed), regime=f"heavy/{regime}", layers=layers, lighting=lighting, sub=3)
