"""Scenario description: cloud layers, their evolution, and the lighting.

A scenario is plain data (JSON-serialisable) so every rendered clip can be
reproduced from its sidecar file.  `sample_scenario` draws one from priors
measured on the real cloudy clips (cloudvol/data/prior.json).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Layer:
    name: str = "deck"
    z_base: float = 1.5            # km above the camera
    thickness: float = 0.6         # km, maximum geometric thickness
    n_planes: int = 12             # altitude samples through the layer
    sigma: float = 14.0            # extinction (1/km) of fully formed cloud
    # coverage over time: (time_s, sky fraction) keyframes, linearly interpolated
    cover: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.5), (3600.0, 0.5)])
    edge: float = 0.30             # coverage margin over which cloud builds to full thickness
    thick_pow: float = 0.8         # how fast the cloud top rises away from the edge
    fill: float = 1.05             # >1: solid cores; <1: porous throughout
    fall: float = 0.35             # softness of the density threshold
    erosion: float = 0.45          # strength of wispy edge erosion
    dens_pow: float = 2.0          # >1: thin translucent margins, dense cores
    veil: float = 0.08             # fraction of the extinction present as thin haze around the cores
    core_var: float = 0.6          # 0..1, mid-scale variation of density (dark thick cores vs thin sheets)
    top_soft: float = 0.7          # cloud top fades between top_soft*top and top (lower = more gradual top)
    base_soft: float = 0.14        # fraction of the slab over which density builds up from the cloud base
    haze: float = 0.0              # fraction of sigma present as uniform haze through the whole slab
    soft_ramp: bool = False        # thickness ramp saturates smoothly (sigmoid) instead of clamping: no creases
    cov_wind_scale: float = 1.0    # the coverage / thickness pattern travels at this fraction of the wind
    cov_period: float = 36.0       # km, tile size of the coverage field
    shape_period: float = 5.0      # km, tile size of the 3-D shape noise
    shape_vperiod: float = 2.4     # km, vertical tile of the shape noise
    detail_period: float = 1.1     # km
    cellular: float = 0.5          # 0 stratiform .. 1 cumuliform
    cov_beta: float = 3.4          # spectral slope of the coverage field
    aniso: float = 1.0             # >1 stretches structure along `orient` (streets, fibres)
    orient: float = 0.0            # radians, image frame
    wind: Tuple[float, float] = (0.002, 0.0)      # km/s, image frame (x right, y down)
    evo_speed: float = 0.0006      # km/s, relative drift of the two coverage fields
    boil: float = 0.00012          # 1/s, vertical drift of the shape noise (in tiles)
    shear: Tuple[float, float] = (0.0003, 0.0)    # km/s, drift of detail relative to shape
    # optional front: cover += amp * tanh((x.n - (pos0 + speed t)) / width)
    front: Optional[Tuple[float, float, float, float, float]] = None   # amp, dir_rad, pos0_km, speed_km_s, width_km


@dataclass
class Source:
    """A ground light (settlement, mine, facility) lighting cloud bases from below."""
    x: float                       # km, image frame, relative to the camera
    y: float
    color: Tuple[float, float, float]   # linear RGB radiance of thick cloud directly above, at 1.5 km
    kappa: float = 0.25            # shadowing strength along the slanted light path


@dataclass
class Lighting:
    sources: List[Source] = field(default_factory=list)
    ambient: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # diffuse upwelling skyglow (linear RGB)
    snow_albedo: float = 0.45      # ground bounce of sky light back onto cloud bases
    kappa: float = 0.16            # diffuse attenuation per unit optical depth (~0.75 (1 - g))
    tau_iso: float = 5.0           # optical depth over which transmitted light loses its direction
    # --- heavy-cloud extensions; the defaults switch them off so earlier scenarios render unchanged
    tau_deep: float = 0.0          # >0: optical depth over which a transmitted glow spreads from tens of degrees to the whole sky
    bounce: float = 0.0            # snow<->cloud multiple reflection of upwelling light under a closed deck (0..~0.85)
    relight_foreground: bool = False   # re-light terrain and lens rim with the clouded sky's light


@dataclass
class Scenario:
    seed: int = 0
    regime: str = "custom"
    layers: List[Layer] = field(default_factory=list)
    lighting: Lighting = field(default_factory=Lighting)
    frame_dt: float = 1.0          # seconds of real time per video frame
    sub: int = 0                   # density samples per slab along the ray (0 = renderer default)

    def to_json(self):
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_dict(d):
        layers = [Layer(**{**l, "cover": [tuple(c) for c in l["cover"]],
                           "wind": tuple(l["wind"]), "shear": tuple(l["shear"]),
                           "front": tuple(l["front"]) if l.get("front") else None}) for l in d["layers"]]
        lg = d["lighting"]
        lighting = Lighting(sources=[Source(**{**s, "color": tuple(s["color"])}) for s in lg["sources"]],
                            ambient=tuple(lg["ambient"]), snow_albedo=lg["snow_albedo"],
                            kappa=lg["kappa"], tau_iso=lg["tau_iso"],
                            tau_deep=lg.get("tau_deep", 0.0), bounce=lg.get("bounce", 0.0),
                            relight_foreground=lg.get("relight_foreground", False))
        return Scenario(seed=d["seed"], regime=d["regime"], layers=layers, lighting=lighting,
                        frame_dt=d.get("frame_dt", 1.0), sub=d.get("sub", 0))


def cover_at(layer, t):
    ts = np.array([c[0] for c in layer.cover]); cs = np.array([c[1] for c in layer.cover])
    return float(np.interp(t, ts, cs))


# ----------------------------------------------------------------------------
# Scenario sampling
# ----------------------------------------------------------------------------
PX_PER_RAD = 262.0 / (np.pi / 2)

# Fallback prior, used only if cloudvol/data/prior.json has not been fitted yet.
_FALLBACK_PRIOR = dict(drift_px_per_frame=[0.2, 0.35, 0.5, 0.65, 0.9], heading_deg=[-150, -100, -60, 20, 120],
                       evo_scale=1.0)

REGIMES = {
    # name: (weight, description)
    "broken": (0.24, "broken stratocumulus: bright banks with clear gaps"),
    "overcast": (0.18, "thick deck with a few holes"),
    "veil": (0.14, "thin high sheet; aurora diffuses through, stars fade"),
    "patchy": (0.12, "scattered low fragments"),
    "rolls": (0.10, "cloud streets / wave rows"),
    "two_layer": (0.12, "thin high sheet above broken low cloud, different winds"),
    "front": (0.10, "a deck edge crossing the sky during the clip"),
}


def load_prior():
    p = os.path.join(DATA, "prior.json")
    if os.path.exists(p):
        return json.load(open(p))
    return _FALLBACK_PRIOR


def era_lighting(era, rng):
    """Ground lighting typical of the camera era (colours in linear camera RGB)."""
    j = lambda lo, hi: float(rng.uniform(lo, hi))
    if era == "A":      # 2015-2018 camera: weak yellow-green ground glow, sky light dominates
        # the nearby yellow lobe is prominent in only a few reference clips: usually weak
        k1, k2 = 1.2 * j(0.15, 1.0) ** 2, j(0.5, 1.5)
        return Lighting(sources=[Source(x=0.1, y=-1.5, color=(0.36 * k1, 0.42 * k1, 0.09 * k1)),
                                 Source(x=-4.0, y=-11.0, color=(2.5 * k2, 2.6 * k2, 0.9 * k2))],
                        ambient=(0.012, 0.018, 0.012))
    k1, k2, k3 = j(0.4, 1.3), j(0.5, 1.5), j(0.3, 1.2)   # 2019+ camera: warm skyglow over the whole dome
    return Lighting(sources=[Source(x=0.2, y=-1.6, color=(0.30 * k1, 0.24 * k1, 0.10 * k1)),
                             Source(x=-4.0, y=-11.0, color=(4.0 * k2, 3.4 * k2, 2.2 * k2))],
                    ambient=(0.15 * k3, 0.115 * k3, 0.068 * k3))


def _timeline(rng, c0, kind=None, T=3600.0):
    """Cover-fraction keyframes over the clip."""
    kind = kind or rng.choice(["steady", "trend", "pulse"], p=[0.35, 0.35, 0.30])
    if kind == "steady":
        ts = np.linspace(0, T, 5)
        cs = c0 + rng.normal(0, 0.05, 5)
    elif kind == "trend":
        d = rng.uniform(0.25, 0.5) * rng.choice([-1, 1])
        ts = np.array([0, rng.uniform(0.2, 0.45) * T, rng.uniform(0.6, 0.85) * T, T])
        cs = c0 + d * np.array([-0.5, -0.4, 0.4, 0.5]) + rng.normal(0, 0.03, 4)
    else:
        tp = rng.uniform(0.3, 0.7) * T
        w = rng.uniform(0.12, 0.25) * T
        d = rng.uniform(0.25, 0.45) * rng.choice([-1, 1])
        ts = np.array([0, tp - w, tp, tp + w, T])
        cs = c0 + np.array([0, 0, d, 0, 0]) + rng.normal(0, 0.03, 5)
    return [(float(t), float(np.clip(c, 0.03, 0.99))) for t, c in zip(ts, cs)]


def _wind(rng, prior, z):
    """Wind (km/s) such that the angular drift at the zenith matches a measured clip."""
    i = rng.integers(len(prior["drift_px_per_frame"]))
    w = prior["drift_px_per_frame"][i] * rng.uniform(0.8, 1.25) / PX_PER_RAD     # rad/s
    h = np.radians(prior["heading_deg"][rng.integers(len(prior["heading_deg"]))] + rng.normal(0, 20))
    return (float(w * z * np.cos(h)), float(w * z * np.sin(h))), float(h)


def _low_layer(rng, prior, kind):
    U = rng.uniform
    e = prior.get("evo_scale", 1.0)
    z = U(0.9, 2.4)
    L = Layer(name=kind, z_base=float(z), n_planes=14)
    L.wind, h = _wind(rng, prior, z)
    L.orient = h
    L.evo_speed = float(U(0.0008, 0.002) * e)
    L.boil = float(U(0.0004, 0.0009) * e)
    L.shear = (float(U(-0.0012, 0.0012) * e), float(U(-0.0012, 0.0012) * e))
    L.core_var = float(U(0.35, 0.9))
    L.shape_period = float(U(2.4, 4.2)); L.detail_period = float(U(0.5, 0.9))
    L.shape_vperiod = float(U(1.3, 2.2)); L.cov_period = float(U(5.0, 9.0) * z)
    if kind == "broken":
        L.thickness, L.sigma = float(U(0.5, 0.9)), float(U(18, 38))
        L.cellular, L.fill, L.fall = float(U(0.3, 0.58)), float(U(0.8, 0.95)), float(U(0.8, 1.1))
        L.edge, L.erosion, L.dens_pow, L.veil = float(U(0.25, 0.45)), float(U(0.5, 0.75)), float(U(2.0, 3.0)), float(U(0.08, 0.16))
        L.cover = _timeline(rng, U(0.4, 0.72))
    elif kind == "overcast":
        L.thickness, L.sigma = float(U(0.6, 1.1)), float(U(24, 45))
        L.cellular, L.fill, L.fall = float(U(0.3, 0.55)), float(U(0.92, 1.08)), float(U(0.85, 1.15))
        L.edge, L.erosion, L.dens_pow, L.veil = float(U(0.32, 0.5)), float(U(0.5, 0.72)), float(U(2.0, 2.8)), float(U(0.1, 0.2))
        L.cover = _timeline(rng, U(0.82, 0.95), kind=rng.choice(["steady", "pulse"]))
        L.cov_beta = 3.0; L.cov_period = float(U(8.0, 14.0) * z)
    elif kind == "patchy":
        z = U(0.7, 1.5); L.z_base = float(z); L.wind, h = _wind(rng, prior, z); L.orient = h
        L.thickness, L.sigma = float(U(0.35, 0.6)), float(U(18, 32))
        L.cellular, L.fill, L.fall = float(U(0.55, 0.8)), float(U(0.75, 0.9)), float(U(0.6, 0.85))
        L.edge, L.erosion, L.dens_pow, L.veil = float(U(0.3, 0.5)), float(U(0.55, 0.75)), float(U(2.4, 3.2)), float(U(0.05, 0.1))
        L.shape_period = float(U(1.8, 2.6)); L.cov_period = float(U(3.5, 6.0) * z); L.cov_beta = 2.8
        L.cover = _timeline(rng, U(0.2, 0.42))
    elif kind == "rolls":
        L.thickness, L.sigma = float(U(0.45, 0.8)), float(U(18, 34))
        L.cellular, L.fill, L.fall = float(U(0.45, 0.65)), float(U(0.85, 1.0)), float(U(0.7, 0.95))
        L.edge, L.erosion, L.dens_pow, L.veil = float(U(0.3, 0.45)), float(U(0.4, 0.6)), float(U(2.0, 2.8)), float(U(0.08, 0.15))
        L.aniso = float(U(1.8, 3.0)); L.orient = float(h + rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.15))
        L.cov_period = float(U(4.0, 7.0) * z); L.cov_beta = 3.0
        L.cover = _timeline(rng, U(0.45, 0.7))
    return L


def _veil_layer(rng, prior):
    U = rng.uniform
    e = prior.get("evo_scale", 1.0)
    z = U(3.5, 7.0)
    L = Layer(name="veil", z_base=float(z), thickness=float(U(0.6, 1.2)), n_planes=8)
    L.wind, h = _wind(rng, prior, z)
    L.sigma = float(U(2.0, 5.5)); L.cellular = float(U(0.0, 0.2)); L.fill = float(U(0.95, 1.15)); L.fall = float(U(1.0, 1.4))
    L.edge = float(U(0.45, 0.7)); L.erosion = float(U(0.1, 0.3)); L.dens_pow = float(U(1.2, 1.6)); L.veil = float(U(0.35, 0.55))
    L.core_var = float(U(0.2, 0.6))
    L.aniso = float(U(1.2, 2.6)); L.orient = float(h + rng.normal(0, 0.3))
    L.shape_period = float(U(7, 13)); L.detail_period = float(U(1.8, 3.0)); L.shape_vperiod = float(U(2.5, 4.0))
    L.cov_period = float(U(8.0, 14.0) * z); L.cov_beta = 3.2
    L.evo_speed = float(U(0.002, 0.005) * e); L.boil = float(U(0.0003, 0.0007) * e)
    L.shear = (float(U(-0.003, 0.003) * e), float(U(-0.003, 0.003) * e))
    L.cover = _timeline(rng, U(0.6, 0.92))
    return L


def sample_scenario(seed, era="A", regime=None):
    rng = np.random.default_rng(seed)
    prior = load_prior()
    names = list(REGIMES)
    if regime is None:
        p = np.array([REGIMES[n][0] for n in names]); regime = str(rng.choice(names, p=p / p.sum()))
    if regime == "veil":
        layers = [_veil_layer(rng, prior)]
    elif regime == "two_layer":
        low = _low_layer(rng, prior, "broken"); low.n_planes = 12
        low.cover = _timeline(rng, rng.uniform(0.3, 0.55))
        layers = [low, _veil_layer(rng, prior)]
    elif regime == "front":
        L = _low_layer(rng, prior, "overcast")
        wx, wy = L.wind
        spd = max(np.hypot(wx, wy), 0.002) * rng.uniform(0.8, 1.6)
        ang = float(np.arctan2(wy, wx))
        t_cross = rng.uniform(900, 2700)                     # when the edge passes the zenith
        sgn = rng.choice([-1.0, 1.0])                        # arriving or departing deck
        L.cover = [(0.0, 0.5), (3600.0, 0.5)]
        L.front = (float(0.47 * sgn), ang, float(-spd * t_cross), float(spd), float(rng.uniform(2.0, 5.0)))
        layers = [L]
    else:
        layers = [_low_layer(rng, prior, regime)]
    return Scenario(seed=int(seed), regime=regime, layers=layers, lighting=era_lighting(era, rng))
