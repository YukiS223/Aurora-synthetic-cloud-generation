"""Volumetric cloud renderer for all-sky frames.

The cloud volume is sampled where every camera ray crosses a stack of
altitude planes.  On that (pixel x plane) lattice three things happen:

1. Density.  Each layer's density is a thresholded 3-D noise field gated by
   an evolving coverage map and a height profile, in world kilometres, advected
   by the wind.  Because it is a true volume seen in perspective, gaps close
   toward the horizon, cloud sides become visible, and layers show parallax.

2. Light propagation.  Optical depth is accumulated plane by plane along
   three families of light paths, each a cheap image warp between
   neighbouring planes: straight up (light from the sky: aurora, airglow),
   straight down (light bounced off the snow), and slanted toward each ground
   light.  This gives every sample its own illumination: bases lit from below,
   cores shadowed from above, bulges facing a town brighter than their lee.

3. View integration.  Front-to-back along the ray.  The clear frame is the
   background and also the light source: what scatters into the line of sight
   is the clear frame itself, blurred more and more widely the deeper the
   light had to travel through cloud.  Thin cloud therefore blurs and halos the
   aurora without dimming it much; thick cloud replaces it with a dim glow in
   the colour of the whole sky; ground lights add their own colour from below.

Everything is computed in approximately linear light and converted back.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import CX, CY, R90, slant_range
from . import textures
from .weather import cover_at

GAMMA = 2.2
X0, Y0, WC, HC = 34, 31, 528, 528          # crop that contains the whole lens circle
THETA_CAP = math.radians(89.3)
H_REF = 1.5                                 # km, reference cloud-base height for source intensities
SUB = 2                                     # density samples per slab along the ray


def _dev():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _pad_wrap(t):
    """Circularly pad the last two dims by one texel (for wrap-around bilinear sampling)."""
    t = torch.cat([t[..., -1:, :], t, t[..., :1, :]], dim=-2)
    return torch.cat([t[..., -1:], t, t[..., :1]], dim=-1)


def _wrap_grid(u, v, n):
    """Texture coords in tile units -> grid_sample coords on a wrap-padded n x n texture."""
    gx = (u - torch.floor(u)) * n + 1.0
    gy = (v - torch.floor(v)) * n + 1.0
    return torch.stack([2.0 * gx / (n + 1) - 1.0, 2.0 * gy / (n + 1) - 1.0], -1)


def _smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _gauss1d(sigma, dev):
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(img, sigma):
    """Separable Gaussian blur of [C,H,W], replicate borders."""
    k = _gauss1d(sigma, img.device)
    r = (len(k) - 1) // 2
    c = img.shape[0]
    x = img[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x[0]


class Renderer:
    def __init__(self, scene, scenario, device=None, verbose=False, sub=SUB):
        self.dev = dev = device or _dev()
        self.sc = scenario
        self.light = scenario.lighting
        self.verbose = verbose
        rng = np.random.default_rng(scenario.seed)

        # ---------------- static scene
        sl = (slice(Y0, Y0 + HC), slice(X0, X0 + WC))
        self.sl = sl
        self.sky = torch.from_numpy(scene["sky"][sl]).to(dev)
        self.hot = torch.from_numpy(scene["hot"][sl]).to(dev)
        self.noise_sigma = torch.tensor(scene["noise_sigma"], device=dev).view(3, 1, 1) / 255.0

        # ---------------- ray geometry
        yy, xx = np.mgrid[Y0:Y0 + HC, X0:X0 + WC].astype(np.float64)
        dx, dy = xx - CX, yy - CY
        r = np.hypot(dx, dy)
        theta = np.minimum(r / R90 * (np.pi / 2), THETA_CAP)
        psi = np.arctan2(dy, dx)
        cosp, sinp = np.cos(psi), np.sin(psi)
        self.inside = torch.from_numpy((r < R90 * 0.995).astype(np.float32)).to(dev)

        # ---------------- plane stack (all layers, sorted by altitude)
        planes = []
        for li, L in enumerate(scenario.layers):
            dz = L.thickness / L.n_planes
            for i in range(L.n_planes):
                planes.append((L.z_base + (i + 0.5) * dz, dz, li, (i + 0.5) / L.n_planes))
        planes.sort(key=lambda p: p[0])
        self.planes = planes
        K = len(planes)
        zs = np.array([p[0] for p in planes])
        self.dz = torch.tensor([p[1] for p in planes], device=dev, dtype=torch.float32).view(K, 1, 1)
        self.layer_of = [p[2] for p in planes]
        self.hrel = [p[3] for p in planes]

        th_tab = np.linspace(0, THETA_CAP, 4000)
        d_tab = [slant_range(th_tab, z) * np.sin(th_tab) for z in zs]
        eps = 1e-3
        X = np.empty((K, HC, WC), np.float32); Y = np.empty_like(X)
        sec = np.empty_like(X); foot = np.empty_like(X)
        # Density is supersampled along the ray inside every slab (SUB heights per
        # plane): a slanted ray crosses a slab over a horizontal distance of
        # dz*tan(theta), and one sample per slab would show up as comb-like banding.
        self.sub = (np.arange(sub) + 0.5) / sub - 0.5
        Xs = np.empty((len(self.sub), K, HC, WC), np.float32); Ys = np.empty_like(Xs)
        for k, z in enumerate(zs):
            for si, so in enumerate(self.sub):
                zz = z + so * planes[k][1]
                dd = np.interp(theta, th_tab, slant_range(th_tab, zz) * np.sin(th_tab))
                Xs[si, k], Ys[si, k] = dd * cosp, dd * sinp
            d = np.interp(theta, th_tab, d_tab[k])
            X[k], Y[k] = d * cosp, d * sinp
            sec[k] = (slant_range(theta, z + eps) - slant_range(theta, z - eps)) / (2 * eps)
            foot[k] = np.interp(theta, th_tab, np.gradient(d_tab[k], th_tab)) * (np.pi / 2 / R90)   # km per pixel, radial
        self.Xs = torch.from_numpy(Xs).to(dev); self.Ys = torch.from_numpy(Ys).to(dev)
        self.sec = torch.from_numpy(np.minimum(sec, 40.0)).to(dev)
        self.foot = torch.from_numpy(foot).to(dev)

        def to_grid(theta2, psi2):
            r2 = theta2 / (np.pi / 2) * R90
            gx = 2.0 * (CX + r2 * np.cos(psi2) - X0 + 0.5) / WC - 1.0
            gy = 2.0 * (CY + r2 * np.sin(psi2) - Y0 + 0.5) / HC - 1.0
            # MPS has no border padding: clamp to the outermost pixel centres instead
            lim = 1.0 - 1.0 / WC
            return np.clip(np.stack([gx, gy], -1), -lim, lim).astype(np.float32)

        # warp grids between neighbouring planes: same world (x, y) one plane up / down
        up = np.zeros((K, HC, WC, 2), np.float32); dn = np.zeros_like(up)
        for k in range(K):
            dk = np.hypot(X[k], Y[k]).astype(np.float64)
            if k + 1 < K:
                up[k] = to_grid(np.interp(dk, d_tab[k + 1], th_tab), psi)
            if k > 0:
                dn[k] = to_grid(np.interp(dk, d_tab[k - 1], th_tab), psi)
        self.g_up = torch.from_numpy(up).to(dev); self.g_dn = torch.from_numpy(dn).to(dev)

        # slanted paths toward each ground source, and its irradiance on every sample
        self.g_src, self.sec_src, self.E_src, self.src_color = [], [], [], []
        for s in self.light.sources:
            g = np.zeros((K, HC, WC, 2), np.float32); ss = np.zeros((K, HC, WC), np.float32)
            E = np.zeros((K, HC, WC), np.float32)
            for k, z in enumerate(zs):
                px, py = X[k].astype(np.float64), Y[k].astype(np.float64)
                dist2 = (px - s.x) ** 2 + (py - s.y) ** 2 + z * z
                E[k] = H_REF ** 2 * z / dist2 ** 1.5
                ss[k] = np.minimum(np.sqrt(dist2) / z, 12.0)
                if k > 0:
                    f = 1.0 - zs[k - 1] / z
                    qx, qy = px + (s.x - px) * f, py + (s.y - py) * f
                    g[k] = to_grid(np.interp(np.hypot(qx, qy), d_tab[k - 1], th_tab), np.arctan2(qy, qx))
            self.g_src.append(torch.from_numpy(g).to(dev)); self.sec_src.append(torch.from_numpy(ss).to(dev))
            self.E_src.append(torch.from_numpy(E).to(dev))
            self.src_color.append(torch.tensor(s.color, device=dev, dtype=torch.float32).view(3, 1, 1))

        # ---------------- noise textures and per-layer level-of-detail weights
        self.tex = []
        for li, L in enumerate(scenario.layers):
            lr = np.random.default_rng(rng.integers(1 << 31))
            cov = textures.coverage_texture(lr, n=512, beta=L.cov_beta, warp=0.03,
                                            kmax=max(4.0, L.cov_period / (2.5 * L.thickness)))
            shp = textures.shape_texture(lr, n=128, nz=32, cellular=L.cellular)
            det = textures.detail_texture(lr, n=64, nz=64)
            ks = [k for k in range(K) if self.layer_of[k] == li]
            fk = self.foot[ks]
            tx_s, tx_d, tx_c = L.shape_period / 128, L.detail_period / 64, L.cov_period / 512
            ws0 = 1 - _smoothstep(1.0, 3.5, fk / tx_s)
            ws1 = _smoothstep(1.0, 3.5, fk / tx_s) * (1 - _smoothstep(8.0, 28.0, fk / tx_s))
            wd = 1 - _smoothstep(0.8, 3.0, fk / tx_d)
            wc = 1 - _smoothstep(4.0, 16.0, fk / tx_c)
            self.tex.append(dict(
                cov=_pad_wrap(torch.from_numpy(cov).to(dev))[None],          # [1,2,514,514]
                shp=torch.from_numpy(shp).to(dev), det=torch.from_numpy(det).to(dev),
                ks=ks, ws0=ws0, ws1=ws1, wd=wd, wc=wc,
                off=torch.tensor(lr.random(8), device=dev, dtype=torch.float32)))
        self.K = K

    # ------------------------------------------------------------------ density
    def _slices(self, vol, zc):
        """Linear interpolation between z-slices of a periodic volume. zc: [n] in tiles."""
        nz = vol.shape[0]
        f = (zc - torch.floor(zc)) * nz
        i0 = torch.floor(f).long() % nz
        i1 = (i0 + 1) % nz
        w = (f - torch.floor(f)).view(-1, 1, 1)
        return vol[i0] * (1 - w) + vol[i1] * w            # [n, N, N]

    def density(self, t):
        D = torch.zeros((self.K, HC, WC), device=self.dev)
        for sub_i, so in enumerate(self.sub):
            D += self._density_at(t, sub_i, float(so))
        return D * (self.inside / len(self.sub))

    def _density_at(self, t, sub_i, so):
        D = torch.zeros((self.K, HC, WC), device=self.dev)
        for li, L in enumerate(self.sc.layers):
            T = self.tex[li]; ks = T["ks"]; n = len(ks); off = T["off"]
            zrel = torch.tensor([self.hrel[k] + so / L.n_planes for k in ks], device=self.dev).view(n, 1, 1)
            zabs = torch.tensor([self.planes[k][0] + so * self.planes[k][1] for k in ks], device=self.dev)
            # wind-advected, rotated, stretched coordinates (km)
            x = self.Xs[sub_i, ks] - L.wind[0] * t
            y = self.Ys[sub_i, ks] - L.wind[1] * t
            co, si = math.cos(L.orient), math.sin(L.orient)
            xr = (x * co + y * si) / L.aniso
            yr = -x * si + y * co
            # coverage: two fields drifting through each other, recombined to a uniform marginal
            e = L.evo_speed * t
            ua = torch.stack([xr / L.cov_period + off[0] + e / L.cov_period,
                              xr / L.cov_period + off[2] - 0.6 * e / L.cov_period], 0)
            va = torch.stack([yr / L.cov_period + off[1] + 0.4 * e / L.cov_period,
                              yr / L.cov_period + off[3] + 0.8 * e / L.cov_period], 0)
            cov = T["cov"]
            ca = F.grid_sample(cov[:, 0:1], _wrap_grid(ua[0], va[0], 512).reshape(1, n * HC, WC, 2),
                               mode="bilinear", padding_mode="zeros", align_corners=True).view(n, HC, WC)
            cb = F.grid_sample(cov[:, 1:2], _wrap_grid(ua[1], va[1], 512).reshape(1, n * HC, WC, 2),
                               mode="bilinear", padding_mode="zeros", align_corners=True).view(n, HC, WC)
            s = 0.5 * (ca + cb)
            c = torch.where(s < 0.5, 2 * s * s, 1 - 2 * (1 - s) ** 2)
            c = T["wc"] * c + (1 - T["wc"]) * 0.5
            cover = cover_at(L, t)
            if L.front is not None:
                amp, ang, pos0, spd, wid = L.front
                cover = cover + amp * torch.tanh((x * math.cos(ang) + y * math.sin(ang) - (pos0 + spd * t)) / wid)
            # the nominal cover boundary sits mid-ramp, so realised cover tracks the target
            w = ((c - (1.0 - cover)) / L.edge + 0.5).clamp(0, 1)
            top = w ** L.thick_pow                                # relative height of the local cloud top
            prof = _smoothstep(0.0, 0.14, zrel) * (1 - _smoothstep(0.7 * top, top + 1e-4, zrel))
            avail = _smoothstep(0.0, 1.0, w) * prof
            # 3-D shape noise at this altitude (slowly boiling), two detail levels + mean
            zc = zabs / L.shape_vperiod + L.boil * t + off[4]
            sl0 = self._slices(T["shp"], zc)                      # [n,128,128]
            sl1 = F.avg_pool2d(sl0[:, None], 8)                   # [n,1,16,16]
            us, vs = xr / L.shape_period + off[5], yr / L.shape_period + off[6]
            s0 = F.grid_sample(_pad_wrap(sl0[:, None]), _wrap_grid(us, vs, 128), mode="bilinear",
                               padding_mode="zeros", align_corners=True)[:, 0]
            s1 = F.grid_sample(_pad_wrap(sl1), _wrap_grid(us, vs, 16), mode="bilinear",
                               padding_mode="zeros", align_corners=True)[:, 0]
            shp = T["ws0"] * s0 + T["ws1"] * s1 + (1 - T["ws0"] - T["ws1"]) * 0.5
            base = ((shp - (1.0 - avail * L.fill)) / L.fall).clamp(0, 1)
            # wispy erosion of the soft parts
            zd = zabs / L.detail_period + 1.7 * L.boil * t + off[7]
            dl = self._slices(T["det"], zd)
            ud = (xr - L.shear[0] * t) / L.detail_period + off[6]
            vd = (yr - L.shear[1] * t) / L.detail_period + off[5]
            d0 = F.grid_sample(_pad_wrap(dl[:, None]), _wrap_grid(ud, vd, 64), mode="bilinear",
                               padding_mode="zeros", align_corners=True)[:, 0]
            det = T["wd"] * d0 + (1 - T["wd"]) * 0.5
            base = (base - L.erosion * (1 - det) * (1 - 0.75 * base)).clamp(0, 1) / (1 - 0.25 * L.erosion)
            base = base.clamp(0, 1)
            # mid-scale variation of how dense the cloud is: thick dark cores next to thin bright sheets
            um, vm = xr / (2.3 * L.shape_period) + off[1], yr / (2.3 * L.shape_period) + off[0] + 0.3 * e / L.cov_period
            mod = F.grid_sample(cov[:, 1:2], _wrap_grid(um * 0.25, vm * 0.25, 512).reshape(1, n * HC, WC, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=True).view(n, HC, WC)
            mod = T["wc"] * mod + (1 - T["wc"]) * 0.5
            sig = L.sigma * torch.exp(L.core_var * 2.4 * (mod - 0.5))
            # dense cores (power law) wrapped in a thin haze fringe (linear term)
            D[ks] = sig * (L.veil * base + (1 - L.veil) * base ** L.dens_pow)
        return D

    # ------------------------------------------------------------------ light
    def _warp(self, img, grid):
        return F.grid_sample(img[None, None], grid[None], mode="bilinear", padding_mode="zeros",
                             align_corners=False)[0, 0]

    def transport(self, D):
        """Returns view transmittance and the scalar in-scatter weight maps."""
        K, lt = self.K, self.light
        tv_dz = D * self.dz                                       # vertical optical depth per plane
        U = torch.zeros_like(D); B = torch.zeros_like(D)
        U[K - 1] = tv_dz[K - 1]
        for k in range(K - 2, -1, -1):
            U[k] = tv_dz[k] + self._warp(U[k + 1], self.g_up[k])
        B[0] = tv_dz[0]
        for k in range(1, K):
            B[k] = tv_dz[k] + self._warp(B[k - 1], self.g_dn[k])
        tau_a = U - 0.5 * tv_dz
        tau_b = B - 0.5 * tv_dz
        tau_col = tau_a + tau_b
        G = []
        for si in range(len(lt.sources)):
            g = torch.zeros_like(D)
            inc = tv_dz * self.sec_src[si]
            g[0] = inc[0]
            for k in range(1, K):
                g[k] = inc[k] + self._warp(g[k - 1], self.g_src[si][k])
            G.append(g - 0.5 * inc)

        kap = lt.kappa
        refl = 0.2 + 0.8 * (kap * tau_col) / (1 + kap * tau_col)   # diffuse reflectance of the local column
        Tt = 1.0 / (1.0 + kap * tau_a)                              # sky light surviving down to the sample
        m = 1.0 - torch.exp(-tau_a / lt.tau_iso)                    # how diffuse it has become
        delta = tv_dz * self.sec
        a = 1.0 - torch.exp(-delta)
        Tv = torch.ones((HC, WC), device=self.dev)
        Wsingle = torch.zeros_like(Tv); Wdiff = torch.zeros_like(Tv); Wg = torch.zeros_like(Tv)
        Ws = [torch.zeros_like(Tv) for _ in lt.sources]
        for k in range(K):
            c = Tv * a[k]
            Wsingle += c * Tt[k] * (1 - m[k])
            Wdiff += c * Tt[k] * m[k]
            Wg += c * refl[k] / (1 + kap * tau_b[k])
            for si, s in enumerate(lt.sources):
                Ws[si] += c * self.E_src[si][k] * refl[k] / (1 + s.kappa * G[si][k])
            Tv = Tv * torch.exp(-delta[k])
        # mean sky-light transmission of the deck onto the ground (for the snow bounce)
        zen = (self.foot[0] < 4 * self.foot[0].min()) & (self.inside > 0)
        t_deck = (1.0 / (1.0 + kap * U[0]))[zen].mean()
        return dict(Tv=Tv, Wsingle=Wsingle, Wdiff=Wdiff, Wg=Wg, Ws=Ws, t_deck=t_deck, tau_col=U[0])

    # ------------------------------------------------------------------ sky as a light source
    def sky_light(self, lin):
        """Blurred versions of the clear frame: what a cloud scatters toward the camera."""
        # stars and hot pixels carry negligible flux: grey opening removes them
        x = -F.max_pool2d(-lin[None], 3, 1, 1)
        x = F.max_pool2d(x, 3, 1, 1)[0]
        m = self.sky[None]
        # Clipped aurora reads white but is not: give saturated pixels the hue of
        # their unsaturated surroundings and some of the brightness that was cut off.
        mx = x.max(0, keepdim=True)[0]
        sat = _smoothstep(0.72, 0.98, mx)
        w = (1 - sat) * m * mx
        hue = _blur(F.avg_pool2d((x * w)[None], 4)[0], 4.0) / _blur(F.avg_pool2d(w[None], 4)[0], 4.0).clamp_min(1e-4)
        hue = F.interpolate(hue[None], size=(HC, WC), mode="bilinear", align_corners=False)[0]
        hue = hue / hue.max(0, keepdim=True)[0].clamp_min(1e-4)
        x = x * (1 - sat) + hue * mx * (1.0 + 1.5 * sat) * sat
        num, den = x * m, m

        def nblur(sig, down):
            n_, d_ = num, den
            if down > 1:
                n_, d_ = F.avg_pool2d(n_[None], down)[0], F.avg_pool2d(d_[None], down)[0]
            n_, d_ = _blur(n_, sig / down), _blur(d_, sig / down)
            o = n_ / d_.clamp_min(1e-3)
            if down > 1:
                o = F.interpolate(o[None], size=(HC, WC), mode="bilinear", align_corners=False)[0]
            return o

        l0, l1, l2 = nblur(4.0, 1), nblur(12.0, 4), nblur(36.0, 8)
        hemi = (num.sum((1, 2)) / den.sum()).view(3, 1, 1).expand(3, HC, WC)
        # single-scatter lobe: sharp forward peak plus a broad tail; diffuse: nearly isotropic
        single = 0.46 * l0 + 0.24 * l1 + 0.16 * l2 + 0.14 * hemi
        diffuse = 0.25 * l2 + 0.75 * hemi
        return single, diffuse, hemi

    # ------------------------------------------------------------------ frame
    @torch.no_grad()
    def render(self, frame, fi, return_aux=False):
        """frame: HxWx3 uint8 (full 608x608). Returns the clouded frame (uint8)."""
        dev, lt = self.dev, self.light
        t = fi * self.sc.frame_dt
        crop = torch.from_numpy(np.ascontiguousarray(frame[self.sl])).to(dev).permute(2, 0, 1).float() / 255.0
        lin = crop ** GAMMA
        single, diffuse, hemi = self.sky_light(lin)
        tr = self.transport(self.density(t))
        Tv = tr["Tv"][None]
        amb = torch.tensor(lt.ambient, device=dev).view(3, 1, 1)
        ground = lt.snow_albedo * tr["t_deck"] * hemi + amb
        out = lin * Tv + tr["Wsingle"][None] * single + tr["Wdiff"][None] * diffuse + tr["Wg"][None] * ground
        for si in range(len(lt.sources)):
            out = out + tr["Ws"][si][None] * self.src_color[si]
        out = out.clamp(0, 1) ** (1.0 / GAMMA)
        # sensor grain: the clear frame's grain is attenuated under cloud, so add back what was lost
        g = torch.Generator(device="cpu").manual_seed(self.sc.seed * 100003 + fi)
        nz = torch.randn((3, HC, WC), generator=g).to(dev)
        nz = _blur(nz, 0.55) * 1.9
        out = out + nz * self.noise_sigma * torch.sqrt((1 - Tv * Tv).clamp(0, 1))
        M = self.sky[None]
        out = crop * (1 - M) + out.clamp(0, 1) * M
        # sensor defects sit on top of whatever the sky does: re-add their excess over the local background
        opened = F.max_pool2d(-F.max_pool2d(-crop[None], 5, 1, 2), 5, 1, 2)[0]
        excess = (lin - opened ** GAMMA).clamp_min(0)
        out = torch.where(self.hot[None], ((out ** GAMMA + excess).clamp(0, 1)) ** (1 / GAMMA), out)
        res = frame.copy()
        res[self.sl] = (out.permute(1, 2, 0) * 255.0 + 0.5).clamp(0, 255).byte().cpu().numpy()
        if return_aux:
            op = ((1 - tr["Tv"]) * self.sky).clamp(0, 1)
            full = np.zeros(frame.shape[:2], np.uint8)
            full[self.sl] = (op * 255 + 0.5).byte().cpu().numpy()
            return res, dict(opacity=full, t_deck=float(tr["t_deck"]),
                             cover=float(((tr["tau_col"] > 0.3).float() * self.sky).sum() / self.sky.sum()))
        return res
