"""Equidistant fisheye geometry of the KHO all-sky camera (608x608 frames).

Pixel radius r from the optical centre maps linearly to zenith angle:
theta = r / R90 * 90deg.  Image azimuth psi is measured in the image plane
(atan2 of pixel offsets); we never need true compass azimuth, only a
consistent frame, so world x = image +x (right), world y = image +y (down).
"""
import numpy as np

CX, CY = 298.5, 295.0     # optical centre (fit to the compass tick ring)
R90 = 262.0               # pixel radius of the 90 deg horizon
R_EARTH = 6371.0          # km


def pixel_grid(h=608, w=608):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - CX, yy - CY
    r = np.hypot(dx, dy)
    theta = r / R90 * (np.pi / 2)
    psi = np.arctan2(dy, dx)
    return r, theta, psi


def slant_range(theta, z):
    """Distance along a ray at zenith angle theta to altitude z (km), curved Earth."""
    c = np.cos(theta)
    return -R_EARTH * c + np.sqrt((R_EARTH * c) ** 2 + 2 * R_EARTH * z + z * z)
