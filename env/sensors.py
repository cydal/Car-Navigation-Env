"""
Ego-centric sensing.

Everything here is expressed in the car's body frame, never world coordinates.
That makes the observation invariant to where on the map the car happens to be,
so a policy learned on one procedural layout transfers to the next instead of
memorising absolute positions.
"""

import numpy as np


class Lidar:
    """360-degree ray-cast range finder returning normalised distances.

    Beam layout: beam 0 points directly behind the car and beams sweep clockwise
    to beam n_beams-1. Straight ahead therefore lands at index n_beams // 2,
    which puts the array's wrap-around discontinuity at the rear where it
    carries the least information -- convenient if you later run a 1D conv over
    the scan.
    """

    def __init__(self, n_beams=32, max_range=60.0, noise_std=0.0, dropout=0.0, rng=None,
                 exact=False, step=0.25):
        self.n_beams = n_beams
        self.max_range = max_range
        self.noise_std = noise_std      # metres of Gaussian range noise
        self.dropout = dropout          # per-beam chance of returning max_range
        # The sampled caster is the default: it agrees with the exact DDA to
        # within step/2 (verified in tests/test_core.py) and is ~2.7x faster,
        # which matters because ray casting dominates env step time. Set
        # exact=True to fall back to the DDA as an oracle.
        self.exact = exact
        self.step = step                # sampling interval when exact=False
        self.rng = rng or np.random.default_rng()
        # -pi .. +pi so that index n/2 == 0 rad == straight ahead
        self.offsets = -np.pi + 2.0 * np.pi * np.arange(n_beams) / n_beams
        self.last_distances = np.full(n_beams, max_range)

    @property
    def forward_index(self):
        return self.n_beams // 2

    def scan(self, city, x, y, heading, obstacles=None):
        """Return normalised ranges in [0, 1]; 1.0 means nothing within range.

        `obstacles` is an optional (M, 3) array of (x, y, radius) circles -- used
        later for traffic and parked cars, which are not part of the tile grid.
        """
        angles = heading + self.offsets
        if self.exact:
            dist = city.cast_rays(x, y, angles, self.max_range)
        else:
            dist = city.cast_rays_fast(x, y, angles, self.max_range, self.step)

        if obstacles is not None and len(obstacles) > 0:
            dist = np.minimum(dist, self._ray_circle_ranges(x, y, angles, obstacles))

        if self.noise_std > 0.0:
            dist = dist + self.rng.normal(0.0, self.noise_std, size=self.n_beams)
        if self.dropout > 0.0:
            miss = self.rng.random(self.n_beams) < self.dropout
            dist = np.where(miss, self.max_range, dist)

        dist = np.clip(dist, 0.0, self.max_range)
        self.last_distances = dist
        return (dist / self.max_range).astype(np.float32)

    def _ray_circle_ranges(self, x, y, angles, obstacles):
        """Closest ray-circle intersection per beam, vectorised over both.

        Solves |o + t*d - c|^2 = r^2 for every (beam, circle) pair at once.
        """
        obstacles = np.asarray(obstacles, dtype=np.float64)
        dx = np.cos(angles)[:, None]                     # (B, 1)
        dy = np.sin(angles)[:, None]
        cx = obstacles[None, :, 0] - x                   # (1, M)
        cy = obstacles[None, :, 1] - y
        r = obstacles[None, :, 2]

        # Ray directions are unit vectors, so the quadratic's a-term is 1.
        b = dx * cx + dy * cy                            # projection onto ray
        c = cx * cx + cy * cy - r * r
        disc = b * b - c
        valid = (disc >= 0.0) & (b > 0.0)                # ahead of the origin only
        t = np.where(valid, b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
        t = np.where(t >= 0.0, t, np.inf)
        return np.min(t, axis=1)

    def hit_points(self, x, y, heading):
        """World-space ray endpoints, for drawing the scan."""
        angles = heading + self.offsets
        return (x + np.cos(angles) * self.last_distances,
                y + np.sin(angles) * self.last_distances)


def relative_bearing(x, y, heading, tx, ty):
    """Distance and body-frame bearing from a pose to a world point.

    Returns (distance, sin(bearing), cos(bearing)). Angles are encoded as a
    sin/cos pair rather than a scalar so the network never sees the
    discontinuity at +/-pi.
    """
    dx, dy = tx - x, ty - y
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx) - heading
    return dist, np.sin(bearing), np.cos(bearing)


def nav_features(car, targets, start_idx, n_lookahead, nav_range):
    """Encode the next `n_lookahead` waypoints as 3 numbers each.

    Absent waypoints (sequence exhausted) are padded with sin=cos=0, which is
    not a valid unit vector and so is unambiguously distinguishable from any
    real bearing.
    """
    feats = []
    for k in range(n_lookahead):
        i = start_idx + k
        if i < len(targets):
            tx, ty = targets[i]
            d, s, c = relative_bearing(car.x, car.y, car.heading, tx, ty)
            feats += [min(d / nav_range, 1.0), s, c]
        else:
            feats += [1.0, 0.0, 0.0]
    return np.asarray(feats, dtype=np.float32)


def dynamics_features(car):
    """Proprioception: five ego-centric numbers, no world coordinates.

    steer_angle is included because it is hidden actuator state -- the wheels lag
    the command, and without it the observation is not Markov.
    """
    p = car.p
    return np.asarray([
        car.speed / p.max_speed,
        np.clip(car.yaw_rate / 2.0, -1.0, 1.0),
        car.steer_angle / p.max_steer,
        np.clip(car.accel / p.max_brake, -1.0, 1.0),
        np.clip(car.slip / p.max_steer, -1.0, 1.0),
    ], dtype=np.float32)
