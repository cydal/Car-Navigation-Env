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


class Radar:
    """Ego-centric, per-sector nearest-moving-vehicle range + closing speed.

    Vehicles only -- it never sees building/city geometry, which is LIDAR's
    job. Sector 0 is centred straight ahead (0 rad in the car's body frame)
    and sectors run clockwise, matching the convention a HUD radar panel wants
    ("front" is sector 0), which is why it differs from `Lidar`'s beam 0
    (centred behind, so the array's wrap-around sits where it carries the
    least information for a 1D conv).

    This is an auxiliary sensor for display/telemetry: it is never mixed into
    `CarNavEnv`'s observation vector, so enabling or reconfiguring it cannot
    change `vector_dim`/`obs_slices` for any existing consumer.
    """

    def __init__(self, n_sectors=8, max_range=60.0, noise_std=0.0, rng=None):
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.noise_std = noise_std      # metres of Gaussian range noise
        self.rng = rng or np.random.default_rng()
        self.last_distances = np.full(n_sectors, max_range, dtype=np.float32)
        self.last_closing_speed = np.zeros(n_sectors, dtype=np.float32)
        # Index into the traffic arrays of the tracked vehicle, or -1 if none.
        self.last_target_idx = np.full(n_sectors, -1, dtype=np.int32)

    @property
    def forward_index(self):
        return 0

    def sector_bearing(self, i):
        """Centre bearing of sector `i`, radians, ego frame (0 = dead ahead)."""
        return 2.0 * np.pi * i / self.n_sectors

    def scan(self, traffic, car):
        """Update `last_distances`/`last_closing_speed`/`last_target_idx`.

        Only considers `traffic`'s moving vehicles (`traffic.n_moving`); a
        stationary parked car has no closing speed to report and radar's
        whole value here is the relative-velocity read that LIDAR can't give.
        """
        n = self.n_sectors
        self.last_distances = np.full(n, self.max_range, dtype=np.float32)
        self.last_closing_speed = np.zeros(n, dtype=np.float32)
        self.last_target_idx = np.full(n, -1, dtype=np.int32)

        m = int(getattr(traffic, "n_moving", 0))
        if m == 0:
            return

        dx = traffic.x[:m] - car.x
        dy = traffic.y[:m] - car.y
        d = np.hypot(dx, dy)
        in_range = d <= self.max_range
        if not np.any(in_range):
            return

        width = 2.0 * np.pi / n
        bearing = (np.arctan2(dy, dx) - car.heading + np.pi) % (2.0 * np.pi) - np.pi
        sector = np.floor((bearing + width / 2.0) / width).astype(int) % n

        # Ego/vehicle world-frame velocities, projected onto the ego->vehicle
        # unit vector: positive `closing` means the vehicle is approaching,
        # matching the automotive-radar sign convention.
        c, s = np.cos(car.heading), np.sin(car.heading)
        vx = traffic.speed[:m] * np.cos(traffic.heading[:m]) - car.speed * c
        vy = traffic.speed[:m] * np.sin(traffic.heading[:m]) - car.speed * s
        safe_d = np.where(d > 1e-6, d, 1.0)
        ux, uy = dx / safe_d, dy / safe_d
        closing = -(vx * ux + vy * uy)

        for i in np.where(in_range)[0]:
            k = int(sector[i])
            if d[i] < self.last_distances[k]:
                dist = float(d[i])
                if self.noise_std > 0.0:
                    dist = float(np.clip(dist + self.rng.normal(0.0, self.noise_std),
                                         0.0, self.max_range))
                self.last_distances[k] = dist
                self.last_closing_speed[k] = float(closing[i])
                self.last_target_idx[k] = int(i)


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
