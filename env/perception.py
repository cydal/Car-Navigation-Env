"""
Structured, occlusion-aware camera + blind-spot sensing.

Auxiliary sensor, same discipline as `Lidar`/`Radar` (sensors.py): computed on
every `reset()`/`step()`, exposed as `env.perception`, never concatenated into
the observation vector. This is for the live viewer's HUD today -- a "what
does the car's camera actually see" readout, not a training signal -- but
nothing stops a future observation block reading it the same way `Radar`'s
data could be, if a structured-camera-input experiment ever wants it.

`CAMERAS` is the canonical camera rig (mount offset in the car's body frame,
field of view, range) for front/left/right/rear. The live viewer's 3D scene
(`web/scene.mjs`) mounts its rendering cameras at these exact same numbers,
sent once over the wire in the server's reset message, rather than keeping a
second hand-copied constant in JavaScript that could quietly drift from this
one.

Occlusion is checked against other tracked *vehicles* only, not buildings: a
target hidden behind a building corner is almost always already out of range
or field of view anyway, and checking every detection against the tile grid
would add a per-detection raycast on top of what LIDAR already spends there
(see `Lidar`'s own docstring on why that raycasting is the thing to be
careful about) for a difference that's rarely visible.
"""

import numpy as np

# name -> (forward_m, right_m, yaw_offset_rad, fov_deg, range_m), all relative
# to the car's own body frame (forward along heading, right perpendicular to
# it). These numbers are shared verbatim with the renderer -- see the module
# docstring.
CAMERAS = {
    "front": (1.8, 0.0, 0.0, 70.0, 200.0),
    "left": (0.3, -0.9, -np.pi / 2.0, 120.0, 30.0),
    "right": (0.3, 0.9, np.pi / 2.0, 120.0, 30.0),
    "rear": (-2.1, 0.0, np.pi, 100.0, 70.0),
}

# A blind-spot zone is a rectangle beside the car, longer behind than ahead
# (mirroring a real wing mirror's blind spot), offset by roughly one lane's
# width. Body frame: `forward` along heading, `right` perpendicular to it.
BLIND_BEHIND_M = 7.0
BLIND_AHEAD_M = 1.5
BLIND_LATERAL_M = 3.5
BLIND_HALF_WIDTH_M = 1.65

MAX_DETECTIONS_PER_CAMERA = 6


def _body_frame(car, world_x, world_y):
    """World (x, y) arrays -> (forward, right) in the car's body frame."""
    fx, fy = np.cos(car.heading), np.sin(car.heading)   # forward unit vector
    rx, ry = -fy, fx                                     # right unit vector
    dx, dy = world_x - car.x, world_y - car.y
    return dx * fx + dy * fy, dx * rx + dy * ry


def _mount(car, forward_m, right_m, yaw_offset):
    """Camera mount position (world x, y) and absolute yaw, from body-frame offsets."""
    fx, fy = np.cos(car.heading), np.sin(car.heading)
    rx, ry = -fy, fx
    mx = car.x + forward_m * fx + right_m * rx
    my = car.y + forward_m * fy + right_m * ry
    return mx, my, car.heading + yaw_offset


def _segment_blocked(ox, oy, tx, ty, cx, cy, half_len, half_wid, heading):
    """True if the segment (ox,oy)-(tx,ty) passes through vehicle `c`'s
    oriented footprint (an axis-aligned box in its own body frame)."""
    c, s = np.cos(-heading), np.sin(-heading)

    def to_local(x, y):
        dx, dy = x - cx, y - cy
        return dx * c - dy * s, dx * s + dy * c

    lox, loy = to_local(ox, oy)
    ltx, lty = to_local(tx, ty)
    dx, dy = ltx - lox, lty - loy
    t0, t1 = 0.0, 1.0
    for lo, hi, o, d in ((-half_len, half_len, lox, dx), (-half_wid, half_wid, loy, dy)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return False
            continue
        ta, tb = (lo - o) / d, (hi - o) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    # Ends just short of the target itself and just past the observer, so a
    # vehicle never occludes itself and the ray's own endpoint doesn't count.
    return 0.002 < t1 and t0 < 0.998


class Perception:
    """Per-camera detections plus left/right blind-spot occupancy.

    `cameras[name]` is a list of `(vehicle_index, distance_m, bearing_rad)`
    tuples, nearest first, capped at `MAX_DETECTIONS_PER_CAMERA`. `bearing_rad`
    is relative to that camera's own facing direction (0 = dead centre).
    `blind_spots[side]` is `{"occupied": bool, "vehicle": index or -1}`.
    """

    def __init__(self):
        self.cameras = {name: [] for name in CAMERAS}
        self.blind_spots = {"left": {"occupied": False, "vehicle": -1},
                             "right": {"occupied": False, "vehicle": -1}}

    def scan(self, traffic, car):
        n = int(getattr(traffic, "n_moving", 0))
        self.cameras = {name: [] for name in CAMERAS}
        if n > 0:
            xs, ys = traffic.x[:n], traffic.y[:n]
            lengths, widths, headings = traffic.length[:n], traffic.width[:n], traffic.heading[:n]
            for name, (fwd, right, yaw, fov_deg, rng) in CAMERAS.items():
                self.cameras[name] = self._scan_camera(
                    car, xs, ys, lengths, widths, headings, fwd, right, yaw, fov_deg, rng)
        self._scan_blind_spots(traffic, car, n)

    def _scan_camera(self, car, xs, ys, lengths, widths, headings, fwd, right, yaw, fov_deg, rng):
        ox, oy, oyaw = _mount(car, fwd, right, yaw)
        half_fov = np.radians(fov_deg) / 2.0

        dx, dy = xs - ox, ys - oy
        dist = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - oyaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))   # wrap to [-pi, pi]
        candidates = np.nonzero((dist <= rng) & (np.abs(bearing) <= half_fov))[0]
        if candidates.size == 0:
            return []

        n = len(xs)
        detections = []
        for i in candidates:
            blocked = False
            for j in range(n):
                if j == i:
                    continue
                if _segment_blocked(ox, oy, xs[i], ys[i], xs[j], ys[j],
                                    lengths[j] / 2.0, widths[j] / 2.0, headings[j]):
                    blocked = True
                    break
            if not blocked:
                detections.append((int(i), float(dist[i]), float(bearing[i])))
        detections.sort(key=lambda t: t[1])
        return detections[:MAX_DETECTIONS_PER_CAMERA]

    def _scan_blind_spots(self, traffic, car, n):
        for side in ("left", "right"):
            self.blind_spots[side] = {"occupied": False, "vehicle": -1}
        if n == 0:
            return
        fwd, right = _body_frame(car, traffic.x[:n], traffic.y[:n])
        in_span = (fwd >= -BLIND_BEHIND_M) & (fwd <= BLIND_AHEAD_M)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            centre = sign * BLIND_LATERAL_M
            in_lane = np.abs(right - centre) <= BLIND_HALF_WIDTH_M
            hit = np.nonzero(in_span & in_lane)[0]
            if hit.size:
                # Nearest along the lane, not just the first index, in case more
                # than one vehicle sits in the zone at once.
                nearest = hit[np.argmin(np.abs(fwd[hit]))]
                self.blind_spots[side] = {"occupied": True, "vehicle": int(nearest)}
