"""
Zebra crossings with pedestrians, and the 30 km/h zones and signs they anchor.

Both are opt-in (`EnvConfig(pedestrians=True)`, `EnvConfig(speed_signs=True)`)
and both append their own observation block behind the existing ones, so
switching them on never moves a byte of the 73-D layout consumers already
depend on -- same discipline as the traffic block when it was added.

Why crossings live on junction arms rather than mid-block: the road graph
already knows every junction and the arms leaving it (`city.road_nodes`,
`city.node_links`), and a zebra just outside the junction box is where a real
one goes. Signalised junctions are skipped so a pedestrian never steps out
under a green light for the car -- that would make yielding and obeying the
signal contradict each other, which is unlearnable.

Pedestrians are kerb-triggered, as in the reference this env borrows the
idea from: a group waits at one kerb and starts across (after a per-person
delay) once the ego approaches within its trigger distance. They then wait
on the far kerb and re-arm once the ego has gone, so a crossing met twice in
one episode still does something the second time. Background traffic yields
to an occupied crossing (traffic.py); the ego has to work that out for
itself, which is the whole point.

Speed zones are anchored to crossings: 30 km/h within `zone_len` metres of
one along its corridor, `speed_limit_kmh` everywhere else, with a "30" sign
at the zone entry and a "50" sign at the exit for each direction of travel.
The env exposes the *active* limit and the next sign ahead directly, so the
vector stays Markov -- no agent-side sign memory is needed, unlike the
reference, whose LLM driver had to be told what it had passed.
"""

import numpy as np

from .traffic import DIRS, _right

PED_FEATURES = 5      # per pedestrian in the observation: same layout as a traffic slot
SIGN_FEATURES = 3     # active limit, next sign distance, next sign limit
PED_RADIUS = 0.3      # collision / LIDAR circle, metres
KERB_INSET = 0.4      # how far inside the corridor edge people stand
ZEBRA_HALF_LEN = 1.5  # zebra spans +/- this along the corridor
REARM_MARGIN = 15.0   # ego must get this far beyond trigger range before a group re-arms


class Street:
    """Crossings + pedestrians + speed zones for one episode. Flat arrays, like Traffic."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.time = 0.0
        self._clear()

    # ------------------------------------------------------------------
    def _clear(self):
        z = np.zeros(0)
        self.cx, self.cy = z, z.copy()                  # crossing centres (corridor centre line)
        self.cdir = np.zeros(0, dtype=np.int64)         # corridor axis, index into DIRS
        # 30-zone extent per crossing, along +a (beyond it) and -a (back through the
        # junction). Asymmetric because each side is clamped to its own arm's length.
        self.zone_f = z.copy()
        self.zone_b = z.copy()
        self.half = 6.0
        # pedestrians
        self.px, self.py, self.pvx, self.pvy = z, z.copy(), z.copy(), z.copy()
        self.pc = np.zeros(0, dtype=np.int64)           # crossing index
        self.side = z.copy()                            # +1 / -1: kerb currently waited on
        self.u = z.copy()                               # offset along the corridor within the zebra
        self.lat = z.copy()                             # signed lateral position, metres
        self.walk = z.copy()
        self.delay = z.copy()
        self.trigger = z.copy()
        self.state = np.zeros(0, dtype=np.int64)        # 0 waiting, 1 crossing
        self.start_at = z.copy()
        self.armed = np.zeros(0, dtype=bool)
        # signs: x, y, limit (m/s), facing dx, facing dy
        self.signs = np.zeros((0, 5))
        self.time = 0.0

    @property
    def n_crossings(self):
        return len(self.cx)

    @property
    def n_pedestrians(self):
        return len(self.pc)       # not px/py: those are derived by _place(), which reads this

    def _axes(self, idx):
        a = DIRS[self.cdir[idx]]
        p = DIRS[_right(self.cdir[idx])]
        return a, p

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    SPAWN_CLEAR_EGO = 25.0   # like Traffic's: never put a zebra on the ego's spawn

    def reset(self, city, ego_x=None, ego_y=None):
        self._clear()
        cfg = self.cfg
        if not (cfg.pedestrians or cfg.speed_signs):
            return
        nodes, links = city.road_nodes, city.node_links
        if nodes is None or len(nodes) == 0:
            return
        self.half = (city.cfg.road_width / 2.0) * city.tile_size
        sig = np.asarray(city.signals, dtype=float).reshape(-1, 2)

        cand = []
        for k in range(len(nodes)):
            arms = [d for d in range(4) if links[k, d] >= 0]
            if len(arms) < 2:
                continue
            if len(sig) and (np.hypot(sig[:, 0] - nodes[k, 0], sig[:, 1] - nodes[k, 1])
                             < city.tile_size).any():
                continue
            # A group stepping out beside a car that has just appeared is a
            # collision nobody could avoid -- the same reason traffic keeps clear.
            if ego_x is not None and np.hypot(nodes[k, 0] - ego_x, nodes[k, 1] - ego_y) < self.SPAWN_CLEAR_EGO:
                continue
            cand.append((k, arms))
        if not cand:
            return

        # Farthest-first over a shuffled list, as `_choose_signals` does, so
        # crossings land on different streets rather than one corridor.
        order = self.rng.permutation(len(cand))
        chosen = []
        for i in order:
            k, arms = cand[i]
            x, y = nodes[k]
            if all((x - nodes[j, 0]) ** 2 + (y - nodes[j, 1]) ** 2 >= cfg.crossing_min_sep ** 2
                   for j, _ in chosen):
                d = int(arms[int(self.rng.integers(len(arms)))])
                chosen.append((k, d))
                if len(chosen) >= cfg.n_crossings:
                    break

        cx, cy, cdir, zone_f, zone_b = [], [], [], [], []
        off = self.half + cfg.crossing_offset            # crossing centre -> junction centre
        for k, d in chosen:
            a = DIRS[d]
            c = nodes[k] + a * off
            # Forward: stay on this arm, short of the next junction's box.
            seg_f = float(np.hypot(*(nodes[links[k, d]] - nodes[k])))
            zf = min(cfg.zone_len, max(6.0, seg_f - off - self.half - 2.0))
            # Back: through our junction and, if the opposite arm exists, along it
            # short of the junction after; otherwise stop at our box's far edge.
            d2 = (d + 2) % 4
            if links[k, d2] >= 0:
                seg_b = float(np.hypot(*(nodes[links[k, d2]] - nodes[k])))
                zb = min(cfg.zone_len, max(6.0, off + seg_b - self.half - 2.0))
            else:
                zb = min(cfg.zone_len, off + self.half - 1.0)
            # Belt and braces: a sign must stand on road. Shrink until it does.
            r = DIRS[_right(d)] * (self.half - 0.5)
            while zf > 4.0 and not (city.is_road(*(c + a * zf + r)) and city.is_road(*(c + a * zf - r))):
                zf -= 1.0
            while zb > 4.0 and not (city.is_road(*(c - a * zb + r)) and city.is_road(*(c - a * zb - r))):
                zb -= 1.0
            cx.append(c[0]); cy.append(c[1]); cdir.append(d)
            zone_f.append(float(zf)); zone_b.append(float(zb))
        self.cx, self.cy = np.array(cx), np.array(cy)
        self.cdir = np.array(cdir, dtype=np.int64)
        self.zone_f, self.zone_b = np.array(zone_f), np.array(zone_b)

        if cfg.speed_signs:
            self._build_signs()
        if cfg.pedestrians:
            self._spawn_people()

    def _build_signs(self):
        limit = self.cfg.speed_limit_kmh / 3.6
        zone = self.cfg.zone_limit_kmh / 3.6
        rows = []
        for i in range(self.n_crossings):
            a, p = self._axes(i)
            c = np.array([self.cx[i], self.cy[i]])
            for s in (1.0, -1.0):                 # both directions of travel along the corridor
                fwd = s * a
                right = s * p                     # right of travel: (-dy, dx) of fwd
                kerb = right * (self.half - 0.5)
                # Travelling +a you enter at the back extent and leave at the front one.
                z_in, z_out = (self.zone_b[i], self.zone_f[i]) if s > 0 else (self.zone_f[i], self.zone_b[i])
                entry = c - fwd * z_in + kerb
                exit_ = c + fwd * z_out + kerb
                rows.append([entry[0], entry[1], zone, fwd[0], fwd[1]])
                rows.append([exit_[0], exit_[1], limit, fwd[0], fwd[1]])
        self.signs = np.array(rows).reshape(-1, 5)

    def _spawn_people(self):
        lo, hi = self.cfg.peds_per_crossing
        rows = []
        for i in range(self.n_crossings):
            n = int(self.rng.integers(lo, hi + 1))
            for _ in range(n):
                side = 1.0 if self.rng.random() < 0.5 else -1.0
                rows.append([i, side,
                             self.rng.uniform(-ZEBRA_HALF_LEN + 0.3, ZEBRA_HALF_LEN - 0.3),
                             self.rng.uniform(1.1, 1.6),
                             self.rng.uniform(0.0, 2.2),
                             self.rng.uniform(40.0, 90.0)])
        if not rows:
            return
        r = np.array(rows)
        self.pc = r[:, 0].astype(np.int64)
        self.side = r[:, 1]
        self.u = r[:, 2]
        self.walk = r[:, 3]
        self.delay = r[:, 4]
        self.trigger = r[:, 5]
        n = len(r)
        self.lat = self.side * (self.half - KERB_INSET)
        self.state = np.zeros(n, dtype=np.int64)
        self.start_at = np.full(n, np.inf)
        self.armed = np.ones(n, dtype=bool)
        self.pvx = np.zeros(n); self.pvy = np.zeros(n)
        self._place()

    def _place(self):
        if self.n_pedestrians == 0:
            return
        a = DIRS[self.cdir[self.pc]]                    # (N, 2)
        p = DIRS[_right(self.cdir[self.pc])]
        self.px = self.cx[self.pc] + a[:, 0] * self.u + p[:, 0] * self.lat
        self.py = self.cy[self.pc] + a[:, 1] * self.u + p[:, 1] * self.lat

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, dt, car):
        self.time += dt
        n = self.n_pedestrians
        if n == 0:
            return
        t = self.time
        cx, cy = self.cx[self.pc], self.cy[self.pc]
        dx, dy = cx - car.x, cy - car.y
        dist = np.hypot(dx, dy)
        approaching = (dx * np.cos(car.heading) + dy * np.sin(car.heading)) > 0.0

        # Re-arm groups the ego has left well behind, so a crossing met again
        # later in the episode still has people ready to step out.
        self.armed |= dist > self.trigger + REARM_MARGIN

        waiting = self.state == 0
        fire = waiting & self.armed & ~np.isfinite(self.start_at) & (dist <= self.trigger) & approaching
        self.start_at[fire] = t + self.delay[fire]
        start = waiting & (t >= self.start_at)
        self.state[start] = 1

        crossing = self.state == 1
        p = DIRS[_right(self.cdir[self.pc])]
        self.pvx = np.where(crossing, -self.side * self.walk * p[:, 0], 0.0)
        self.pvy = np.where(crossing, -self.side * self.walk * p[:, 1], 0.0)
        self.lat = np.where(crossing, self.lat - self.side * self.walk * dt, self.lat)

        edge = self.half - KERB_INSET
        arrived = crossing & (self.side * self.lat <= -edge)
        if arrived.any():
            self.lat[arrived] = -self.side[arrived] * edge
            self.side[arrived] = -self.side[arrived]
            self.state[arrived] = 0
            self.start_at[arrived] = np.inf
            self.armed[arrived] = False
        self._place()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def occupied(self):
        """Per crossing: someone is on the road or committed to stepping out."""
        occ = np.zeros(self.n_crossings, dtype=bool)
        if self.n_pedestrians:
            busy = (self.state == 1) | np.isfinite(self.start_at)
            np.logical_or.at(occ, self.pc[busy], True)
        return occ

    def on_road(self):
        return self.state == 1 if self.n_pedestrians else np.zeros(0, dtype=bool)

    def yielding(self, car, ahead_m=18.0):
        """True if a pedestrian is crossing (or about to) just ahead in our corridor --
        the stuck detector must not end an episode for waiting at a zebra, exactly
        as it must not for waiting at a red."""
        if self.n_pedestrians == 0:
            return False
        busy = (self.state == 1) | np.isfinite(self.start_at)
        if not busy.any():
            return False
        c, s = np.cos(car.heading), np.sin(car.heading)
        dx, dy = self.px[busy] - car.x, self.py[busy] - car.y
        fwd = dx * c + dy * s
        lat = np.abs(-dx * s + dy * c)
        return bool(((fwd > -1.0) & (fwd < ahead_m) & (lat < self.half + 1.0)).any())

    def limit_at(self, x, y):
        """Active speed limit (m/s) at a world point."""
        base = self.cfg.speed_limit_kmh / 3.6
        if not self.cfg.speed_signs or self.n_crossings == 0:
            return base
        a = DIRS[self.cdir]
        p = DIRS[_right(self.cdir)]
        dx, dy = x - self.cx, y - self.cy
        along = dx * a[:, 0] + dy * a[:, 1]
        lat = np.abs(dx * p[:, 0] + dy * p[:, 1])
        if ((along >= -self.zone_b) & (along <= self.zone_f) & (lat < self.half)).any():
            return self.cfg.zone_limit_kmh / 3.6
        return base

    def next_sign(self, car, rng_m):
        """(distance, limit m/s) of the nearest sign ahead facing our way, or None."""
        if len(self.signs) == 0:
            return None
        c, s = np.cos(car.heading), np.sin(car.heading)
        facing = self.signs[:, 3] * c + self.signs[:, 4] * s
        dx, dy = self.signs[:, 0] - car.x, self.signs[:, 1] - car.y
        fwd = dx * c + dy * s
        lat = np.abs(-dx * s + dy * c)
        ok = (facing > 0.7) & (fwd > 0.5) & (fwd <= rng_m) & (lat < self.half + 1.0)
        if not ok.any():
            return None
        i = int(np.argmin(np.where(ok, fwd, np.inf)))
        return float(fwd[i]), float(self.signs[i, 2])

    def circles(self, x, y, radius):
        """(K, 3) pedestrian circles within `radius`, for LIDAR -- people are geometry too."""
        if self.n_pedestrians == 0:
            return np.zeros((0, 3))
        d2 = (self.px - x) ** 2 + (self.py - y) ** 2
        idx = np.nonzero(d2 < (radius + PED_RADIUS) ** 2)[0]
        if len(idx) == 0:
            return np.zeros((0, 3))
        return np.stack([self.px[idx], self.py[idx], np.full(len(idx), PED_RADIUS)], axis=1)

    def hits(self, car):
        """True if the ego's oriented box overlaps any pedestrian."""
        if self.n_pedestrians == 0:
            return False
        c, s = np.cos(car.heading), np.sin(car.heading)
        dx, dy = self.px - car.x, self.py - car.y
        fx = dx * c + dy * s
        fy = -dx * s + dy * c
        hl, hw = car.p.length / 2.0, car.p.width / 2.0
        ox = np.maximum(np.abs(fx) - hl, 0.0)
        oy = np.maximum(np.abs(fy) - hw, 0.0)
        return bool((ox * ox + oy * oy < PED_RADIUS ** 2).any())

    def obs_features(self, car, n, rng_m, max_speed):
        """Nearest `n` pedestrians, same 5-feature layout as a traffic slot."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        pad = np.array([1.0, 0.0, 0.0, 0.0, 0.0] * n, dtype=np.float32)
        if self.n_pedestrians == 0:
            return pad
        dx, dy = self.px - car.x, self.py - car.y
        d = np.hypot(dx, dy)
        order = np.argsort(d)[:n]
        order = order[d[order] < rng_m]
        if len(order) == 0:
            return pad
        c, s = np.cos(car.heading), np.sin(car.heading)
        bearing = np.arctan2(dy[order], dx[order]) - car.heading
        vx = self.pvx[order] - car.speed * c
        vy = self.pvy[order] - car.speed * s
        out = pad.copy()
        block = np.stack([
            d[order] / rng_m,
            np.sin(bearing),
            np.cos(bearing),
            np.clip((vx * c + vy * s) / max_speed, -1.0, 1.0),
            np.clip((-vx * s + vy * c) / max_speed, -1.0, 1.0),
        ], axis=1).astype(np.float32)
        out[:block.size] = block.ravel()
        return out

    def sign_features(self, car, n, rng_m, max_speed):
        """[active limit / max_speed, next sign dist / rng_m, next sign limit / max_speed]."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        active = self.limit_at(car.x, car.y) / max_speed
        nxt = self.next_sign(car, rng_m)
        if nxt is None:
            return np.array([active, 1.0, 0.0], dtype=np.float32)
        dist, limit = nxt
        return np.array([active, min(dist / rng_m, 1.0), limit / max_speed], dtype=np.float32)
