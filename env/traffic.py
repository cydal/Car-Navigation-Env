"""
Rule-based traffic: cars driving the street grid, plus cars parked at the kerb.

Two decisions shape everything here.

**Moving traffic runs on rails.** Each vehicle follows a route built from
corridor centre lines, resampled to a uniform 1 m polyline, and advances along it
under a speed controller. It is not a second driving agent. The alternative --
give every vehicle a steering controller -- fails badly in a 12 m corridor: one
vehicle that misjudges a corner wedges itself in a doorway and blocks a street
for the rest of the episode, and the ego is then punished for a situation it
could not have caused. On rails, a vehicle physically cannot leave the road, so
the *only* thing that can vary is its speed, which is exactly the part the ego
has to reason about.

**Traffic brakes for the ego, with finite authority.** Vehicles decelerate for
whatever is ahead in their lane, including the ego, at `A_DECEL`. That keeps
collisions the agent's fault -- the env must never hand out an unavoidable −100,
the same principle that forces the spawn-clearance check in `world.py`. But the
authority is finite and the reaction is purely longitudinal, so cutting across a
lane at the last moment still gets you hit. Traffic is forgiving, not psychic.

Vehicle footprints are the physics truth; the renderer scales each mesh to match.
Both LIDAR and collision see a vehicle as three overlapping circles along its
axis, so what the sensor reports and what you crash into can never disagree.
"""

import numpy as np

# Unit vectors for the four road directions, and the index arithmetic that goes
# with them: right-of-travel is (dx, dy) -> (-dy, dx), which is (d + 1) % 4 here.
# In these screen-down coordinates that means the +y kerb of an east-west street
# carries eastbound traffic -- a right-hand-traffic world.
DIRS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


def _right(d):
    return (d + 1) % 4


# (model stem in kenney_car-kit, length m, width m). Lengths stay at or under
# 5.6 m: a parked vehicle spans its tile plus the overhang into both neighbours,
# and `world._parking_spots` only guarantees one tile of clearance either side.
VEHICLE_KINDS = (
    ("sedan",            4.40, 1.90),
    ("sedan-sports",     4.30, 1.90),
    ("hatchback-sports", 4.10, 1.85),
    ("suv",              4.60, 2.00),
    ("suv-luxury",       4.80, 2.00),
    ("taxi",             4.40, 1.90),
    ("police",           4.50, 1.95),
    ("van",              5.20, 2.05),
    ("delivery",         5.40, 2.10),
    ("truck",            5.60, 2.10),
)

KIND_LENGTH = np.array([k[1] for k in VEHICLE_KINDS])
KIND_WIDTH = np.array([k[2] for k in VEHICLE_KINDS])

TRAFFIC_FEATURES = 5    # per moving vehicle in the observation; see obs_features


class Traffic:
    """Population of parked and moving vehicles for one episode.

    State is held as flat parallel arrays rather than per-vehicle objects so the
    whole population steps in a handful of numpy operations. That is not
    premature: the env's headline number is vector-path throughput, and a Python
    loop over a dozen vehicles per step would be a measurable share of it.
    """

    DS = 1.0                # route resample spacing, metres
    # Lane centre, metres right of the corridor centre. Not a styling choice: a
    # 12 m corridor holds two 1.9 m flows, so at 2.0 m an ego on the centre line
    # passes oncoming traffic with 0.10 m to spare on each side, which no steering
    # controller can hold -- four separate ego-side fixes (extra braking, swept
    # forward room, a standing keep-right bias, an oncoming dodge) all failed
    # because they were attacking the wrong variable. At 3.0 m the gap is 1.10 m
    # each side, and it is the largest value that still clears the parked cars
    # (kerb inset 1.10 m leaves their inner edge exactly 3.95 m out).
    LANE_OFFSET = 3.0
    FILLET = 6.0            # tangent length of the corner-rounding curve
    A_ACCEL = 2.2           # gentle pull-away
    A_DECEL = 6.0           # braking authority: finite, so traffic *can* fail
    A_LAT = 2.5             # lateral budget; sets speed through corners
    GAP_MIN = 3.5           # bumper gap held when stopped behind something
    STOP_MARGIN = 2.0       # metres short of a stop line
    ROUTE_M = 900.0         # a 50 s episode covers <= ~450 m, so routes rarely run out
    SPAWN_CLEAR_EGO = 18.0  # never spawn traffic on top of the ego
    SPAWN_CLEAR_OTHER = 9.0
    # Signals start on a random phase, so a vehicle placed inside a junction box
    # or within its own stopping distance of the line may be facing a red it
    # physically cannot honour -- it runs the light on step 0 through no fault of
    # its controller. Clear the box (6 m) plus the stopping distance from the
    # fastest cruise (9.45 m/s -> 7.4 m) plus STOP_MARGIN.
    SPAWN_CLEAR_SIGNAL = 18.0

    def __init__(self, city, cfg, rng):
        self.city = city
        self.cfg = cfg
        self.rng = rng
        self._clear()

    # ------------------------------------------------------------------
    def _clear(self):
        z = np.zeros(0)
        self.x, self.y, self.heading, self.speed = z, z.copy(), z.copy(), z.copy()
        self.length, self.width = z.copy(), z.copy()
        self.kind = np.zeros(0, dtype=np.int32)
        self.parked = np.zeros(0, dtype=bool)
        self.n_moving = 0
        self.route_xy = np.zeros((0, 1, 2))
        self.route_th = np.zeros((0, 1))
        self.route_v = np.zeros((0, 1))
        self.route_n = np.zeros(0, dtype=np.int32)
        self.s = z.copy()
        self.cruise = z.copy()
        self.end_node = np.zeros(0, dtype=np.int32)
        self.end_dir = np.zeros(0, dtype=np.int32)
        self._half = 0.0
        self._nodes = np.zeros((0, 2))
        self._sig_node = None

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def reset(self, ego_x, ego_y):
        """Spawn a fresh population. Moving vehicles come first, then parked.

        Moving first because the observation block only encodes moving vehicles,
        so keeping them in a contiguous prefix makes that slice free.
        """
        self._clear()
        cfg = self.cfg
        if not cfg.traffic:
            return

        moving = self._spawn_moving(int(cfg.n_traffic), ego_x, ego_y)
        parked = self._spawn_parked(int(cfg.n_parked), ego_x, ego_y, moving)
        self.n_moving = len(moving)

        rows = moving + parked
        if not rows:
            return
        self.x = np.array([r[0] for r in rows])
        self.y = np.array([r[1] for r in rows])
        self.heading = np.array([r[2] for r in rows])
        self.kind = np.array([r[3] for r in rows], dtype=np.int32)
        self.length = KIND_LENGTH[self.kind].copy()
        self.width = KIND_WIDTH[self.kind].copy()
        self.speed = np.zeros(len(rows))
        self.parked = np.array([False] * len(moving) + [True] * len(parked))

        if moving:
            self._pack_routes([r[4] for r in moving])
            self.s = np.array([r[5] for r in moving])
            self.cruise = np.array([r[6] for r in moving])
            self.end_node = np.array([r[7] for r in moving], dtype=np.int32)
            self.end_dir = np.array([r[8] for r in moving], dtype=np.int32)
            self._apply_routes()
            self.speed[:self.n_moving] = self.cruise * 0.6
            self._cache_static()

    def _cache_static(self):
        """Precompute everything `step` would otherwise rebuild 50 times a second.

        Signals and junctions do not move, and a vehicle's size never changes, so
        rebuilding those arrays every step is pure overhead -- and at eight
        vehicles the per-call numpy cost dominates the arithmetic, which is why
        this is worth doing at all.
        """
        n = self.n_moving
        self._half = (self.city.cfg.road_width / 2.0) * self.city.tile_size
        nodes = self.city.road_nodes
        self._nodes = (np.zeros((0, 2)) if nodes is None or len(nodes) == 0
                       else np.asarray(nodes, dtype=float))
        # `_follow_target` compares against every moving vehicle *plus the ego*, so
        # carry a trailing slot that only the ego's live values are written into.
        self._f_len = np.empty(n + 1)
        self._f_wid = np.empty(n + 1)
        self._f_len[:n] = self.length[:n]
        self._f_wid[:n] = self.width[:n]
        self._f_x = np.empty(n + 1)
        self._f_y = np.empty(n + 1)
        self._f_v = np.empty(n + 1)

        # Signalised crossings are a subset of the junction lattice, so a signal is
        # stored as a *column index* into the shared geometry rather than as its own
        # coordinates. `_choose_signals` picks from `_find_intersections`, which are
        # the same 4-way crossings `_build_road_graph` emits, so the match is exact
        # up to float noise -- but take the nearest node and assert it, because a
        # silent mismatch would have traffic stopping for a light on another street.
        sig = np.asarray(self.city.signals, dtype=float).reshape(-1, 2)
        if len(sig) == 0 or len(self._nodes) == 0:
            self._sig_node = np.zeros(0, dtype=np.int64)
        else:
            d = np.hypot(self._nodes[None, :, 0] - sig[:, 0, None],
                         self._nodes[None, :, 1] - sig[:, 1, None])
            self._sig_node = np.argmin(d, axis=1)
            assert d.min(axis=1).max() < self.city.tile_size, "signal off the road graph"

    def _spawn_moving(self, n, ego_x, ego_y):
        nodes = self.city.road_nodes
        if n <= 0 or nodes is None or len(nodes) < 2:
            return []

        sig = np.asarray(self.city.signals, dtype=float).reshape(-1, 2)

        rows = []
        for _ in range(n):
            for _try in range(20):
                route, end_node, end_dir = self._make_route()
                if route is None or len(route[0]) < 12:
                    continue
                pts = route[0]
                # Start somewhere in the first stretch, not always at a junction.
                s0 = float(self.rng.uniform(0.0, min(40.0, (len(pts) - 6) * self.DS)))
                px, py = pts[int(s0 / self.DS)]
                if (px - ego_x) ** 2 + (py - ego_y) ** 2 < self.SPAWN_CLEAR_EGO ** 2:
                    continue
                if len(sig) and (np.hypot(sig[:, 0] - px, sig[:, 1] - py)
                                 < self.SPAWN_CLEAR_SIGNAL).any():
                    continue
                if any((px - r[0]) ** 2 + (py - r[1]) ** 2 < self.SPAWN_CLEAR_OTHER ** 2
                       for r in rows):
                    continue
                kind = int(self.rng.integers(len(VEHICLE_KINDS)))
                cruise = float(self.rng.uniform(0.75, 1.05)) * self.cfg.traffic_speed
                rows.append([px, py, 0.0, kind, route, s0, cruise, end_node, end_dir])
                break
        return rows

    def _spawn_parked(self, n, ego_x, ego_y, moving):
        spots = self.city.parking_spots
        if n <= 0 or spots is None or len(spots) == 0:
            return []

        order = self.rng.permutation(len(spots))
        rows = []
        for i in order:
            if len(rows) >= n:
                break
            px, py, ph = spots[i]
            # Leave the ego's spawn alone: it is aligned with the roomiest
            # direction and checked for swept clearance against *buildings*, so a
            # car parked on top of it would reintroduce the unwinnable spawn the
            # clearance check exists to remove.
            if (px - ego_x) ** 2 + (py - ego_y) ** 2 < self.SPAWN_CLEAR_EGO ** 2:
                continue
            # Spread them along the kerb. This also stops two cars parking
            # directly opposite each other, which would leave 7.7 m of a 12 m
            # corridor -- under the car's 9.9 m U-turn diameter.
            if any((px - r[0]) ** 2 + (py - r[1]) ** 2 < 12.0 ** 2 for r in rows):
                continue
            if any((px - r[0]) ** 2 + (py - r[1]) ** 2 < 10.0 ** 2 for r in moving):
                continue
            rows.append([px, py, float(ph), int(self.rng.integers(len(VEHICLE_KINDS)))])
        return rows

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def _make_route(self, node=None, d=None):
        """Build one vehicle's route: (points, headings, speed limit) + end state."""
        nodes = self.city.road_nodes
        links = self.city.node_links
        if nodes is None or len(nodes) == 0:
            return None, 0, 0
        if node is None:
            node = int(self.rng.integers(len(nodes)))
        seq_n, seq_d = self._walk(node, d, self.ROUTE_M)
        if len(seq_d) < 2:
            return None, 0, 0
        pts = self._lane_polyline(seq_n, seq_d)
        route = self._profile(pts)
        return route, seq_n[-1], seq_d[-1]

    def _walk(self, node, prev_d, min_len):
        """Random walk on the corridor lattice, biased toward going straight."""
        nodes = self.city.road_nodes
        links = self.city.node_links
        seq_n, seq_d = [node], []
        total = 0.0
        while total < min_len:
            opts = [k for k in range(4) if links[node, k] >= 0]
            if not opts:
                break
            if prev_d is not None:
                back = (prev_d + 2) % 4
                # Reversing is a last resort, which is what keeps traffic out of
                # pointless U-turns at every leaf of the lattice.
                fwd = [k for k in opts if k != back]
                opts = fwd if fwd else opts
            w = np.array([3.0 if k == prev_d else 1.0 for k in opts])
            k = int(self.rng.choice(opts, p=w / w.sum()))
            nxt = int(links[node, k])
            total += float(np.hypot(*(nodes[nxt] - nodes[node])))
            seq_n.append(nxt)
            seq_d.append(k)
            node, prev_d = nxt, k
        return seq_n, seq_d

    def _lane_polyline(self, seq_n, seq_d):
        """Lane centre line through the node sequence, with rounded corners."""
        nodes = self.city.road_nodes
        L = self.LANE_OFFSET
        pts = [nodes[seq_n[0]] + L * DIRS[_right(seq_d[0])]]
        for k, d in enumerate(seq_d):
            B = nodes[seq_n[k + 1]]
            if k + 1 == len(seq_d):
                pts.append(B + L * DIRS[_right(d)])
                break
            d2 = seq_d[k + 1]
            if d2 == d:
                continue                        # straight through; lane unchanged
            if d2 == (d + 2) % 4:
                pts.extend(self._uturn_arc(B, d, L))
                continue
            P = self._corner(B, d, d2, L)
            pts.extend(_bezier(P - self.FILLET * DIRS[d], P, P + self.FILLET * DIRS[d2]))
        return pts

    @staticmethod
    def _corner(B, d, d2, L):
        """Where the two offset lane lines meet -- the vertex a turn rounds off.

        Right turns land this *before* the junction centre and left turns after
        it, which is why turning traffic reads correctly instead of pivoting on
        the spot.
        """
        d1v, d2v = DIRS[d], DIRS[d2]
        r1, r2 = DIRS[_right(d)], DIRS[_right(d2)]
        A = np.array([[d1v[0], -d2v[0]], [d1v[1], -d2v[1]]])
        t, _u = np.linalg.solve(A, L * (r2 - r1))
        return B + L * r1 + t * d1v

    @staticmethod
    def _uturn_arc(B, d, L, n=9):
        """Semicircle joining the two lanes of one street, for a forced reversal."""
        r = DIRS[_right(d)]
        a0 = np.arctan2(r[1], r[0])
        return [B + L * np.array([np.cos(a), np.sin(a)])
                for a in a0 - np.linspace(0.0, np.pi, n)]

    def _profile(self, pts):
        """Resample to uniform spacing and precompute heading + speed limit."""
        p = np.asarray(pts, dtype=np.float64)
        keep = np.concatenate([[True], (np.abs(np.diff(p, axis=0)) > 1e-9).any(axis=1)])
        p = p[keep]
        if len(p) < 3:
            return None
        seg = np.hypot(*np.diff(p, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, cum[-1], self.DS)
        if len(s) < 12:
            return None
        xy = np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1)

        step = np.diff(xy, axis=0)
        th = np.arctan2(step[:, 1], step[:, 0])
        th = np.unwrap(np.concatenate([th, th[-1:]]))

        # Corner speed from the lateral-acceleration budget, then a backward pass
        # so a vehicle is always already slow enough to take the next corner --
        # without it, traffic arrives at a turn too fast and the speed limiter
        # snaps it, which looks like teleporting and is not something a world
        # model could learn to predict.
        curv = np.abs(np.diff(th, prepend=th[0])) / self.DS
        v = np.minimum(np.sqrt(self.A_LAT / np.maximum(curv, 1e-4)), 1e3)
        v[-1] = 0.0

        # The backward pass in closed form. `u[i] = min(u[i], u[i+1] + k)` on
        # u = v^2 is a sequential recurrence, but its fixed point is
        # `min over j >= i of u[j] + k*(j - i)`, which factorises into a reverse
        # running minimum of `u[j] + k*j`. Worth the algebra: routes are ~900
        # samples long and eight of them are built per reset, so the obvious
        # Python loop was the single most expensive thing in `reset`.
        k = 2.0 * self.A_DECEL * self.DS
        j = np.arange(len(v)) * k
        u = np.minimum.accumulate((v * v + j)[::-1])[::-1] - j
        v = np.sqrt(np.maximum(u, 0.0))
        return xy, th, v

    def _pack_routes(self, routes):
        n = len(routes)
        p = max(len(r[0]) for r in routes)
        self.route_xy = np.zeros((n, p, 2))
        self.route_th = np.zeros((n, p))
        self.route_v = np.zeros((n, p))
        self.route_n = np.zeros(n, dtype=np.int32)
        for i, (xy, th, v) in enumerate(routes):
            k = len(xy)
            self.route_xy[i, :k] = xy
            self.route_th[i, :k] = th
            self.route_v[i, :k] = v
            # Pad by holding the last pose at zero speed, so a vehicle whose
            # route runs out parks itself instead of indexing into garbage.
            self.route_xy[i, k:] = xy[-1]
            self.route_th[i, k:] = th[-1]
            self.route_n[i] = k

    def _regenerate(self, i):
        """Give vehicle i a new route from where its last one ended."""
        route, end_node, end_dir = self._make_route(int(self.end_node[i]),
                                                    int(self.end_dir[i]))
        if route is None:
            return
        xy, th, v = route
        k = min(len(xy), self.route_xy.shape[1])
        self.route_xy[i, :k] = xy[:k]
        self.route_th[i, :k] = th[:k]
        self.route_v[i, :k] = v[:k]
        self.route_xy[i, k:] = xy[k - 1]
        self.route_th[i, k:] = th[k - 1]
        self.route_v[i, k:] = 0.0
        self.route_n[i] = k
        self.end_node[i], self.end_dir[i] = end_node, end_dir
        self.s[i] = 0.0

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def _apply_routes(self):
        """Write pose for every moving vehicle from its arc-length position."""
        n = self.n_moving
        if n == 0:
            return
        rows = np.arange(n)
        f = self.s / self.DS
        i0 = np.clip(f.astype(np.int64), 0, self.route_n - 1)
        i1 = np.clip(i0 + 1, 0, self.route_n - 1)
        frac = np.clip(f - i0, 0.0, 1.0)[:, None]

        p0 = self.route_xy[rows, i0]
        p1 = self.route_xy[rows, i1]
        p = p0 + (p1 - p0) * frac
        self.x[:n], self.y[:n] = p[:, 0], p[:, 1]

        t0 = self.route_th[rows, i0]
        t1 = self.route_th[rows, i1]
        self.heading[:n] = (t0 + (t1 - t0) * frac[:, 0]) % (2.0 * np.pi)
        self._idx = i0

    def step(self, dt, car, lights, street=None):
        """Advance all moving vehicles one timestep."""
        n = self.n_moving
        if n == 0:
            return
        rows = np.arange(n)
        geom = self._junction_geometry()
        target = self.route_v[rows, self._idx]
        target = np.minimum(target, self.cruise)
        target = np.minimum(target, self._signal_target(lights, geom))
        target = np.minimum(target, self._junction_target(geom, car))
        target = np.minimum(target, self._follow_target(car))
        if street is not None and street.n_crossings:
            target = np.minimum(target, self._crossing_target(street))

        dv = target - self.speed[:n]
        self.speed[:n] = np.maximum(0.0, self.speed[:n] + np.clip(
            dv, -self.A_DECEL * dt, self.A_ACCEL * dt))
        self.s += self.speed[:n] * dt

        for i in np.nonzero(self.s >= (self.route_n - 2) * self.DS)[0]:
            self._regenerate(int(i))
        self._apply_routes()

    def _junction_geometry(self):
        """Where each moving vehicle sits relative to every junction box.

        Returns `(in_box, along, lateral, c, s, ew)` where `along` is the signed
        distance to the near stop line along the direction of travel and `lateral`
        the offset from the crossing corridor.

        Computed once per step and shared by the signal check and the reservation.
        Signals sit on a *subset* of these same nodes, so doing it twice meant
        duplicating the trigonometry on the hottest path in the module.

        A traffic vehicle's axis is never ambiguous the way the ego's is: it is
        travelling down a corridor centre line by construction, so the sign of its
        heading settles the question outright.
        """
        n = self.n_moving
        nodes = self._nodes
        half = self._half
        dx = nodes[None, :, 0] - self.x[:n, None]            # (N, K)
        dy = nodes[None, :, 1] - self.y[:n, None]
        c, s = np.cos(self.heading[:n]), np.sin(self.heading[:n])
        ew = np.abs(c) >= np.abs(s)                          # travelling along x?
        col = ew[:, None]
        in_box = (np.abs(dx) < half) & (np.abs(dy) < half)
        along = np.where(col, dx * np.sign(c)[:, None],
                         dy * np.sign(s)[:, None]) - half
        lateral = np.where(col, np.abs(dy), np.abs(dx))
        return in_box, along, lateral, c, s, ew

    def _signal_target(self, lights, geom):
        """Speed cap from the next red or yellow on each vehicle's own approach."""
        n = self.n_moving
        if not lights or self._sig_node is None or len(self._sig_node) == 0:
            return np.full(n, np.inf)

        half = self._half
        _, along, lateral, _, _, ew = geom
        cols = self._sig_node
        along = along[:, cols]
        lateral = lateral[:, cols]

        red = np.array([[tl.state('ew') != 'green', tl.state('ns') != 'green']
                        for tl in lights])          # (M, 2)
        stop = np.where(ew[:, None], red[None, :, 0], red[None, :, 1])

        # Strictly *before* the stop line. Once past it a vehicle always clears the
        # box, which is both the real rule and a hard requirement for the junction
        # reservation below: a vehicle frozen mid-junction by a light that changed
        # under it would hold its reservation forever and deadlock every approach.
        claims = stop & (along > 0.0) & (lateral < half)
        room = np.maximum(0.0, along - self.STOP_MARGIN)
        v = np.where(claims, np.sqrt(2.0 * self.A_DECEL * room), np.inf)
        return v.min(axis=1)

    def _junction_target(self, geom, car):
        """Speed cap from one-vehicle-at-a-time reservation of each junction.

        Only 6 of ~56 crossings carry signals, so without this traffic simply
        drives through itself at every unsignalised junction -- measured at a
        0.25 m closest centre-to-centre approach, i.e. cars visibly
        interpenetrating. That is worse than it sounds: the merged geometry ends up
        in LIDAR and in the image, and it teaches a world model that vehicles pass
        through each other.

        The protocol is a four-way stop. A junction is claimed by whoever is inside
        it; among vehicles still approaching, the nearest goes first. Only
        *crossing* traffic counts -- a vehicle in the same lane is a follower, and
        making it wait for the junction to clear as well would turn every platoon
        into single-file stop-start.

        **The ego is in this lattice too.** Without it, a vehicle drives through an
        ego sitting in a junction at full cruise: `_follow_target` is the only thing
        that sees the ego at all and it is same-lane longitudinal, so an ego crossing
        at 90 degrees is invisible to every part of the traffic model. That made
        crossing conflicts the largest single category of ego collision, 22 of 43,
        at a mean ego speed of 4.2 m/s -- i.e. the ego was the slower party in most
        of them. The rule applied is exactly the one traffic already uses on itself,
        the box belongs to whoever is inside it, so the ego still has to judge a gap
        to enter; it just is not T-boned once it is committed and cannot retreat.

        Each vehicle is reduced to two junction indices before any pairwise work:
        the box it is inside (at most one -- boxes are 12 m across and no two
        junctions sit closer than ~29 m) and the nearest one ahead, which is the
        only one it can be stopped by. That turns the reservation from an
        (N, N, K) tensor over every junction into an (N, N) comparison of indices.
        """
        n = self.n_moving
        if n == 0 or len(self._nodes) == 0:
            return np.full(n, np.inf)

        half = self._half
        in_box, along, lateral, c, s, _ = geom
        ahead = (~in_box) & (along > 0.0) & (along < 32.0) & (lateral < half)

        # -1 when there is nothing to reduce, which no real junction index equals.
        j_in = np.where(in_box.any(axis=1), in_box.argmax(axis=1), -1)
        d_all = np.where(ahead, along, np.inf)
        j_next = np.argmin(d_all, axis=1)
        d = d_all[np.arange(n), j_next]
        approaching = np.isfinite(d)
        j_next = np.where(approaching, j_next, -1)

        # Lateral offset of every other vehicle from my line of travel: small means
        # it shares my lane, so `_follow_target` owns it, not the reservation.
        pdx = self.x[None, :n] - self.x[:n, None]
        pdy = self.y[None, :n] - self.y[:n, None]
        cross = np.abs(-pdx * s[:, None] + pdy * c[:, None]) > 1.2

        mine = j_next[:, None]                              # (N, 1) my next junction
        held = cross & (mine == j_in[None, :])
        beaten = (cross & (mine == j_next[None, :])
                  & (d[None, :] < d[:, None]))

        # Same reservation, with the ego as the holder. It only ever *holds* a box,
        # never wins an approach race, so the ego gains no priority it has not
        # already taken by being there.
        e_box = ((np.abs(self._nodes[:, 0] - car.x) < half)
                 & (np.abs(self._nodes[:, 1] - car.y) < half))
        e_in = int(np.argmax(e_box)) if e_box.any() else -1
        e_cross = np.abs(-(car.x - self.x[:n]) * s + (car.y - self.y[:n]) * c) > 1.2
        held_ego = e_cross & (j_next == e_in) & (e_in >= 0)

        blocked = approaching & ((held | beaten).any(axis=1) | held_ego)

        room = np.maximum(0.0, d - self.STOP_MARGIN)
        return np.where(blocked, np.sqrt(2.0 * self.A_DECEL * room), np.inf)

    def _crossing_target(self, street):
        """Speed cap from an occupied zebra crossing ahead on the vehicle's corridor.

        Traffic yields to people, the ego does not get that for free -- the
        same asymmetry as signals, where traffic obeys the light and the ego
        has to learn to. A vehicle already past the crossing centre is
        released, so a person stepping out behind it cannot freeze it.
        """
        n = self.n_moving
        occ = street.occupied()
        if not occ.any():
            return np.full(n, np.inf)
        cx, cy = street.cx[occ], street.cy[occ]
        dx = cx[None, :] - self.x[:n, None]
        dy = cy[None, :] - self.y[:n, None]
        c, s = np.cos(self.heading[:n]), np.sin(self.heading[:n])
        along = dx * c[:, None] + dy * s[:, None]
        lat = np.abs(-dx * s[:, None] + dy * c[:, None])
        claims = (along > 0.0) & (along < 45.0) & (lat < street.half)
        # Rest with the nose a stop margin short of the zebra's near edge.
        room = np.maximum(0.0, along - 1.5 - self.STOP_MARGIN - self.length[:n, None] / 2.0)
        v = np.where(claims, np.sqrt(2.0 * self.A_DECEL * room), np.inf)
        return v.min(axis=1)

    def _follow_target(self, car):
        """Speed cap from the nearest thing ahead in the same lane.

        Parked cars are excluded on purpose. They sit ~1.95 m off the lane centre,
        inside any sane same-lane threshold, so counting them would stop every
        vehicle behind every parked car and freeze the city. The lane is clear of
        them by construction -- that is what `LANE_OFFSET` buys.
        """
        n = self.n_moving
        ox, oy, ov = self._f_x, self._f_y, self._f_v
        ow, ol = self._f_wid, self._f_len
        ox[:n] = self.x[:n]; ox[n] = car.x
        oy[:n] = self.y[:n]; oy[n] = car.y
        ov[:n] = self.speed[:n]; ov[n] = car.speed
        ow[n] = car.p.width
        ol[n] = car.p.length

        h = self.heading[:n]
        c, s = np.cos(h)[:, None], np.sin(h)[:, None]
        dx = ox[None, :] - self.x[:n, None]
        dy = oy[None, :] - self.y[:n, None]
        fwd = dx * c + dy * s                       # along own heading
        lat = np.abs(-dx * s + dy * c)

        gap = fwd - (self.length[:n, None] + ol[None, :]) / 2.0
        same_lane = lat < (self.width[:n, None] + ow[None, :]) / 2.0 + 0.4
        ahead = same_lane & (fwd > 0.0) & (gap < 40.0)
        np.fill_diagonal(ahead[:, :n], False)       # never follow yourself

        # Match the leader's speed plus whatever the remaining gap allows.
        room = np.maximum(0.0, gap - self.GAP_MIN)
        v = np.where(ahead, np.sqrt(ov[None, :] ** 2 + 2.0 * self.A_DECEL * room), np.inf)
        return v.min(axis=1)

    # ------------------------------------------------------------------
    # Sensing and collision
    # ------------------------------------------------------------------
    def _select(self, x, y, radius):
        if len(self.x) == 0:
            return np.zeros(0, dtype=np.int64)
        d2 = (self.x - x) ** 2 + (self.y - y) ** 2
        lim = radius + self.length.max() / 2.0
        return np.nonzero(d2 < lim * lim)[0]

    def circles(self, x, y, radius):
        """(K, 3) array of (cx, cy, r) approximating every vehicle within range.

        Three circles per vehicle, radius = half-width, spaced so their union is
        exactly the body's rounded rectangle: with the centres at 0 and
        +/-(L-W)/2 the union has no gap as long as L <= 3W, which every entry in
        VEHICLE_KINDS satisfies. Using this for LIDAR *and* for collision is the
        point -- the ranges the agent is given can never disagree with the shape
        it crashes into.
        """
        idx = self._select(x, y, radius)
        if len(idx) == 0:
            return np.zeros((0, 3))
        off = (self.length[idx] - self.width[idx]) / 2.0
        c, s = np.cos(self.heading[idx]), np.sin(self.heading[idx])
        k = np.array([-1.0, 0.0, 1.0])[None, :]
        cx = (self.x[idx][:, None] + k * off[:, None] * c[:, None]).ravel()
        cy = (self.y[idx][:, None] + k * off[:, None] * s[:, None]).ravel()
        r = np.repeat(self.width[idx] / 2.0, 3)
        return np.stack([cx, cy, r], axis=1)

    def hits(self, car):
        """True if the ego's oriented box overlaps any vehicle."""
        reach = car.p.length / 2.0 + 3.0
        circ = self.circles(car.x, car.y, reach)
        if len(circ) == 0:
            return False
        # Circle centres into the ego's body frame, then closest point on the box.
        c, s = np.cos(car.heading), np.sin(car.heading)
        dx, dy = circ[:, 0] - car.x, circ[:, 1] - car.y
        fx = dx * c + dy * s
        fy = -dx * s + dy * c
        hl, hw = car.p.length / 2.0, car.p.width / 2.0
        ox = np.maximum(np.abs(fx) - hl, 0.0)
        oy = np.maximum(np.abs(fy) - hw, 0.0)
        return bool((ox * ox + oy * oy < circ[:, 2] ** 2).any())

    def obs_features(self, car, n, rng_m, max_speed):
        """Nearest `n` *moving* vehicles, ego-centric, in [-1, 1].

        Per vehicle (TRAFFIC_FEATURES values):
            dist_norm    centre-to-centre distance / rng_m
            sin_bearing  ego-centric bearing to the vehicle
            cos_bearing
            vx_rel       its velocity minus ours, in our frame, / max_speed
            vy_rel

        This block exists for the velocity alone. LIDAR already reports where the
        metal is and roughly how it is shaped, but a single scan cannot say
        whether the car ahead is stopped or doing 10 m/s away from you -- so
        without velocity the observation is not Markov for anything involving
        traffic, the same argument that puts `steer_angle` in the dynamics block.

        Parked cars are deliberately absent: their velocity is always zero, so
        they would fill the nearest-`n` slots with nothing LIDAR does not already
        carry. They are static geometry, no different in kind from a wall.

        Absent slots pad with sin = cos = 0, which is not a unit vector and so
        cannot be confused with a real bearing.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        pad = np.array([1.0, 0.0, 0.0, 0.0, 0.0] * n, dtype=np.float32)
        m = self.n_moving
        if m == 0:
            return pad

        dx = self.x[:m] - car.x
        dy = self.y[:m] - car.y
        d = np.hypot(dx, dy)
        order = np.argsort(d)[:n]
        order = order[d[order] < rng_m]
        if len(order) == 0:
            return pad

        c, s = np.cos(car.heading), np.sin(car.heading)
        dxo, dyo = dx[order], dy[order]
        bearing = np.arctan2(dyo, dxo) - car.heading
        vx = self.speed[order] * np.cos(self.heading[order]) - car.speed * c
        vy = self.speed[order] * np.sin(self.heading[order]) - car.speed * s

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


def _bezier(p0, p1, p2, n=9):
    """Quadratic Bezier, used to round a lane corner with a continuous tangent."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return list((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)
