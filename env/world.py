"""
Procedural city world.

The world is a 2D tile grid where each tile is either ROAD (drivable) or
BUILDING (solid). Everything downstream -- collisions, LIDAR, spawn points --
is derived from this grid, so the renderer never participates in simulation.

Coordinates: world space is metres with +x right, +y down, matching the grid
layout so that world (x, y) maps to grid cell (int(y / tile_size), int(x / tile_size)).
Headings are radians, 0 = +x, increasing clockwise on screen.
"""

import numpy as np
from collections import deque

ROAD = 0
BUILDING = 1


class CityConfig:
    """Generation parameters for a procedural city block layout."""

    def __init__(
        self,
        width=64,
        height=64,
        tile_size=4.0,
        # 3 tiles = 12 m. Must exceed the car's U-turn diameter
        # (2 * (min_turn_radius + half_width) ~= 9.9 m) or a car whose target is
        # behind it is geometrically stuck. Drop to 2 for a much harder env.
        road_width=3,
        min_block=3,
        max_block=7,
        border=2,
        block_segment_prob=0.10,
        plaza_prob=0.10,
        max_plaza=4,
    ):
        self.width = width                          # grid columns
        self.height = height                        # grid rows
        self.tile_size = tile_size                  # metres per tile
        self.road_width = road_width                # tiles per road corridor
        self.min_block = min_block                  # smallest building block span
        self.max_block = max_block                  # largest building block span
        self.border = border                        # solid ring keeping the car in bounds
        self.block_segment_prob = block_segment_prob  # chance a street segment is walled off
        self.plaza_prob = plaza_prob                # chance a block becomes an open plaza
        self.max_plaza = max_plaza                  # cap on plazas per map


class ProceduralCity:
    """A generated city grid with ray casting and spawn-point sampling."""

    def __init__(self, config=None, seed=None):
        self.cfg = config or CityConfig()
        self.rng = np.random.default_rng(seed)
        self.tile_size = self.cfg.tile_size
        self.width = self.cfg.width
        self.height = self.cfg.height
        self.grid = None
        self.road_cells = None      # (N, 2) array of (row, col) reachable road tiles
        self.generate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, seed=None):
        """Build a new city layout. Safe to call repeatedly for map randomisation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.cfg
        for _ in range(20):
            grid = np.full((cfg.height, cfg.width), BUILDING, dtype=np.uint8)
            v_roads = self._road_positions(cfg.width)
            h_roads = self._road_positions(cfg.height)

            for x in v_roads:
                grid[cfg.border:cfg.height - cfg.border, x:x + cfg.road_width] = ROAD
            for y in h_roads:
                grid[y:y + cfg.road_width, cfg.border:cfg.width - cfg.border] = ROAD

            self._carve_plazas(grid, v_roads, h_roads)
            self._block_segments(grid, v_roads, h_roads)
            grid[:cfg.border, :] = BUILDING
            grid[-cfg.border:, :] = BUILDING
            grid[:, :cfg.border] = BUILDING
            grid[:, -cfg.border:] = BUILDING

            cells = self._keep_largest_component(grid)
            # Reject degenerate maps (a couple of stub streets is not a city).
            if len(cells) >= 0.05 * cfg.width * cfg.height:
                self.grid = grid
                self.road_cells = cells
                return

        # Fall back to whatever the last attempt produced rather than looping forever.
        self.grid = grid
        self.road_cells = cells

    def _road_positions(self, extent):
        """Pick corridor start indices spaced by randomised block widths."""
        cfg = self.cfg
        positions = []
        pos = cfg.border
        limit = extent - cfg.border - cfg.road_width
        while pos <= limit:
            positions.append(pos)
            block = int(self.rng.integers(cfg.min_block, cfg.max_block + 1))
            pos += cfg.road_width + block
        return positions

    def _carve_plazas(self, grid, v_roads, h_roads):
        """Open up whole blocks into parking lots / squares for layout variety."""
        cfg = self.cfg
        if not v_roads or not h_roads or cfg.max_plaza <= 0:
            return
        made = 0
        for vi in range(len(v_roads) - 1):
            for hi in range(len(h_roads) - 1):
                if made >= cfg.max_plaza or self.rng.random() > cfg.plaza_prob:
                    continue
                x0 = v_roads[vi] + cfg.road_width
                x1 = v_roads[vi + 1]
                y0 = h_roads[hi] + cfg.road_width
                y1 = h_roads[hi + 1]
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    grid[y0:y1, x0:x1] = ROAD
                    made += 1

    def _block_segments(self, grid, v_roads, h_roads):
        """Wall off occasional street segments to break the pure Manhattan grid."""
        cfg = self.cfg
        rw = cfg.road_width

        for x in v_roads:
            for hi in range(len(h_roads) - 1):
                if self.rng.random() > cfg.block_segment_prob:
                    continue
                y0 = h_roads[hi] + rw
                y1 = h_roads[hi + 1]
                if y1 > y0:
                    grid[y0:y1, x:x + rw] = BUILDING

        for y in h_roads:
            for vi in range(len(v_roads) - 1):
                if self.rng.random() > cfg.block_segment_prob:
                    continue
                x0 = v_roads[vi] + rw
                x1 = v_roads[vi + 1]
                if x1 > x0:
                    grid[y:y + rw, x0:x1] = BUILDING

    def _keep_largest_component(self, grid):
        """Flood fill road tiles, keep the biggest region, wall off the rest.

        Guarantees every road tile is mutually reachable, so a sampled target is
        never unreachable from a sampled spawn.
        """
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        best = []

        for sy in range(h):
            for sx in range(w):
                if grid[sy, sx] != ROAD or visited[sy, sx]:
                    continue
                component = []
                queue = deque([(sy, sx)])
                visited[sy, sx] = True
                while queue:
                    cy, cx = queue.popleft()
                    component.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and grid[ny, nx] == ROAD:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                if len(component) > len(best):
                    best = component

        keep = np.zeros((h, w), dtype=bool)
        if best:
            cells = np.array(best, dtype=np.int32)
            keep[cells[:, 0], cells[:, 1]] = True
        else:
            cells = np.zeros((0, 2), dtype=np.int32)
        grid[(grid == ROAD) & ~keep] = BUILDING
        return cells

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def extent(self):
        """World size in metres as (x_max, y_max)."""
        return self.width * self.tile_size, self.height * self.tile_size

    def is_road(self, x, y):
        """True if the world point lies on a drivable tile."""
        col = int(np.floor(x / self.tile_size))
        row = int(np.floor(y / self.tile_size))
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False
        return bool(self.grid[row, col] == ROAD)

    def collides(self, x, y, heading, length, width):
        """Oriented-box collision test against buildings.

        Samples the four corners plus the four edge midpoints of the car's
        footprint. Unlike a single centre-point test this cannot clip a
        building corner while the centre still sits on road.
        """
        hl, hw = length / 2.0, width / 2.0
        local = np.array([
            [hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw],   # corners
            [hl, 0.0], [-hl, 0.0], [0.0, hw], [0.0, -hw],  # edge midpoints
        ])
        c, s = np.cos(heading), np.sin(heading)
        xs = x + local[:, 0] * c - local[:, 1] * s
        ys = y + local[:, 0] * s + local[:, 1] * c

        cols = np.floor(xs / self.tile_size).astype(np.int32)
        rows = np.floor(ys / self.tile_size).astype(np.int32)
        oob = (cols < 0) | (cols >= self.width) | (rows < 0) | (rows >= self.height)
        if oob.any():
            return True
        return bool((self.grid[rows, cols] == BUILDING).any())

    def cast_rays(self, x, y, angles, max_range):
        """Vectorised DDA grid traversal for a fan of rays from one origin.

        All rays advance in lockstep, one grid line crossing per iteration, with
        finished rays masked off -- exact (not sampled) distances at numpy speed.

        Returns distances in metres; rays that hit nothing return max_range.
        """
        ts = self.tile_size
        angles = np.asarray(angles, dtype=np.float64)
        n = angles.shape[0]

        px, py = x / ts, y / ts
        eps = 1e-12
        dir_x = np.cos(angles)
        dir_y = np.sin(angles)
        dir_x = np.where(np.abs(dir_x) < eps, eps, dir_x)
        dir_y = np.where(np.abs(dir_y) < eps, eps, dir_y)

        map_x = np.full(n, int(np.floor(px)), dtype=np.int64)
        map_y = np.full(n, int(np.floor(py)), dtype=np.int64)

        delta_x = np.abs(1.0 / dir_x)
        delta_y = np.abs(1.0 / dir_y)
        step_x = np.where(dir_x > 0, 1, -1).astype(np.int64)
        step_y = np.where(dir_y > 0, 1, -1).astype(np.int64)

        side_x = np.where(dir_x > 0, map_x + 1 - px, px - map_x) * delta_x
        side_y = np.where(dir_y > 0, map_y + 1 - py, py - map_y) * delta_y

        max_grid = max_range / ts
        dist = np.full(n, max_range, dtype=np.float64)
        active = np.ones(n, dtype=bool)

        # Each iteration crosses one grid line; a ray of length R spans at most 2R.
        for _ in range(int(2 * max_grid) + 4):
            if not active.any():
                break

            use_x = active & (side_x < side_y)
            use_y = active & ~use_x

            # Distance at which we enter the next cell is the *current* side value.
            travelled = np.where(use_x, side_x, side_y)

            map_x = np.where(use_x, map_x + step_x, map_x)
            side_x = np.where(use_x, side_x + delta_x, side_x)
            map_y = np.where(use_y, map_y + step_y, map_y)
            side_y = np.where(use_y, side_y + delta_y, side_y)

            oob = (map_x < 0) | (map_x >= self.width) | (map_y < 0) | (map_y >= self.height)
            safe_x = np.clip(map_x, 0, self.width - 1)
            safe_y = np.clip(map_y, 0, self.height - 1)
            solid = oob | (self.grid[safe_y, safe_x] == BUILDING)

            hit = active & solid & (travelled <= max_grid)
            dist = np.where(hit, travelled * ts, dist)
            active = active & ~solid & (travelled <= max_grid)

        return dist

    def cast_rays_fast(self, x, y, angles, max_range, step=0.25):
        """Approximate ray cast by fixed-interval sampling along each beam.

        Trades the DDA's exactness for a single batched grid lookup: every
        (beam, sample) point is tested at once, so cost is a handful of numpy ops
        on one (B, S) array instead of ~2*range/tile_size sequential iterations.
        Roughly 5-8x faster than `cast_rays`, which matters because the LIDAR
        dominates step time.

        Accuracy is +/- step/2 in the ordinary case. Plain sampling is *not* safe
        on its own: a beam grazing a convex building corner crosses only a few
        centimetres of that tile, so no sample lands inside it and the wall is
        missed entirely (measured: ~0.2% of beams, reporting 20 m of clearance
        where the true range is 6 m). That failure mode is not cosmetic -- the
        tile is a full building and a car driving down that beam crashes, so the
        agent would be punished for trusting its own sensor.

        Corner-safety fix: whenever consecutive samples change both row and
        column the beam crossed a cell corner and skipped a cell. Which one is
        decided the same way the DDA decides -- by comparing the distance to the
        vertical boundary against the distance to the horizontal one -- so the
        skipped cell is identified exactly rather than conservatively. Testing
        *both* diagonal neighbours instead would be safe but truncates a beam
        that legitimately squeezes past a corner, and those long ranges are what
        a gap-following policy steers by. For an axis-aligned step the extra
        lookup collapses onto a cell already sampled and costs nothing.
        """
        ts = self.tile_size
        angles = np.asarray(angles, dtype=np.float64)
        eps = 1e-12

        key = (float(max_range), float(step))
        if getattr(self, "_samp_key", None) != key:
            self._samp = np.arange(step, max_range + step, step)
            self._samp_key = key
        d = self._samp                                        # (S,)

        # Sample points for every beam at every distance: (B, S)
        dir_x = np.cos(angles)
        dir_y = np.sin(angles)
        xs = x + dir_x[:, None] * d[None, :]
        ys = y + dir_y[:, None] * d[None, :]
        cols = (xs / ts).astype(np.int32)
        rows = (ys / ts).astype(np.int32)

        oob = (cols < 0) | (cols >= self.width) | (rows < 0) | (rows >= self.height)
        np.clip(cols, 0, self.width - 1, out=cols)
        np.clip(rows, 0, self.height - 1, out=rows)
        solid = oob | (self.grid[rows, cols] == BUILDING)

        # Corner crossings: recover the cell the beam entered between samples.
        # step << tile_size, so at most one boundary per axis is crossed per step
        # and the boundary is at ts * max(c0, c1) whichever way the beam travels.
        r0, c0 = rows[:, :-1], cols[:, :-1]
        r1, c1 = rows[:, 1:], cols[:, 1:]
        sx = np.where(np.abs(dir_x) < eps, eps, dir_x)[:, None]
        sy = np.where(np.abs(dir_y) < eps, eps, dir_y)[:, None]
        t_vert = (ts * np.maximum(c0, c1) - x) / sx
        t_horz = (ts * np.maximum(r0, r1) - y) / sy
        vert_first = t_vert < t_horz
        er = np.where(vert_first, r0, r1)
        ec = np.where(vert_first, c1, c0)
        solid[:, 1:] |= self.grid[er, ec] == BUILDING

        hit = solid.any(axis=1)
        first = solid.argmax(axis=1)                          # 0 where no hit; masked below
        # Centre the quantisation error rather than always over-reporting clearance.
        return np.where(hit, np.maximum(d[first] - 0.5 * step, 0.0), max_range)

    # ------------------------------------------------------------------
    # Spawn sampling
    # ------------------------------------------------------------------
    def sample_road_point(self, jitter=0.5):
        """Uniformly sample a world point on a reachable road tile."""
        idx = self.rng.integers(0, len(self.road_cells))
        row, col = self.road_cells[idx]
        ts = self.tile_size
        j = jitter * ts * 0.5
        x = (col + 0.5) * ts + self.rng.uniform(-j, j)
        y = (row + 0.5) * ts + self.rng.uniform(-j, j)
        return float(x), float(y)

    def forward_clearance(self, x, y, heading, length, width, limit=12.0, step=0.5):
        """How far the car's whole footprint can advance along `heading`.

        Unlike a ray cast this sweeps the oriented box, so it accounts for the
        car's width and for a heading that is only slightly off the street axis.
        Returns `limit` if the path is clear that far.
        """
        d = step
        while d <= limit:
            if self.collides(x + np.cos(heading) * d, y + np.sin(heading) * d,
                             heading, length, width):
                return d - step
            d += step
        return limit

    def sample_free_pose(self, length, width, tries=200, align=True, jitter_deg=14.0,
                         n_probe=16, clearance_mult=2.0, min_forward=None):
        """Sample a drivable (x, y, heading) for a car of the given footprint.

        With align=True the heading is chosen by probing for the direction with
        the most room and jittering it, so the car starts *along* the street
        rather than nose-first into a facade. A random heading in a corridor only
        a few metres wider than the car is not a recoverable situation -- it is
        an instant crash the agent cannot be blamed for, and it poisons training
        with unavoidable negative returns.

        Not colliding *at* the spawn point is too weak a test on its own: the
        probe is a centre-line ray, which ignores the car's width, and the jitter
        is applied after it, so a pose can pass and still leave the car unable to
        move. The accepted pose must therefore have `min_forward` metres of swept
        room ahead of it. Measured effect: spawns that crash within 1 m of the
        start drop from 1.0% of episodes to 0.
        """
        probe = np.linspace(0, 2.0 * np.pi, n_probe, endpoint=False)
        want = length * clearance_mult
        if min_forward is None:
            min_forward = length * 1.5

        best = None
        best_room = -1.0
        for attempt in range(tries):
            x, y = self.sample_road_point(jitter=0.4)
            if align:
                d = self.cast_rays(x, y, probe, max_range=60.0)
                # Choose among the roomiest directions so both ends of a street
                # (and every arm of an intersection) stay reachable.
                order = np.argsort(-d)
                k = int(self.rng.integers(0, min(3, n_probe)))
                heading = float(probe[order[k]])
                if d[order[k]] < want:
                    continue                      # too cramped, resample position
                heading += np.radians(self.rng.uniform(-jitter_deg, jitter_deg))
            else:
                heading = float(self.rng.uniform(0, 2.0 * np.pi))

            if self.collides(x, y, heading, length, width):
                continue
            room = self.forward_clearance(x, y, heading, length, width,
                                          limit=min_forward)
            if room >= min_forward:
                return x, y, float(heading % (2.0 * np.pi))
            if room > best_room:
                best, best_room = (x, y, float(heading % (2.0 * np.pi))), room

        # Nothing met the bar (a very cramped map): return the roomiest pose seen
        # rather than an unvalidated one.
        if best is not None:
            return best
        x, y = self.sample_road_point()
        return x, y, 0.0

    def sample_point_near(self, x, y, min_dist, max_dist, tries=200):
        """Sample a road point within an annulus around (x, y).

        Used to place targets far enough to be a real navigation problem but
        close enough to be reachable inside the episode budget.
        """
        best, best_gap = None, np.inf
        for _ in range(tries):
            px, py = self.sample_road_point()
            d = np.hypot(px - x, py - y)
            if min_dist <= d <= max_dist:
                return px, py
            gap = min_dist - d if d < min_dist else d - max_dist
            if gap < best_gap:
                best, best_gap = (px, py), gap
        return best
