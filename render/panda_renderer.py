"""
Third-person Panda3D renderer for CarNavEnv.

This module is *only* imported when images are actually wanted -- the simulation
core never touches it, so headless training never loads a graphics stack.

Two design constraints drove the implementation:

1. `build_scene` runs on every episode reset, because the map is resampled.
   Creating a node per building tile and letting Panda flatten them takes
   hundreds of milliseconds, which would dominate reset cost. Instead the whole
   city is emitted as a *single* mesh: face visibility, vertices and normals are
   computed with numpy and the vertex buffer is handed to Panda as raw bytes.
   Interior faces between adjacent buildings are culled at build time.

2. `capture` is a pure function of the environment state -- no camera smoothing,
   no frame-to-frame carry-over. A smoothed chase camera looks nicer to a human
   but makes the image observation depend on history, which silently breaks both
   determinism under a fixed seed and the Markov property the vector state is
   careful to preserve. Smoothing is available but defaults off; turn it on for
   watching, not for training.
"""

import numpy as np
from panda3d.core import loadPrcFileData

# Must be configured before ShowBase is constructed.
loadPrcFileData("", "audio-library-name null")   # no sound device, faster startup
loadPrcFileData("", "sync-video 0")              # never block on vsync
loadPrcFileData("", "notify-level-display error")

from panda3d.core import (  # noqa: E402
    AmbientLight, CardMaker, DirectionalLight, Fog, Geom, GeomNode, GeomTriangles,
    GeomVertexArrayFormat, GeomVertexData, GeomVertexFormat, GraphicsOutput,
    InternalName, LineSegs, NodePath, Texture, Vec3, Vec4,
)

# position + normal + rgba, all float32 -- matches VERTEX_DTYPE below byte for byte.
_afmt = GeomVertexArrayFormat()
_afmt.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
_afmt.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
_afmt.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
VERTEX_FORMAT = GeomVertexFormat.registerFormat(GeomVertexFormat(_afmt))

VERTEX_DTYPE = np.dtype([("v", "<f4", 3), ("n", "<f4", 3), ("c", "<f4", 4)])

# A quad's four corners expand to two triangles in this order.
_QUAD_TO_TRIS = np.array([0, 1, 2, 0, 2, 3])

SKY = (0.53, 0.62, 0.74)

_BASE = None        # ShowBase is a process-wide singleton in Panda3D.


def _get_base(size, offscreen):
    """Create (or reuse) the single ShowBase instance for this process."""
    global _BASE
    if _BASE is None:
        loadPrcFileData("", f"window-type {'offscreen' if offscreen else 'onscreen'}")
        loadPrcFileData("", f"win-size {size[0]} {size[1]}")
        from direct.showbase.ShowBase import ShowBase
        _BASE = ShowBase()
        _BASE.disableMouse()
        _BASE.setBackgroundColor(*SKY)
    return _BASE


def _hash01(a, b):
    """Deterministic pseudo-random floats in [0, 1) from two integer arrays.

    Used for building heights and shades. Keyed on tile coordinates rather than
    an RNG so a given tile looks the same every time it appears, which makes
    successive procedural maps feel like one consistent city.
    """
    h = np.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return h - np.floor(h)


def _quads_to_mesh(corners, normals, colors):
    """Turn (N, 4, 3) quad corners into an (N*6,) structured vertex array."""
    n_quads = corners.shape[0]
    if n_quads == 0:
        return np.zeros(0, dtype=VERTEX_DTYPE)
    tris = corners[:, _QUAD_TO_TRIS, :].reshape(-1, 3)
    out = np.zeros(len(tris), dtype=VERTEX_DTYPE)
    out["v"] = tris
    out["n"] = np.repeat(normals, 6, axis=0)
    out["c"] = np.repeat(colors, 6, axis=0)
    return out


def _mesh_to_node(name, verts):
    """Wrap a structured vertex array in a Panda GeomNode via a raw byte copy."""
    vdata = GeomVertexData(name, VERTEX_FORMAT, Geom.UHStatic)
    vdata.setNumRows(len(verts))
    vdata.modifyArray(0).modifyHandle().setData(verts.tobytes())

    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(Geom.NTUint32)
    prim.addConsecutiveVertices(0, len(verts))
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def _box_mesh(lo, hi, color, top_color=None):
    """Axis-aligned box as 6 outward-facing quads. lo/hi are (x, y, z) tuples."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [
        # (corners, normal)
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
        ([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], (0, 0, -1)),
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
        ([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], (0, 1, 0)),
        ([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], (-1, 0, 0)),
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)),
    ]
    corners = np.array([f[0] for f in faces], dtype=np.float32)
    normals = np.array([f[1] for f in faces], dtype=np.float32)
    cols = np.tile(np.asarray(color, dtype=np.float32), (6, 1))
    if top_color is not None:
        cols[0] = top_color
    return _quads_to_mesh(corners, normals, cols)


class PandaRenderer:
    """Renders CarNavEnv from a chase camera to an RGB array or a window."""

    def __init__(self, offscreen=True, size=64, fov=60.0,
                 cam_dist=13.0, cam_height=6.5, look_ahead=9.0, smooth=0.0,
                 build_min=6.0, build_max=24.0, show_rays=None, show_minimap=None):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.offscreen = offscreen
        self.cam_dist = cam_dist
        self.cam_height = cam_height
        self.look_ahead = look_ahead
        self.smooth = smooth          # 0 = stateless (required for training)
        # Proximity ring defaults on for onscreen windows, off for offscreen so
        # it never appears in training image observations.
        self.show_rays = show_rays if show_rays is not None else (not offscreen)
        # Minimap is on by default for onscreen (demo) windows, off for offscreen
        # (training image captures) so the map overlay never enters training obs.
        self.show_minimap = show_minimap if show_minimap is not None else (not offscreen)
        self.build_min = build_min
        self.build_max = build_max

        self.base = _get_base(self.size, offscreen)
        self.base.camLens.setFov(fov)
        self.base.camLens.setNear(0.3)
        self.base.camLens.setFar(400.0)

        self._city_np = None
        self._rays_np = None
        self._cam_pos = None          # only used when smooth > 0
        self._setup_lights()
        self._setup_actors()

        # 2-D overlay minimap (top-right corner, onscreen only).
        self._mm_np = None
        self._mm_city_np = None
        self._mm_city_tex = None
        self._mm_car_np = None
        self._mm_wpt_nps = []
        self._mm_extent = (1.0, 1.0)
        if self.show_minimap:
            self._setup_minimap_frame()

        self._tex = None
        if offscreen:
            self._tex = Texture()
            self.base.win.addRenderTexture(
                self._tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

    # ------------------------------------------------------------------
    def _setup_lights(self):
        render = self.base.render
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.45, 0.47, 0.52, 1))
        render.setLight(render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.85, 0.82, 0.75, 1))
        sun_np = render.attachNewNode(sun)
        sun_np.setHpr(-35, -55, 0)
        render.setLight(sun_np)

        # Hides the map boundary and gives the flat grid some depth cueing.
        fog = Fog("fog")
        fog.setColor(*SKY)
        fog.setExpDensity(0.0055)
        render.setFog(fog)

    def _setup_actors(self):
        """Build the car and target markers once; they are only repositioned."""
        import os
        self.car_np = self.base.render.attachNewNode("car")

        _here = os.path.dirname(os.path.abspath(__file__))
        glb = os.path.normpath(os.path.join(
            _here, "..", "kenney_car-kit", "Models", "GLB format", "sedan.glb"))

        if os.path.exists(glb):
            model = self.base.loader.loadModel(glb)
            model.reparentTo(self.car_np)
            # The Kenney sedan faces +Y after glTF load; our world forward is +X.
            # setH(-90) on the sub-node rotates the nose from +Y to +X.
            model.setH(90)
            # Sedan bounding box: Y=2.55 m long.  Scale so length ≈ 4.4 m.
            model.setScale(1.72)
        else:
            # Fallback: procedural box mesh (no external assets needed).
            RED   = (0.80, 0.16, 0.13, 1)
            DARK  = (0.13, 0.14, 0.17, 1)
            GREY  = (0.22, 0.22, 0.24, 1)
            BLACK = (0.10, 0.10, 0.11, 1)
            HEAD  = (0.95, 0.92, 0.75, 1)
            TAIL  = (0.92, 0.12, 0.10, 1)
            car_parts = [
                _box_mesh((-2.20, -0.95, 0.00), ( 2.20,  0.95, 0.18), GREY),
                _box_mesh((-2.10, -0.95, 0.18), ( 2.10,  0.95, 0.74), RED),
                _box_mesh(( 0.30, -0.88, 0.74), ( 2.05,  0.88, 1.06), RED),
                _box_mesh((-2.05, -0.88, 0.74), (-0.30,  0.88, 0.92), RED),
                _box_mesh((-0.90, -0.80, 0.92), ( 0.52,  0.80, 1.54), DARK),
                _box_mesh(( 2.05, -0.92, 0.18), ( 2.28,  0.92, 0.54), GREY),
                _box_mesh((-2.28, -0.92, 0.18), (-2.05,  0.92, 0.54), GREY),
                _box_mesh(( 2.05, -0.88, 0.54), ( 2.30, -0.42, 0.76), HEAD),
                _box_mesh(( 2.05,  0.42, 0.54), ( 2.30,  0.88, 0.76), HEAD),
                _box_mesh((-2.30, -0.88, 0.54), (-2.05, -0.42, 0.76), TAIL),
                _box_mesh((-2.30,  0.42, 0.54), (-2.05,  0.88, 0.76), TAIL),
                _box_mesh(( 0.87, -1.13, 0.00), ( 1.63, -0.95, 0.76), BLACK),
                _box_mesh(( 0.87,  0.95, 0.00), ( 1.63,  1.13, 0.76), BLACK),
                _box_mesh((-1.63, -1.13, 0.00), (-0.87, -0.95, 0.76), BLACK),
                _box_mesh((-1.63,  0.95, 0.00), (-0.87,  1.13, 0.76), BLACK),
            ]
            _mesh_to_node("car_body", np.concatenate(car_parts)).reparentTo(self.car_np)

        # A tall thin post is visible over buildings and from any bearing, which
        # matters because the marker has to be findable, not just pretty.
        self.target_nps = []
        for i in range(8):
            post = _mesh_to_node(
                f"target{i}",
                _box_mesh((-0.6, -0.6, 0.0), (0.6, 0.6, 11.0), (0.15, 0.95, 0.35, 1)))
            post.reparentTo(self.base.render)
            post.setLightOff()          # emissive-looking, reads as a beacon
            post.hide()
            self.target_nps.append(post)

    # ------------------------------------------------------------------
    # Minimap (2D top-down overlay)
    # ------------------------------------------------------------------

    # Minimap bounds in render2d coordinates (-1..1 on a square window).
    _MM_X0, _MM_X1 = 0.54, 0.98
    _MM_Z0, _MM_Z1 = 0.56, 0.98

    def _setup_minimap_frame(self):
        """Create the fixed skeleton: background, city card, waypoint markers."""
        self._mm_np = self.base.render2d.attachNewNode("minimap")

        pad = 0.028
        cm = CardMaker("mm_bg")
        cm.setFrame(self._MM_X0 - pad, self._MM_X1 + pad,
                    self._MM_Z0 - pad, self._MM_Z1 + pad)
        bg = self._mm_np.attachNewNode(cm.generate())
        bg.setColor(0.06, 0.07, 0.09, 1)

        # City texture quad (texture data filled on each build_scene).
        self._mm_city_tex = Texture("mm_city")
        cm2 = CardMaker("mm_city")
        cm2.setFrame(self._MM_X0, self._MM_X1, self._MM_Z0, self._MM_Z1)
        self._mm_city_np = self._mm_np.attachNewNode(cm2.generate())
        self._mm_city_np.setTexture(self._mm_city_tex)

        # Border drawn on top of the city quad.
        brd = LineSegs()
        brd.setColor(0.55, 0.58, 0.62, 1)
        brd.setThickness(1.5)
        brd.moveTo(self._MM_X0, 0, self._MM_Z0)
        brd.drawTo(self._MM_X1, 0, self._MM_Z0)
        brd.drawTo(self._MM_X1, 0, self._MM_Z1)
        brd.drawTo(self._MM_X0, 0, self._MM_Z1)
        brd.drawTo(self._MM_X0, 0, self._MM_Z0)
        self._mm_np.attachNewNode(brd.create()).setLightOff()

        # Waypoint squares — repositioned each frame, 8 slots for any task size.
        sq = 0.015
        for _ in range(8):
            cm3 = CardMaker("mm_wpt")
            cm3.setFrame(-sq, sq, -sq, sq)
            wpt = self._mm_np.attachNewNode(cm3.generate())
            wpt.setLightOff()
            wpt.hide()
            self._mm_wpt_nps.append(wpt)

    def _build_minimap_texture(self, city):
        """Upload a fresh top-down city image to the minimap texture card."""
        S = 2   # pixels per tile — enough to see street grid without blurring
        h, w = city.height * S, city.width * S
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        # Expand the grid (each tile → S×S pixels) with np.kron — fast, one op.
        expanded = np.kron(city.grid, np.ones((S, S), dtype=city.grid.dtype))
        arr[expanded == 0] = (38, 42, 46)    # road — dark blue-grey
        arr[expanded == 1] = (82, 78, 74)    # building — warm grey

        # Panda reads RAM images bottom-up; flip rows so row 0 (top of map)
        # appears at the top of the minimap quad.
        self._mm_city_tex.setup2dTexture(w, h, Texture.T_unsigned_byte,
                                         Texture.F_rgb8)
        self._mm_city_tex.setRamImage(arr[::-1].tobytes())
        self._mm_city_tex.setMagfilter(Texture.FTNearest)
        self._mm_city_tex.setMinfilter(Texture.FTNearest)
        self._mm_extent = city.extent          # (W*ts, H*ts)

    def _world_to_mm(self, wx, wy):
        """World (x, y) → render2d minimap (rx, rz).

        World +X maps to minimap right (+rx); world +Y maps to minimap down
        (-rz) because the world's +Y axis points down the screen and render2d's
        +Z points up.
        """
        ex, ey = self._mm_extent
        rx = self._MM_X0 + (wx / ex) * (self._MM_X1 - self._MM_X0)
        rz = self._MM_Z1 - (wy / ey) * (self._MM_Z1 - self._MM_Z0)
        return float(rx), float(rz)

    def _update_minimap(self, env):
        """Rebuild the car arrow + waypoint markers each frame."""
        if self._mm_car_np is not None:
            self._mm_car_np.removeNode()
            self._mm_car_np = None

        car = env.car
        cx, cz = self._world_to_mm(car.x, car.y)

        # Forward and left unit vectors in minimap/render2d space.
        # heading 0 = world +X (right on minimap), increasing CW.
        # World +Y points DOWN, so it maps to minimap -Z.
        fwd_x = np.cos(car.heading)
        fwd_z = -np.sin(car.heading)
        lft_x, lft_z = -fwd_z, fwd_x      # CCW 90° of forward

        sz = 0.020
        segs = LineSegs()

        # Proximity ring — fixed size around the car icon, colour only changes.
        # Green = clear, red = imminent collision.
        lidar = getattr(env, "lidar", None)
        if lidar is not None and hasattr(lidar, "last_distances"):
            min_dist = float(np.min(lidar.last_distances))
            t = np.clip(min_dist / lidar.max_range, 0.0, 1.0)
            ring_r = sz * 2.2   # fixed, just outside the car triangle
            segs.setThickness(2.0)
            segs.setColor(1.0 - t, t, 0.05, 0.85)
            N = 32
            for i in range(N + 1):
                a = 2.0 * np.pi * i / N
                if i == 0:
                    segs.moveTo(cx + np.cos(a) * ring_r, 0, cz + np.sin(a) * ring_r)
                else:
                    segs.drawTo(cx + np.cos(a) * ring_r, 0, cz + np.sin(a) * ring_r)

        # Car as a filled triangle: tip forward, base behind.
        tip_x = cx + fwd_x * sz
        tip_z = cz + fwd_z * sz
        bl_x = cx - fwd_x * sz * 0.6 + lft_x * sz * 0.6
        bl_z = cz - fwd_z * sz * 0.6 + lft_z * sz * 0.6
        br_x = cx - fwd_x * sz * 0.6 - lft_x * sz * 0.6
        br_z = cz - fwd_z * sz * 0.6 - lft_z * sz * 0.6

        segs.setThickness(2.5)
        segs.setColor(1.0, 0.38, 0.08, 1.0)
        segs.moveTo(bl_x, 0, bl_z)
        segs.drawTo(tip_x, 0, tip_z)
        segs.drawTo(br_x, 0, br_z)
        segs.drawTo(bl_x, 0, bl_z)

        self._mm_car_np = self._mm_np.attachNewNode(segs.create())
        self._mm_car_np.setLightOff()

        # Waypoint squares — just reposition, colour, and show/hide.
        pending = list(env.targets[env.target_idx:])
        for i, wpt_np in enumerate(self._mm_wpt_nps):
            if i < len(pending):
                tx, ty = pending[i]
                mx, mz = self._world_to_mm(tx, ty)
                wpt_np.setPos(mx, 0, mz)
                wpt_np.setColor(*((0.22, 0.95, 0.38, 1) if i == 0
                                  else (0.92, 0.76, 0.15, 1)))
                wpt_np.show()
            else:
                wpt_np.hide()

    # ------------------------------------------------------------------
    def build_scene(self, city):
        """(Re)build the city mesh. Called on every reset, so it must be quick."""
        if self._city_np is not None:
            self._city_np.removeNode()

        ts = city.tile_size
        grid = city.grid
        solid = grid == 1
        rows, cols = np.nonzero(solid)

        # Out-of-bounds neighbours count as solid: the outward faces of the
        # boundary wall can never be seen from inside the city, so skip them.
        def neighbour_solid(dr, dc):
            r, c = rows + dr, cols + dc
            inside = (r >= 0) & (r < city.height) & (c >= 0) & (c < city.width)
            out = np.ones(len(rows), dtype=bool)
            out[inside] = solid[r[inside], c[inside]]
            return out

        rnd = _hash01(rows, cols)
        hgt = (self.build_min + (self.build_max - self.build_min) * rnd).astype(np.float32)
        x0 = (cols * ts).astype(np.float32)
        y0 = (rows * ts).astype(np.float32)
        x1, y1 = x0 + ts, y0 + ts
        z0 = np.zeros(len(rows), dtype=np.float32)

        shade = (0.42 + 0.30 * _hash01(cols, rows)).astype(np.float32)
        base_col = np.stack([shade * 0.98, shade * 0.95, shade * 0.90,
                             np.ones_like(shade)], axis=1)

        def quad(pts, normal, cols_rgba, mask):
            corners = np.stack([np.stack(p, axis=1) for p in pts], axis=1)[mask]
            normals = np.tile(np.asarray(normal, dtype=np.float32), (mask.sum(), 1))
            return _quads_to_mesh(corners, normals, cols_rgba[mask])

        allv = np.ones(len(rows), dtype=bool)
        # Roofs get a flat lighter tone so blocks read as distinct volumes.
        roof_col = np.clip(base_col * 1.18, 0, 1)
        roof_col[:, 3] = 1.0

        parts = [
            quad([(x0, y0, hgt), (x1, y0, hgt), (x1, y1, hgt), (x0, y1, hgt)],
                 (0, 0, 1), roof_col, allv),
            quad([(x0, y0, z0), (x1, y0, z0), (x1, y0, hgt), (x0, y0, hgt)],
                 (0, -1, 0), base_col, ~neighbour_solid(-1, 0)),
            quad([(x1, y1, z0), (x0, y1, z0), (x0, y1, hgt), (x1, y1, hgt)],
                 (0, 1, 0), base_col, ~neighbour_solid(1, 0)),
            quad([(x0, y1, z0), (x0, y0, z0), (x0, y0, hgt), (x0, y1, hgt)],
                 (-1, 0, 0), base_col, ~neighbour_solid(0, -1)),
            quad([(x1, y0, z0), (x1, y1, z0), (x1, y1, hgt), (x1, y0, hgt)],
                 (1, 0, 0), base_col, ~neighbour_solid(0, 1)),
        ]

        # Road surface, emitted as one quad per tile rather than a single plane.
        # A featureless plane is nearly free to draw but gives a vision policy
        # almost no optical flow, which makes speed unobservable from pixels --
        # exactly the signal an image-based world model needs. Per-tile shading
        # puts visible edges on the ground for the cost of ~1k extra quads.
        grows, gcols = np.nonzero(~solid)
        gx0 = (gcols * ts).astype(np.float32)
        gy0 = (grows * ts).astype(np.float32)
        gx1, gy1 = gx0 + ts, gy0 + ts
        gz = np.zeros(len(grows), dtype=np.float32)
        tone = (0.20 + 0.075 * _hash01(grows * 7 + 1, gcols * 13 + 3)).astype(np.float32)
        road_col = np.stack([tone, tone * 1.04, tone * 1.10, np.ones_like(tone)], axis=1)
        parts.append(_quads_to_mesh(
            np.stack([np.stack(p, axis=1) for p in
                      [(gx0, gy0, gz), (gx1, gy0, gz), (gx1, gy1, gz), (gx0, gy1, gz)]],
                     axis=1),
            np.tile(np.array([(0, 0, 1)], dtype=np.float32), (len(grows), 1)),
            road_col))

        self._city_np = _mesh_to_node("city", np.concatenate(parts))
        self._city_np.reparentTo(self.base.render)
        self._cam_pos = None

        if self.show_minimap and self._mm_city_tex is not None:
            self._build_minimap_texture(city)

    # ------------------------------------------------------------------
    def _place_camera(self, env):
        car = env.car
        fx, fy = np.cos(car.heading), np.sin(car.heading)
        want = Vec3(car.x - fx * self.cam_dist,
                    car.y - fy * self.cam_dist,
                    self.cam_height)
        if self.smooth > 0.0 and self._cam_pos is not None:
            want = self._cam_pos * self.smooth + want * (1.0 - self.smooth)
        self._cam_pos = want
        self.base.camera.setPos(want)
        self.base.camera.lookAt(car.x + fx * self.look_ahead,
                                car.y + fy * self.look_ahead, 1.2)

    def _place_actors(self, env):
        car = env.car
        self.car_np.setPos(car.x, car.y, 0.0)
        self.car_np.setH(np.degrees(car.heading))

        # Only show waypoints still to be visited, brightest first.
        pending = env.targets[env.target_idx:]
        for i, np_ in enumerate(self.target_nps):
            if i < len(pending):
                tx, ty = pending[i]
                np_.setPos(tx, ty, 0.0)
                np_.setColor(*((0.15, 0.95, 0.35, 1) if i == 0
                               else (0.85, 0.75, 0.20, 1)))
                np_.show()
            else:
                np_.hide()

    def _draw_rays(self, env):
        """Proximity ring in the 3D scene.  Green = clear, red = close.

        Always clears first so a stale ring never stays frozen when the
        overlay is switched off between frames.
        """
        if self._rays_np is not None:
            self._rays_np.removeNode()
            self._rays_np = None
        lidar = getattr(env, "lidar", None)
        if not self.show_rays or lidar is None:
            return
        min_dist = float(np.min(lidar.last_distances))
        t = np.clip(min_dist / lidar.max_range, 0.0, 1.0)
        segs = LineSegs()
        segs.setThickness(2.5)
        segs.setColor(1.0 - t, t, 0.05, 0.9)
        radius = 3.0   # world metres, outside the car body
        N = 48
        cx, cy = env.car.x, env.car.y
        for i in range(N + 1):
            a = 2.0 * np.pi * i / N
            px, py = cx + np.cos(a) * radius, cy + np.sin(a) * radius
            if i == 0:
                segs.moveTo(px, py, 0.5)
            else:
                segs.drawTo(px, py, 0.5)
        self._rays_np = self.base.render.attachNewNode(segs.create())
        self._rays_np.setLightOff()

    def sync(self, env):
        """Point the camera and place the actors for the current env state.

        Split out from `capture` so an interactive window can let Panda drive its
        own render loop instead of us calling renderFrame by hand.
        """
        if self._city_np is None:
            self.build_scene(env.city)
        self._place_camera(env)
        self._place_actors(env)
        self._draw_rays(env)
        if self.show_minimap:
            self._update_minimap(env)

    def capture(self, env, size=None):
        """Render one frame and return it as an (H, W, 3) uint8 array."""
        if size is not None:
            want = (size, size) if isinstance(size, int) else tuple(size)
            if want != self.size:
                raise RuntimeError(
                    f"renderer was built for {self.size} but {want} was requested; "
                    "construct a PandaRenderer with the size you intend to use")
        self.sync(env)
        self.base.graphicsEngine.renderFrame()

        if self._tex is None:                     # onscreen: grab the framebuffer
            self._tex = Texture()
            self.base.win.addRenderTexture(
                self._tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            self.base.graphicsEngine.renderFrame()

        raw = self._tex.getRamImageAs("RGB")
        w, h = self._tex.getXSize(), self._tex.getYSize()
        img = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(h, w, 3)
        return img[::-1].copy()                   # Panda's RAM image is bottom-up

    def close(self):
        """Drop the scene. The ShowBase singleton outlives individual renderers."""
        if self._city_np is not None:
            self._city_np.removeNode()
            self._city_np = None
        if self._rays_np is not None:
            self._rays_np.removeNode()
            self._rays_np = None
        if self._mm_np is not None:
            self._mm_np.removeNode()
            self._mm_np = None
            self._mm_car_np = None
