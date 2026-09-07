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
loadPrcFileData("", "framebuffer-multisample 1") # MSAA on
loadPrcFileData("", "multisamples 4")            # 4× samples

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

SKY = (0.52, 0.64, 0.82)

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
        self._tl_nps = []             # [(static_np, [r_np, y_np, g_np]), ...]
        self._tl_lamp_nps = []        # [[r_np, y_np, g_np], ...] — parallel to city.intersections
        # Vehicle meshes are loaded once per model and *instanced* per vehicle.
        # Reloading a .glb for each of ~26 vehicles on every reset would cost more
        # than the whole city mesh does.
        self._veh_protos = {}         # kind index -> prototype NodePath (detached)
        self._veh_root = None         # NodePath holding one instance per vehicle
        self._veh_nps = []
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
        # Cool blue-sky ambient — represents diffuse light from the open sky.
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.28, 0.32, 0.42, 1))
        render.setLight(render.attachNewNode(amb))

        # Warm golden key light — late-afternoon sun from the south-west.
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.92, 0.82, 0.60, 1))
        sun_np = render.attachNewNode(sun)
        sun_np.setHpr(-35, -55, 0)
        render.setLight(sun_np)

        # Weak cool fill from the opposite side — softens harsh shadow faces.
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.10, 0.13, 0.20, 1))
        fill_np = render.attachNewNode(fill)
        fill_np.setHpr(145, -20, 0)
        render.setLight(fill_np)

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
            _here, "..", "kenney_car-kit", "Models", "GLB format", "race.glb"))

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
    # Traffic lights
    # ------------------------------------------------------------------

    # Approaches, in Panda's Y-mirrored world. For each: the direction of travel,
    # the corner the mast stands on (signs of x, y offsets from the crossing
    # centre) and the axis of traffic it governs.
    #
    # The mast sits on the near-right corner of the approach and the arm reaches
    # left, out over the middle of the corridor, so the head hangs above the stop
    # line facing back down the approach -- the one place an approaching driver is
    # already looking. A single pole in one corner (the previous design) shows its
    # edge or its back from most approaches and cannot be read at all from two.
    _TL_APPROACHES = (
        # (travel dx, travel dy, corner x sign, corner y sign, axis)
        (0.0,  1.0,  1.0, -1.0, 'ns'),   # northbound: enters from -Y
        (0.0, -1.0, -1.0,  1.0, 'ns'),   # southbound: enters from +Y
        (1.0,  0.0, -1.0, -1.0, 'ew'),   # eastbound:  enters from -X
        (-1.0, 0.0,  1.0,  1.0, 'ew'),   # westbound:  enters from +X
    )

    # Signal-head geometry, metres.
    _MAST_H    = 5.6    # mast height; head hangs below, well clear of the car
    _MAST_W    = 0.22
    _ARM_Z     = 5.4    # underside of the horizontal arm
    _ARM_T     = 0.14   # arm thickness
    _HEAD_TOP  = 5.15
    _LAMP_H    = 0.30
    _LAMP_GAP  = 0.09
    _LAMP_HALF = 0.19   # lamp half-extent across the face
    _PLATE_D   = 0.07   # backplate thickness
    _PROUD     = 0.06   # how far the lens stands out in front of the backplate

    def _setup_traffic_lights(self, city):
        """Build one overhead signal head per approach at every signalised crossing.

        Four masts per crossing, one per approach, each on the near-right corner
        with the arm reaching out over the corridor so the head sits above the stop
        line facing oncoming traffic. Northbound/southbound heads show the N-S
        phase, eastbound/westbound the E-W phase, so the signal a driver is looking
        at is always the one that governs them -- there is nothing to disambiguate.

        Lens visibility: the lens box stands `_PROUD` metres in front of its
        backplate, so the driver sees its full front face. The previous design
        buried the lamps inside a housing box, which occluded every face of them.
        """
        for static_np, lamp_nps in self._tl_nps:
            static_np.removeNode()
            for ln in lamp_nps:
                ln.removeNode()
        self._tl_nps = []
        self._tl_lamp_nps = []

        signals = getattr(city, "signals", None) or []
        if not signals:
            return

        half = (city.cfg.road_width / 2.0) * city.tile_size
        mw = self._MAST_W / 2
        plate_h = 3 * self._LAMP_H + 2 * self._LAMP_GAP + 0.14
        plate_z0 = self._HEAD_TOP - plate_h
        MAST_COL  = (0.17, 0.17, 0.18, 1)
        PLATE_COL = (0.07, 0.07, 0.08, 1)
        DIM = [(0.20, 0.03, 0.03, 1), (0.20, 0.16, 0.03, 1), (0.03, 0.20, 0.03, 1)]

        for cx, cy in signals:
            px, py = cx, -cy          # crossing centre in Panda's Y-mirrored world
            static_parts = []
            heads = {'ns': [], 'ew': []}

            for tdx, tdy, sx, sy, axis in self._TL_APPROACHES:
                # Mast on the near-right corner of this approach.
                mx, my = px + sx * half, py + sy * half
                static_parts.append(_box_mesh(
                    (mx - mw, my - mw, 0.0), (mx + mw, my + mw, self._MAST_H), MAST_COL))

                # Head hangs over the corridor centreline, at the stop line: step
                # back from the crossing centre along the approach's travel axis.
                hx = px - tdx * half
                hy = py - tdy * half

                # Arm spans from the mast across to the head position.
                ax0, ax1 = sorted((mx, hx))
                ay0, ay1 = sorted((my, hy))
                static_parts.append(_box_mesh(
                    (ax0 - mw, ay0 - mw, self._ARM_Z),
                    (ax1 + mw, ay1 + mw, self._ARM_Z + self._ARM_T), MAST_COL))

                # Backplate is perpendicular to travel; lenses face -travel.
                lat = self._LAMP_HALF + 0.05           # plate half-width across face
                if axis == 'ns':
                    plate_lo = (hx - lat, hy - tdy * self._PLATE_D if tdy > 0 else hy, plate_z0)
                    plate_hi = (hx + lat, hy if tdy > 0 else hy + self._PLATE_D, self._HEAD_TOP)
                else:
                    plate_lo = (hx - tdx * self._PLATE_D if tdx > 0 else hx, hy - lat, plate_z0)
                    plate_hi = (hx if tdx > 0 else hx + self._PLATE_D, hy + lat, self._HEAD_TOP)
                static_parts.append(_box_mesh(plate_lo, plate_hi, PLATE_COL))

                # Three lenses, red on top, standing proud of the plate face.
                lamps = []
                for j in range(3):
                    lz0 = plate_z0 + 0.07 + (2 - j) * (self._LAMP_H + self._LAMP_GAP)
                    lz1 = lz0 + self._LAMP_H
                    h = self._LAMP_HALF
                    if axis == 'ns':
                        # faces -tdy: lens sits on the oncoming side of the plate
                        y_face = hy - tdy * self._PLATE_D
                        lo = (hx - h, min(y_face, y_face - tdy * self._PROUD), lz0)
                        hi = (hx + h, max(y_face, y_face - tdy * self._PROUD), lz1)
                    else:
                        x_face = hx - tdx * self._PLATE_D
                        lo = (min(x_face, x_face - tdx * self._PROUD), hy - h, lz0)
                        hi = (max(x_face, x_face - tdx * self._PROUD), hy + h, lz1)
                    ln = _mesh_to_node(f"tl_{axis}{j}", _box_mesh(lo, hi, DIM[j]))
                    ln.reparentTo(self.base.render)
                    ln.setLightOff()      # reads as self-illuminated, not shaded
                    lamps.append(ln)
                heads[axis].append(lamps)

            static_np = _mesh_to_node("tl_static", np.concatenate(static_parts))
            static_np.reparentTo(self.base.render)

            flat = [ln for hs in heads.values() for lamps in hs for ln in lamps]
            self._tl_nps.append((static_np, flat))
            self._tl_lamp_nps.append(heads)

    # Bright / dim colours: index 0 = red, 1 = yellow, 2 = green
    _TL_ON  = [(1.00, 0.06, 0.04, 1), (1.00, 0.88, 0.04, 1), (0.10, 1.00, 0.12, 1)]
    _TL_OFF = [(0.20, 0.03, 0.03, 1), (0.20, 0.16, 0.03, 1), (0.03, 0.20, 0.03, 1)]
    _TL_ORDER = ('red', 'yellow', 'green')

    def _update_traffic_lights(self, env):
        for tl, heads in zip(getattr(env, 'traffic_lights', []), self._tl_lamp_nps):
            for axis, head_list in heads.items():
                state = tl.state(axis)
                for lamps in head_list:
                    for j, name in enumerate(self._TL_ORDER):
                        lamps[j].setColor(
                            *(self._TL_ON[j] if state == name else self._TL_OFF[j]))

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------
    def _vehicle_proto(self, kind):
        """Load and cache one vehicle mesh, scaled so it matches its footprint.

        The physics footprint is authoritative: each mesh is measured after load
        and scaled to the length the simulation uses, rather than the simulation
        adopting whatever size the artist happened to model. Otherwise LIDAR would
        report one car and the picture would show another.
        """
        if kind in self._veh_protos:
            return self._veh_protos[kind]

        import os
        from env.traffic import VEHICLE_KINDS
        stem, length, width = VEHICLE_KINDS[kind]
        _here = os.path.dirname(os.path.abspath(__file__))
        glb = os.path.normpath(os.path.join(
            _here, "..", "kenney_car-kit", "Models", "GLB format", f"{stem}.glb"))

        holder = NodePath(f"veh_proto_{stem}")
        if os.path.exists(glb):
            model = self.base.loader.loadModel(glb)
            lo, hi = model.getTightBounds()
            raw = float(hi[1] - lo[1])            # Kenney models face +Y on load
            model.reparentTo(holder)
            model.setH(90)                        # nose from +Y to our +X
            if raw > 1e-3:
                holder.setScale(length / raw)
        else:                                     # keep working without the asset pack
            _mesh_to_node(f"veh_box_{stem}", _box_mesh(
                (-length / 2, -width / 2, 0.0), (length / 2, width / 2, 1.35),
                (0.55, 0.57, 0.62, 1))).reparentTo(holder)

        self._veh_protos[kind] = holder
        return holder

    def _setup_traffic(self, traffic):
        """Create one instanced node per vehicle for this episode."""
        if self._veh_root is not None:
            self._veh_root.removeNode()
            self._veh_root = None
        self._veh_nps = []
        if traffic is None or len(getattr(traffic, "kind", ())) == 0:
            return

        self._veh_root = self.base.render.attachNewNode("traffic")
        for i, kind in enumerate(traffic.kind):
            np_ = self._veh_root.attachNewNode(f"veh{i}")
            self._vehicle_proto(int(kind)).instanceTo(np_)
            self._veh_nps.append(np_)

    def _update_traffic(self, env):
        traffic = getattr(env, "traffic", None)
        if traffic is None:
            return
        for i, np_ in enumerate(self._veh_nps):
            if i >= len(traffic.x):
                break
            np_.setPos(float(traffic.x[i]), -float(traffic.y[i]), 0.0)  # Y-mirrored
            np_.setH(-np.degrees(float(traffic.heading[i])))

    # ------------------------------------------------------------------
    def build_scene(self, city, traffic=None):
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

        # Curbs: narrow raised concrete strips along every building-road boundary.
        # They use the same road-neighbour masks as the building walls above.
        cw_f = np.float32(0.28)   # curb width in metres
        ch_f = np.float32(0.14)   # curb height in metres
        ch_a = np.full(len(rows), ch_f, dtype=np.float32)
        curb_col = np.tile(np.array([0.60, 0.58, 0.55, 1.0], dtype=np.float32),
                           (len(rows), 1))
        cym, cyp = y0 - cw_f, y1 + cw_f  # south / north outer edges
        cxm, cxp = x0 - cw_f, x1 + cw_f  # west  / east  outer edges
        south = ~neighbour_solid(-1, 0)
        north = ~neighbour_solid(1,  0)
        west  = ~neighbour_solid(0, -1)
        east  = ~neighbour_solid(0,  1)
        # top faces (normal +Z) — follow the same CCW-in-physics-XY convention as roofs
        parts += [
            quad([(x0, cym, ch_a), (x1, cym, ch_a), (x1, y0, ch_a), (x0, y0, ch_a)], (0, 0, 1), curb_col, south),
            quad([(x0, y1, ch_a), (x1, y1, ch_a), (x1, cyp, ch_a), (x0, cyp, ch_a)], (0, 0, 1), curb_col, north),
            quad([(cxm, y0, ch_a), (x0, y0, ch_a), (x0, y1, ch_a), (cxm, y1, ch_a)], (0, 0, 1), curb_col, west),
            quad([(x1, y0, ch_a), (cxp, y0, ch_a), (cxp, y1, ch_a), (x1, y1, ch_a)], (0, 0, 1), curb_col, east),
        ]
        # outer faces (road-side vertical faces) — same winding as the matching wall normals
        parts += [
            quad([(x0, cym, z0), (x1, cym, z0), (x1, cym, ch_a), (x0, cym, ch_a)], (0, -1, 0), curb_col, south),
            quad([(x1, cyp, z0), (x0, cyp, z0), (x0, cyp, ch_a), (x1, cyp, ch_a)], (0,  1, 0), curb_col, north),
            quad([(cxm, y1, z0), (cxm, y0, z0), (cxm, y0, ch_a), (cxm, y1, ch_a)], (-1, 0, 0), curb_col, west),
            quad([(cxp, y0, z0), (cxp, y1, z0), (cxp, y1, ch_a), (cxp, y0, ch_a)], ( 1, 0, 0), curb_col, east),
        ]

        city_verts = np.concatenate(parts)
        # Physics Y increases "screen-down" (map convention); Panda3D is right-hand
        # Z-up where +Y is "north".  Bake the Y-mirror directly into vertex data
        # so all 3D placements can use (x, -physics_y) coordinates throughout.
        city_verts['v'][:, 1] *= -1
        city_verts['n'][:, 1] *= -1
        # Negating Y reverses each triangle's winding from CCW to CW (back-facing).
        # Restore front-face orientation by swapping v1↔v2 in every triangle.
        # Vertices are a flat list; stride-3 indexing hits exactly v1 and v2.
        tmp = city_verts[1::3].copy()
        city_verts[1::3] = city_verts[2::3]
        city_verts[2::3] = tmp
        self._city_np = _mesh_to_node("city", city_verts)
        self._city_np.reparentTo(self.base.render)
        self._cam_pos = None

        self._setup_traffic_lights(city)
        self._setup_traffic(traffic)

        if self.show_minimap and self._mm_city_tex is not None:
            self._build_minimap_texture(city)

    # ------------------------------------------------------------------
    def _place_camera(self, env):
        car = env.car
        fx, fy = np.cos(car.heading), np.sin(car.heading)
        # City/car/beacons all live in Panda's Y-mirrored world (-physics_y).
        # Camera behind car: car_panda − panda_fwd·dist, where panda_fwd=(cos h, −sin h).
        want = Vec3(car.x - fx * self.cam_dist,
                    -car.y + fy * self.cam_dist,
                    self.cam_height)
        if self.smooth > 0.0 and self._cam_pos is not None:
            want = self._cam_pos * self.smooth + want * (1.0 - self.smooth)
        self._cam_pos = want
        self.base.camera.setPos(want)
        self.base.camera.lookAt(car.x + fx * self.look_ahead,
                                -car.y - fy * self.look_ahead, 1.2)

    def _place_actors(self, env):
        car = env.car
        self.car_np.setPos(car.x, -car.y, 0.0)       # Y-mirrored world
        self.car_np.setH(-np.degrees(car.heading))

        # Only show waypoints still to be visited, brightest first.
        pending = env.targets[env.target_idx:]
        for i, np_ in enumerate(self.target_nps):
            if i < len(pending):
                tx, ty = pending[i]
                np_.setPos(tx, -ty, 0.0)              # Y-mirrored world
                np_.setColor(*((0.15, 0.95, 0.35, 1) if i == 0
                               else (0.85, 0.75, 0.20, 1)))
                np_.show()
            else:
                np_.hide()

    def _draw_rays(self, env):
        """Per-direction proximity ring in the 3D scene.

        Each segment is coloured by the LIDAR distance in that direction:
        green = clear, red = obstacle close.  Always clears first so a stale
        ring never stays frozen when the overlay is switched off.
        """
        if self._rays_np is not None:
            self._rays_np.removeNode()
            self._rays_np = None
        lidar = getattr(env, "lidar", None)
        if not self.show_rays or lidar is None:
            return

        dists = lidar.last_distances        # (n_beams,) raw metres
        n_beams = len(dists)
        max_r = float(lidar.max_range)
        heading = float(env.car.heading)
        # Ring is drawn in Panda's Y-mirrored world; car sits at (x, -y).
        cx, cy = env.car.x, -env.car.y

        segs = LineSegs()
        segs.setThickness(3.0)
        radius = 3.0   # metres, just outside the car body

        # Ring angle `a` is in Panda space.  Convert to physics angle (-a) to
        # look up the right LIDAR beam (beam k at physics heading+π+2πk/n).
        N = 64
        for i in range(N + 1):
            a = 2.0 * np.pi * i / N
            rel = (-a - heading - np.pi) % (2.0 * np.pi)
            k = int(round(rel * n_beams / (2.0 * np.pi))) % n_beams
            t = float(np.clip(dists[k] / max_r, 0.0, 1.0))
            segs.setColor(1.0 - t, t, 0.05, 0.9)
            px = cx + np.cos(a) * radius
            py = cy + np.sin(a) * radius
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
            self.build_scene(env.city, getattr(env, "traffic", None))
        self._place_camera(env)
        self._place_actors(env)
        self._draw_rays(env)
        self._update_traffic_lights(env)
        self._update_traffic(env)
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
        for static_np, lamp_nps in self._tl_nps:
            static_np.removeNode()
            for ln in lamp_nps:
                ln.removeNode()
        self._tl_nps = []
        self._tl_lamp_nps = []
        if self._veh_root is not None:
            self._veh_root.removeNode()
            self._veh_root = None
        self._veh_nps = []
        if self._mm_np is not None:
            self._mm_np.removeNode()
            self._mm_np = None
            self._mm_car_np = None
