"""
Shared cockpit-dashboard palette and 2D panel builders.

Centralises the colors/panel geometry that used to be scattered as inline
literals across `panda_renderer.py` (minimap, LIDAR ring, traffic lights), and
gives `hud.py` the same building blocks so every new overlay looks like one
coherent dashboard instead of a pile of ad-hoc widgets.

Everything here operates in Panda3D's `render2d` normalized space (-1..1 on a
square window) and uses only `CardMaker`/`LineSegs`, the same primitives the
existing minimap already relies on -- no DirectGUI, no extra dependencies.
"""

from panda3d.core import CardMaker, LineSegs, Vec4

# --- palette ---------------------------------------------------------------
# Dark cockpit theme: near-black panels, cool text, warm/teal accents.
BG          = (0.055, 0.062, 0.075, 0.88)   # panel background
BG_SOLID    = (0.055, 0.062, 0.075, 1.00)
BORDER      = (0.30, 0.33, 0.37, 1.00)
BORDER_DIM  = (0.18, 0.20, 0.23, 1.00)
TEXT        = (0.92, 0.94, 0.96, 1.00)
TEXT_DIM    = (0.60, 0.64, 0.68, 1.00)
ACCENT      = (0.20, 0.75, 0.95, 1.00)      # teal -- headings / active state
WARN        = (0.95, 0.75, 0.15, 1.00)      # yellow -- caution
GOOD        = (0.20, 0.85, 0.35, 1.00)      # green -- clear / good
BAD         = (0.92, 0.20, 0.18, 1.00)      # red -- danger / crash

# Traffic-light-style bright/dim triples, index 0=red, 1=yellow, 2=green.
TL_ON  = ((1.00, 0.06, 0.04, 1), (1.00, 0.88, 0.04, 1), (0.10, 1.00, 0.12, 1))
TL_OFF = ((0.20, 0.03, 0.03, 1), (0.20, 0.16, 0.03, 1), (0.03, 0.20, 0.03, 1))

# LIDAR/radar range-gradient endpoints (near = danger, far = clear).
RANGE_NEAR = (0.95, 0.18, 0.12, 1.0)
RANGE_FAR  = (0.15, 0.85, 0.35, 1.0)


def lerp_color(near, far, t):
    """Blend two RGBA tuples; t=0 -> near, t=1 -> far."""
    t = max(0.0, min(1.0, t))
    return tuple(n + (f - n) * t for n, f in zip(near, far))


def panel_card(parent, x0, x1, z0, z1, bg_color=BG, border_color=BORDER,
               border_thickness=1.5, name="panel"):
    """Background quad + optional border, the same idiom as the old minimap frame.

    Returns (bg_nodepath, border_nodepath_or_None).
    """
    cm = CardMaker(f"{name}_bg")
    cm.setFrame(x0, x1, z0, z1)
    bg = parent.attachNewNode(cm.generate())
    bg.setColor(*bg_color)
    bg.setTransparency(True)

    border_np = None
    if border_color is not None:
        seg = LineSegs()
        seg.setColor(*border_color)
        seg.setThickness(border_thickness)
        seg.moveTo(x0, 0, z0)
        seg.drawTo(x1, 0, z0)
        seg.drawTo(x1, 0, z1)
        seg.drawTo(x0, 0, z1)
        seg.drawTo(x0, 0, z0)
        border_np = parent.attachNewNode(seg.create())
        border_np.setLightOff()
    return bg, border_np


def bar_meter(parent, x0, x1, z, height, frac, low_color=GOOD, high_color=BAD,
              bg_color=BORDER_DIM, name="bar"):
    """Horizontal filled bar: dim full-width track + colored frac-width fill.

    `frac` in [0, 1]; color interpolates low_color -> high_color as frac grows,
    so callers can drive both "how much" and "how urgent" from one number
    (e.g. LIDAR closeness, decision confidence, speed-vs-limit).

    Returns (track_nodepath, fill_nodepath) -- the caller keeps the fill
    NodePath to resize it on later updates instead of rebuilding the meter.
    """
    frac = max(0.0, min(1.0, frac))
    cm_bg = CardMaker(f"{name}_track")
    cm_bg.setFrame(x0, x1, z, z + height)
    track = parent.attachNewNode(cm_bg.generate())
    track.setColor(*bg_color)

    fill = None
    if frac > 0.0:
        cm_fg = CardMaker(f"{name}_fill")
        cm_fg.setFrame(x0, x0 + (x1 - x0) * frac, z, z + height)
        fill = parent.attachNewNode(cm_fg.generate())
        fill.setColor(*lerp_color(low_color, high_color, frac))
    return track, fill


def set_bar_frac(parent, old_fill, x0, x1, z, height, frac, low_color=GOOD,
                  high_color=BAD, name="bar"):
    """Rebuild a bar_meter's fill quad in place (Panda cards aren't resizable)."""
    if old_fill is not None:
        old_fill.removeNode()
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.0:
        return None
    cm_fg = CardMaker(f"{name}_fill")
    cm_fg.setFrame(x0, x0 + (x1 - x0) * frac, z, z + height)
    fill = parent.attachNewNode(cm_fg.generate())
    fill.setColor(*lerp_color(low_color, high_color, frac))
    return fill
