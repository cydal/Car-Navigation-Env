"""Regenerate the README figures.

    python tools/figures.py            # all of them, into docs/
    python tools/figures.py --only task

Frames are chosen by **content predicates read from the observation vector**, not
by stepping to a hard-coded step number. A fixed step count picks whatever the
agent happened to be doing and quietly rots the moment any random stream shifts;
a predicate like "the nearest signal is red, within 25 m, and ahead of us" keeps
producing a figure of the thing it claims to show. The predicates use
`env.obs_slices` and the public observation, so a figure caption that says "this
is what the agent sees" is literally true, and a layout change breaks the
generator rather than silently mislabelling a picture.

One Panda3D `ShowBase` per process, and its framebuffer size is fixed when it is
built, so the 960x540 scene shots and the 64x64 agent-view grid cannot be made in
the same process. This script re-invokes itself for the second size.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import carnav
from baselines.scripted import GapFollower

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
WIDE = (960, 540)
SEEDS = range(5000, 5060)

# Metres per unit in the normalised observation, per block. These are the env's
# own constants; the whole point of reading them here is that a figure built on
# the wrong scale would still look plausible.
TL_RANGE = 60.0
TRAFFIC_RANGE = 60.0


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # most Linux
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _font(size=13):
    """A scalable font if the platform has one, else PIL's bitmap default.

    PIL's default is a fixed ~11px bitmap, which at these image sizes rendered the
    panel labels as an unreadable smudge. Falling back to it rather than requiring
    a font keeps this script working on a machine with no system fonts -- the
    labels just get small again, which is a cosmetic loss, not a failure.
    """
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _label(d, xy, text, fill, font, outline=(0, 0, 0)):
    """Text with a 1px outline on all eight sides, so it survives any background."""
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                d.text((x + dx, y + dy), text, fill=outline, font=font)
    d.text((x, y), text, fill=fill, font=font)


# ----------------------------------------------------------------------
# Frame selection
# ----------------------------------------------------------------------
def _unpack(obs, sl):
    """The three blocks the predicates need, in metres and unit vectors."""
    tl = obs[sl["traffic_light"]]
    tr = obs[sl["traffic"]].reshape(-1, 5) if sl["traffic"].stop > sl["traffic"].start \
        else np.zeros((0, 5))
    nav = obs[sl["nav"]].reshape(-1, 3)
    return tl, tr, nav


def score_hero(obs, sl, info):
    """A clean establishing shot: waypoint pillar dead ahead, car actually moving.

    Scored rather than thresholded so the best frame in 60 episodes wins instead
    of the first merely-acceptable one.
    """
    tl, tr, nav = _unpack(obs, sl)
    d, s, c = nav[0]
    if c < 0.93 or info["speed"] < 6.0:
        return -1.0
    dist = d * 150.0
    if not 25.0 < dist < 70.0:
        return -1.0
    return c * info["speed"] * (1.0 - abs(dist - 45.0) / 45.0)


def score_signal(obs, sl, info):
    """Nearest signal red, close, and ahead -- the stop-line decision."""
    tl, tr, nav = _unpack(obs, sl)
    if tl.size == 0:
        return -1.0
    dist, s, c, red, yellow, green = tl[0] * TL_RANGE, tl[1], tl[2], tl[3], tl[4], tl[5]
    if red < 0.5 or c < 0.8 or not 8.0 < dist < 30.0:
        return -1.0
    return c * (30.0 - dist)


def score_traffic(obs, sl, info):
    """As many moving vehicles as possible, close, in front."""
    tl, tr, nav = _unpack(obs, sl)
    if tr.shape[0] == 0:
        return -1.0
    dist, cos = tr[:, 0] * TRAFFIC_RANGE, tr[:, 2]
    near = (dist < 35.0) & (cos > 0.2)
    if near.sum() < 1:
        return -1.0
    # Count dominates, closeness breaks ties: two cars in frame beats one closer one.
    return near.sum() * 100.0 + (35.0 - dist[near].min())


def best_frame(env, renderer, scorer, seeds):
    """Drive the scripted baseline and keep the single highest-scoring frame.

    The renderer is passed in rather than read off the env: the env holds it as a
    private `_renderer` precisely because the simulation does not depend on it,
    and a docs tool should not be the thing that makes that private name load-bearing.

    Returns (image_or_None, record). The image is captured lazily -- only when a
    frame beats the incumbent -- because a 960x540 grab costs far more than a step
    and the overwhelming majority of steps are not interesting.
    """
    sl = env.obs_slices
    drv = GapFollower.for_env(env)
    best, best_img, best_rec = -1.0, None, None
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        drv.reset()
        while True:
            obs, _, te, tr_, info = env.step(drv.act(obs))
            s = scorer(obs, sl, info)
            if s > best:
                best = s
                best_rec = dict(seed=seed, step=info["step"], score=round(float(s), 2),
                                speed=round(info["speed"], 1))
                renderer.sync(env)
                best_img = renderer.capture(env).copy()
            if te or tr_:
                break
    return best_img, best_rec


# ----------------------------------------------------------------------
# Scene shots (960x540, one ShowBase)
# ----------------------------------------------------------------------
def scene_shots():
    from render.panda_renderer import PandaRenderer

    # Overlays are forced on: they default off for offscreen buffers so that a
    # debug ring can never leak into an image *observation*, but these are
    # documentation, and the ring is the clearest picture of what LIDAR gives.
    r = PandaRenderer(offscreen=True, size=WIDE, show_rays=True, show_minimap=True)
    env = carnav.make(width=48, height=48, renderer=r, seed=0)

    for name, scorer in (("hero", score_hero), ("signal", score_signal),
                         ("traffic", score_traffic)):
        img, rec = best_frame(env, r, scorer, SEEDS)
        if img is None:
            print(f"  {name}: no frame matched the predicate -- skipped")
            continue
        Image.fromarray(img).save(os.path.join(DOCS, f"{name}.png"))
        print(f"  docs/{name}.png  {rec}")


# ----------------------------------------------------------------------
# Agent view (64x64, a second ShowBase, hence a second process)
# ----------------------------------------------------------------------
def agent_view(px=64, scale=4, cols=4, rows=2):
    from render.panda_renderer import PandaRenderer

    r = PandaRenderer(offscreen=True, size=px)
    env = carnav.make(width=48, height=48, obs_type="both", renderer=r,
                      image_size=px, seed=0)
    drv = GapFollower.for_env(env)
    sl = env.obs_slices

    # Spread the panels across distinct situations rather than sampling one
    # episode every N steps: a grid of eight near-identical corridors says
    # nothing about what the policy has to tell apart.
    want = [("open road", score_hero), ("at a signal", score_signal),
            ("traffic ahead", score_traffic)]
    panels, labels = [], []
    for label, scorer in want:
        best, img = -1.0, None
        for seed in list(SEEDS)[:20]:
            obs, info = env.reset(seed=seed)
            drv.reset()
            while True:
                obs, _, te, tr_, info = env.step(drv.act(obs["vector"]))
                s = scorer(obs["vector"], sl, info)
                if s > best:
                    best, img = s, obs["image"].copy()
                if te or tr_:
                    break
        if img is not None:
            panels.append(img)
            labels.append(label)

    # Fill the rest of the grid with an ordinary drive, so the figure shows the
    # variety the encoder actually sees.
    obs, info = env.reset(seed=5007)
    drv.reset()
    k = 0
    while len(panels) < cols * rows:
        obs, _, te, tr_, info = env.step(drv.act(obs["vector"]))
        k += 1
        if k % 55 == 0:
            panels.append(obs["image"].copy())
            labels.append("")
        if te or tr_:
            obs, info = env.reset()
            drv.reset()

    grid = np.concatenate([np.concatenate(panels[i * cols:(i + 1) * cols], axis=1)
                           for i in range(rows)], axis=0)
    img = Image.fromarray(grid).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    font = _font(15)
    # Panel borders, so eight dark frames do not read as one image.
    for i in range(1, cols):
        d.line([(i * px * scale, 0), (i * px * scale, img.height)], fill=(255, 255, 255))
    for i in range(1, rows):
        d.line([(0, i * px * scale), (img.width, i * px * scale)], fill=(255, 255, 255))
    for i, label in enumerate(labels[:cols * rows]):
        if label:
            _label(d, ((i % cols) * px * scale + 6, (i // cols) * px * scale + 5),
                   label, (255, 235, 60), font)
    img.save(os.path.join(DOCS, "agent_view.png"))
    print(f"  docs/agent_view.png  {cols}x{rows} panels at {px}px, {scale}x nearest")


# ----------------------------------------------------------------------
# Top-down task diagram (no Panda3D at all)
# ----------------------------------------------------------------------
def task_diagram(px_per_tile=15):
    """A plan view of one completed episode: what the 73 numbers are describing.

    Deliberately drawn from the same public state the observation is built from
    (`env.targets`, `env.traffic.*`, `env.traffic_lights`, `lidar.last_distances`)
    rather than from the renderer, so it stays truthful with panda3d uninstalled --
    and so it can show the LIDAR fan and the waypoint capture radii, neither of
    which the 3D view can convey.

    A *successful* episode is searched for rather than a fixed seed being trusted
    to still be one. The baseline completes ~31% of routes, so a hard-coded seed
    is a coin flip that the figure still shows a finished route after any change
    to the random streams -- and a figure captioned "the task" that shows a crash
    is worse than no figure.
    """
    env = carnav.make(width=48, height=48, seed=0)
    drv = GapFollower.for_env(env)
    trail, info = None, None
    for seed in SEEDS:
        obs, info = env.reset(seed=seed)
        drv.reset()
        path = [(env.car.x, env.car.y)]
        while True:
            obs, _, te, tr_, info = env.step(drv.act(obs))
            path.append((info["x"], info["y"]))
            if te or tr_:
                break
        # Longest successful route wins: it crosses more of the city, so the
        # figure shows the street network rather than one corner of it.
        if info["reason"] == "success" and (trail is None or len(path) > len(trail)):
            trail, chosen = path, seed
    if trail is None:
        raise RuntimeError("no successful episode in the seed range")
    obs, info = env.reset(seed=chosen)          # replay it, to leave the env posed
    drv.reset()
    for _ in range(len(trail) - 1):
        obs, _, te, tr_, info = env.step(drv.act(obs))

    city = env.city
    k = px_per_tile / city.tile_size            # metres -> pixels
    W = int(city.width * px_per_tile)
    H = int(city.height * px_per_tile)
    LEGEND = 44

    def P(x, y):
        return x * k, y * k

    img = Image.new("RGB", (W, H + LEGEND), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = _font(13)
    d.rectangle([0, 0, W, H], fill=(238, 238, 235))          # road surface

    # Buildings as filled tiles, coalesced into horizontal runs per row so a
    # 48x48 grid is a few hundred rectangles rather than 2,304.
    solid = city.grid != 0
    for row in range(city.height):
        runs = np.flatnonzero(np.diff(np.r_[0, solid[row].astype(np.int8), 0]))
        for a, b in zip(runs[::2], runs[1::2]):
            d.rectangle([a * px_per_tile, row * px_per_tile,
                         b * px_per_tile - 1, (row + 1) * px_per_tile - 1],
                        fill=(122, 132, 148))

    # Parked cars, then moving ones, as oriented boxes.
    t = env.traffic
    for i in range(len(t.x)):
        L, Wd = t.length[i] * k / 2, t.width[i] * k / 2
        ca, sa = np.cos(t.heading[i]), np.sin(t.heading[i])
        cx, cy = P(t.x[i], t.y[i])
        pts = [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
               for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
        d.polygon(pts, fill=(120, 124, 132) if t.parked[i] else (215, 120, 40))

    # Signals: the phase is live state, so colour each one by what it is showing.
    for tlight in env.traffic_lights:
        cx, cy = P(tlight.x, tlight.y)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(60, 60, 60))
        for axis, off in (("ns", -2.5), ("ew", 2.5)):
            col = {"red": (225, 45, 45), "yellow": (235, 195, 40),
                   "green": (60, 200, 90)}[tlight.state(axis)]
            d.ellipse([cx - 1.6 + off, cy - 1.6, cx + 1.6 + off, cy + 1.6], fill=col)

    # The LIDAR fan at the final pose, drawn *before* the annotations: it is
    # background context, and on top it overdrew the waypoint numbers.
    # `last_distances` is the metre-valued scan the final observation was built
    # from and `offsets` are its beam angles, so this draws the actual 32 numbers
    # in the obs rather than a re-derivation that could disagree about beam 0.
    ranges = env.lidar.last_distances
    angles = env.car.heading + env.lidar.offsets
    cx, cy = P(env.car.x, env.car.y)
    for rng, ang in zip(ranges, angles):
        d.line([cx, cy, cx + rng * k * np.cos(ang), cy + rng * k * np.sin(ang)],
               fill=(250, 170, 60), width=1)

    # The driven trajectory, under the waypoints so the circles stay readable.
    d.line([P(x, y) for x, y in trail], fill=(25, 25, 25), width=3)
    sx, sy = P(*trail[0])
    d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(255, 255, 255),
              outline=(25, 25, 25), width=2)
    _label(d, (sx + 9, sy - 7), "start", (25, 25, 25), font, outline=(255, 255, 255))

    # Waypoints, in order, with the capture radius drawn to scale -- the radius is
    # the point: 6 m against 40 m hops is why the bonuses cannot chain.
    for i, (tx, ty) in enumerate(env.targets):
        cx, cy = P(tx, ty)
        rr = env.cfg.target_radius * k
        col = (40, 90, 210)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=col, width=3)
        _label(d, (cx + rr + 4, cy - 8), str(i + 1), col, font, outline=(255, 255, 255))

    # The ego last, on top of its own scan.
    L, Wd = env.car.p.length * k / 2, env.car.p.width * k / 2
    ca, sa = np.cos(env.car.heading), np.sin(env.car.heading)
    d.polygon([(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
               for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))],
              fill=(210, 40, 40))

    # Legend, wrapped onto two rows -- one row overflowed the 48-tile width and
    # silently clipped the last entry.
    row = [("ego", (210, 40, 40)), ("LIDAR, 32 beams", (250, 170, 60)),
           ("trajectory", (25, 25, 25)), ("waypoint + 6 m radius", (40, 90, 210))], \
          [("moving traffic", (215, 120, 40)), ("parked car", (120, 124, 132)),
           ("traffic signal", (225, 45, 45)), ("building", (122, 132, 148))]
    for r, entries in enumerate(row):
        x, y = 6, H + 6 + r * 18
        for label, col in entries:
            d.rectangle([x, y + 1, x + 10, y + 11], fill=col, outline=(90, 90, 90))
            d.text((x + 15, y), label, fill=(20, 20, 20), font=font)
            x += 28 + int(d.textlength(label, font=font))

    img.save(os.path.join(DOCS, "task.png"))
    print(f"  docs/task.png  seed {chosen}, {len(trail)} steps to "
          f"{info['reason']}, {info['targets_reached']}/{info['n_targets']} "
          f"waypoints, {W}x{H + LEGEND}")


# ----------------------------------------------------------------------
FIGURES = {"scenes": scene_shots, "agent": agent_view, "task": task_diagram}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(FIGURES), help="generate one figure set")
    args = ap.parse_args()
    os.makedirs(DOCS, exist_ok=True)

    if args.only:
        FIGURES[args.only]()
        return

    print("task diagram (no renderer)")
    task_diagram()
    print("scene shots (960x540 ShowBase)")
    scene_shots()
    # A fresh process: this ShowBase is a singleton with a fixed buffer size.
    print("agent view (64x64 ShowBase, separate process)")
    sub = subprocess.run([sys.executable, os.path.abspath(__file__), "--only", "agent"],
                         cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write("".join(l for l in sub.stdout.splitlines(True)
                            if "docs/" in l or "Traceback" in l))
    if sub.returncode:
        sys.stderr.write(sub.stderr[-2000:])
        sys.exit(sub.returncode)


if __name__ == "__main__":
    main()
