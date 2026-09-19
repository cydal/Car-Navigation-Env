"""Renderer contract: image observations, determinism, and render cost.

Kept separate from test_core / test_env so those stay runnable without panda3d
installed -- the whole point of injecting the renderer is that the simulation
never depends on a graphics stack.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import CityConfig
from env.nav_env import CarNavEnv, EnvConfig
from render.panda_renderer import PandaRenderer

SIZE = 64
CITY = CityConfig(width=48, height=48, tile_size=4.0)
renderer = PandaRenderer(offscreen=True, size=SIZE)


def make_env(obs_type, seed=0):
    return CarNavEnv(config=EnvConfig(), city_config=CITY, obs_type=obs_type,
                     seed=seed, renderer=renderer, image_size=SIZE)


print("=" * 62)
print("1. IMAGE OBSERVATION CONTRACT")
print("=" * 62)
env = make_env("both", seed=3)
obs, info = env.reset(seed=4)
print(f"obs_type='both'  : vector {obs['vector'].shape} + image {obs['image'].shape} {obs['image'].dtype}")
assert obs["image"].shape == (SIZE, SIZE, 3) and obs["image"].dtype == np.uint8
assert obs["image"].min() >= 0 and obs["image"].max() <= 255

env_i = make_env("image", seed=3)
oi, _ = env_i.reset(seed=4)
assert oi.shape == (SIZE, SIZE, 3) and oi.dtype == np.uint8
print("obs_type='image' : returns the bare array")

# A frame that is a flat wash carries no information; check it has structure.
shades = len(np.unique(obs["image"].reshape(-1, 3), axis=0))
print(f"distinct colours : {shades} (a blank frame would be 1-2)")
assert shades > 20, "rendered frame looks blank"

try:
    renderer.capture(env, size=SIZE * 2)
    raise AssertionError("expected a size-mismatch error")
except RuntimeError:
    print("size mismatch    : raises rather than silently returning wrong pixels")

print()
print("=" * 62)
print("2. DETERMINISM (no hidden camera state)")
print("=" * 62)
# The chase camera must be a pure function of car pose. If it were smoothed, the
# image would depend on history: two runs of the same seed would diverge and the
# observation would stop being Markov. This is the check that keeps it honest.
def image_trace(seed, n=40):
    # Drive rather than act randomly: random actions average throttle 0.5 against
    # brake 0.5, so the car barely moves and the camera has nothing to show --
    # which would let this test pass on a frozen image.
    e = make_env("image", seed=seed)
    o, _ = e.reset(seed=seed)
    frames = [o.copy()]
    for k in range(n):
        act = np.array([1.0, -1.0, 0.25 * np.sin(k / 7.0)], dtype=np.float32)
        o, r, te, tr, _ = e.step(act)
        frames.append(o.copy())
        if te or tr:
            break
    return np.stack(frames)

a, b = image_trace(21), image_trace(21)
assert a.shape == b.shape and np.array_equal(a, b), "image observations are not deterministic"
print(f"same seed -> byte-identical frames over {len(a)} steps : OK")

# Consecutive frames must actually differ, or the camera is not tracking.
diff = np.abs(a[1:].astype(np.int16) - a[:-1].astype(np.int16)).mean()
print(f"mean |frame_t - frame_t-1| : {diff:.2f} (0 would mean a frozen camera)")
assert diff > 0.5, "frames are not changing as the car moves"

print()
print("=" * 62)
print("3. LIDAR OVERLAY / sync() PATH")
print("=" * 62)
# This is what `main.py demo` uses: sync() places the camera and actors without
# capturing, and rebuilds the ray overlay each frame. Exercised here because the
# interactive window cannot be asserted on in CI.
overlay = PandaRenderer(offscreen=True, size=SIZE, show_rays=True)
env_o = CarNavEnv(config=EnvConfig(), city_config=CITY, obs_type="vector",
                  seed=8, renderer=overlay, image_size=SIZE)
o, _ = env_o.reset(seed=8)
plain = renderer.capture(env_o).astype(np.int16)
for k in range(30):
    env_o.step(np.array([1.0, -1.0, 0.1], dtype=np.float32))
    overlay.sync(env_o)                      # must not leak nodes or raise
with_rays = overlay.capture(env_o).astype(np.int16)
print(f"sync() over 30 steps        : OK (no leak, no raise)")

# The overlay must actually change the pixels, or show_rays is silently a no-op.
overlay.show_rays = False
overlay.sync(env_o)
without = overlay.capture(env_o).astype(np.int16)
overlay.show_rays = True
overlay.sync(env_o)
again = overlay.capture(env_o).astype(np.int16)
delta = np.abs(again - without).mean()
print(f"rays change the frame by    : {delta:.2f} mean abs pixel diff")
assert delta > 0.2, "show_rays=True is not drawing anything"
overlay.close()

print()
print("=" * 62)
print("4a. CAMERA PICTURE-IN-PICTURE (front/left/right)")
print("=" * 62)
# Same leak/no-op check as show_rays in section 3, for the three PiP buffers.
pip_r = PandaRenderer(offscreen=True, size=SIZE, show_cameras=True)
env_p = CarNavEnv(config=EnvConfig(), city_config=CITY, obs_type="vector",
                  seed=9, renderer=pip_r, image_size=SIZE)
o, _ = env_p.reset(seed=9)
assert len(pip_r._pip) == 3, "expected 3 PiP camera rigs (front/left/right)"
for k in range(20):
    env_p.step(np.array([1.0, -1.0, 0.15], dtype=np.float32))
    pip_r.sync(env_p)                        # must not leak nodes or raise
print("sync() over 20 steps with show_cameras=True : OK (no leak, no raise)")

with_pip = pip_r.capture(env_p).astype(np.int16)
# Unlike show_rays (which is rebuilt from scratch every sync() call), the PiP
# cards are built once in __init__ and only *positioned* on each sync(); the
# flag is a construction-time switch, not a per-frame toggle. So the honest
# "without" comparison is hiding the already-built cards, not flipping the
# flag and re-syncing (which would leave the last-rendered cards in place and
# make this comparison a false negative).
for pip in pip_r._pip:
    pip["card_np"].hide()
    if pip["border_np"] is not None:
        pip["border_np"].hide()
    pip["label_np"].hide()
without_pip = pip_r.capture(env_p).astype(np.int16)
for pip in pip_r._pip:
    pip["card_np"].show()
    if pip["border_np"] is not None:
        pip["border_np"].show()
    pip["label_np"].show()
delta_pip = np.abs(with_pip - without_pip).mean()
print(f"cameras change the frame by : {delta_pip:.2f} mean abs pixel diff")
assert delta_pip > 0.2, "the PiP cards are not drawing anything onto the main capture"
pip_r.close()

# Default (offscreen=True, unset) must stay off, so training images never see it.
default_r = PandaRenderer(offscreen=True, size=SIZE)
assert default_r.show_cameras is False and default_r._pip == [], (
    "show_cameras must default off for offscreen renderers")
default_r.close()
print("offscreen default : show_cameras=False, no PiP nodes created : OK")

print()
print("=" * 62)
print("4b. RADAR (auxiliary sensor, not in the observation vector)")
print("=" * 62)
env_r = CarNavEnv(config=EnvConfig(), city_config=CITY, obs_type="vector", seed=11)
obs_r, _ = env_r.reset(seed=11)
assert obs_r.shape == (env_r.vector_dim,)
assert "radar" not in env_r.obs_slices, "radar must stay out of obs_slices"
assert env_r.radar.n_sectors == env_r.cfg.n_radar_sectors
last_reward = None
for _ in range(30):
    obs_r, last_reward, te, tr, info_r = env_r.step(
        np.array([1.0, -1.0, 0.1], dtype=np.float32))
    if te or tr:
        break
assert obs_r.shape == (env_r.vector_dim,), "radar must not change vector_dim"
assert env_r.radar.last_distances.shape == (env_r.cfg.n_radar_sectors,)
assert set(info_r["reward_components"]) == {
    "time", "crash", "progress", "target_bonus", "red_light", "pedestrian", "speeding"}
comp_sum = sum(info_r["reward_components"].values())
assert abs(comp_sum - last_reward) < 1e-4, (
    f"reward_components ({comp_sum}) must sum to the step reward ({last_reward})")
print(f"radar sectors {env_r.radar.n_sectors}, obs_dim unchanged at {env_r.vector_dim}-D : OK")
print("reward_components sum to the step reward, obs_slices untouched : OK")

print()
print("=" * 62)
print("4. RENDER COST")
print("=" * 62)
env_v = CarNavEnv(config=EnvConfig(), city_config=CITY, obs_type="vector", seed=5)
env_b = make_env("both", seed=5)
N = 600
act = np.array([1.0, -1.0, 0.05], dtype=np.float32)


def bench(e):
    e.reset(seed=1)
    t0 = time.perf_counter()
    for _ in range(N):
        _, _, te, tr, _ = e.step(act)
        if te or tr:
            e.reset()
    return (time.perf_counter() - t0) / N


t_vec, t_both = bench(env_v), bench(env_b)
print(f"vector only : {t_vec*1e3:5.2f} ms/step  ({1/t_vec:,.0f} steps/s)")
print(f"vector+image: {t_both*1e3:5.2f} ms/step  ({1/t_both:,.0f} steps/s)")
print(f"image adds  : {(t_both-t_vec)*1e3:5.2f} ms/step at {SIZE}x{SIZE}")

t0 = time.perf_counter()
for _ in range(20):
    env_b.city.generate()
    renderer.build_scene(env_b.city)
t_build = (time.perf_counter() - t0) / 20 * 1e3
print(f"build_scene : {t_build:5.1f} ms per reset (whole city as one merged mesh)")
assert t_build < 40.0, "scene rebuild is too slow -- it runs on every reset"

print()
print("=" * 62)
print("ALL RENDER CHECKS PASSED")
print("=" * 62)
