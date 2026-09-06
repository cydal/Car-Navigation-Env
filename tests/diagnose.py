"""Trace a scripted episode step by step to find why the car crashes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import CityConfig, ProceduralCity, ROAD
from env.nav_env import CarNavEnv, EnvConfig
from env.car import CarParams
from baselines.scripted import GapFollower

np.set_printoptions(precision=2, suppress=True, linewidth=140)

# ----------------------------------------------------------------------
print("=" * 70)
print("A. GEOMETRY: does the car physically fit and turn in these streets?")
print("=" * 70)
p = CarParams()
for rw in (2, 3, 4):
    for ts in (4.0,):
        corridor = rw * ts
        print(f"road_width={rw} tiles @ {ts}m = {corridor:>5.1f}m corridor | "
              f"car {p.length}x{p.width}m | min turn radius {p.wheelbase/np.tan(p.max_steer):.2f}m")

# How much lateral room does a car need to swing through a 90-degree corner?
# Outer swept radius = sqrt((R + w/2)^2 + L_front^2) where L_front ~ length.
R = p.wheelbase / np.tan(p.max_steer)
outer = np.hypot(R + p.width / 2, p.length * 0.85)
inner = R - p.width / 2
print(f"\nturning: inner radius {inner:.2f}m, outer swept radius {outer:.2f}m")
print(f"  -> a 90 deg turn needs a corridor of at least ~{outer - inner:.2f}m of lateral room")

# Fraction of random poses on road that collide, per road width.
print(f"\n{'road_width':>11}{'corridor':>10}{'collide %':>11}{'free-pose tries':>17}")
for rw in (2, 3, 4):
    city = ProceduralCity(CityConfig(width=48, height=48, tile_size=4.0, road_width=rw), seed=7)
    n, coll = 3000, 0
    for _ in range(n):
        x, y = city.sample_road_point(jitter=1.0)
        h = city.rng.uniform(0, 2 * np.pi)
        if city.collides(x, y, h, p.length, p.width):
            coll += 1
    print(f"{rw:>11}{rw*4.0:>9.1f}m{coll/n*100:>10.1f}%{'':>17}")

# ----------------------------------------------------------------------
print()
print("=" * 70)
print("B. STEERING SIGN CONVENTION (open field, no walls)")
print("=" * 70)
from env.car import Car
car = Car(p)
car.reset(0, 0, 0, speed=8.0)
for _ in range(40):
    car.step(0.2, 0, 1.0, 0.05)
print(f"steer=+1 from heading 0: pos=({car.x:.1f},{car.y:.1f}) heading={np.degrees(car.heading):.1f} deg")
print(f"  -> +steer turns toward +y (screen-down / clockwise). y grew: {car.y > 0.5}")
car.reset(0, 0, 0, speed=8.0)
for _ in range(40):
    car.step(0.2, 0, -1.0, 0.05)
print(f"steer=-1 from heading 0: pos=({car.x:.1f},{car.y:.1f}) heading={np.degrees(car.heading):.1f} deg")

# Does the controller steer toward a target that is to the car's right/down?
drv = GapFollower()
n = 32
for label, bearing in (("target dead ahead", 0.0), ("target 45 deg right/down", np.radians(45)),
                       ("target 45 deg left/up", np.radians(-45)), ("target behind", np.pi)):
    obs = np.zeros(46, dtype=np.float32)
    obs[:n] = 1.0                       # nothing in range anywhere
    obs[n] = 8.0 / p.max_speed          # speed
    obs[n + 5] = 0.5                    # target distance
    obs[n + 6] = np.sin(bearing)
    obs[n + 7] = np.cos(bearing)
    a = drv.act(obs)
    print(f"  {label:<26} -> steer={a[2]:+.2f} throttle={(a[0]+1)/2:.2f} brake={(a[1]+1)/2:.2f}")

# ----------------------------------------------------------------------
print()
print("=" * 70)
print("C. SINGLE EPISODE TRACE")
print("=" * 70)
cfg = EnvConfig()
env = CarNavEnv(config=cfg, city_config=CityConfig(width=48, height=48, tile_size=4.0),
                obs_type="vector", seed=99)
obs, info = env.reset(seed=1000)
drv = GapFollower(n_beams=cfg.n_beams, lidar_range=cfg.lidar_range, max_speed=env.car.p.max_speed)
ts = env.city.tile_size
path = [(env.car.x, env.car.y)]

print(f"spawn: ({env.car.x:.1f},{env.car.y:.1f}) tile({int(env.car.x/ts)},{int(env.car.y/ts)}) "
      f"heading={np.degrees(env.car.heading):.0f} deg")
print(f"target[0]: {env.targets[0]} dist={info['dist_to_target']:.1f}m")
print()
print(f"{'step':>4}{'x':>7}{'y':>7}{'hdg':>6}{'spd':>6}{'steer':>7}{'bear':>7}"
      f"{'nose':>7}{'minR':>7}{'dist':>7}{'rew':>8}")
fwd = cfg.n_beams // 2
for i in range(400):
    a = drv.act(obs)
    scan = obs[:cfg.n_beams] * cfg.lidar_range
    nav = obs[cfg.n_beams + 5:]
    bearing = np.degrees(np.arctan2(nav[1], nav[2]))
    nose = scan[fwd - 4:fwd + 5].min()
    obs, r, te, tr, info = env.step(a)
    path.append((env.car.x, env.car.y))
    if i < 30 or i % 10 == 0 or te or tr:
        print(f"{i:>4}{env.car.x:>7.1f}{env.car.y:>7.1f}{np.degrees(env.car.heading):>6.0f}"
              f"{info['speed']:>6.1f}{a[2]:>7.2f}{bearing:>7.0f}{nose:>7.1f}{scan.min():>7.1f}"
              f"{info['dist_to_target']:>7.1f}{r:>8.2f}")
    if te or tr:
        print(f"\nEND: {info['reason']} after {info['step']} steps, reward={info['episode_reward']:.1f}")
        break

# ASCII map with the driven path
print()
print("D. TRAJECTORY (S=spawn, T=target, o=path, X=crash)")
rows = []
for r_ in range(env.city.height):
    rows.append(["." if env.city.grid[r_, c] == ROAD else "#" for c in range(env.city.width)])
for (px, py) in path:
    c, r_ = int(px / ts), int(py / ts)
    if 0 <= r_ < env.city.height and 0 <= c < env.city.width and rows[r_][c] in ".#":
        rows[r_][c] = "o"
for ti, (tx, ty) in enumerate(env.targets):
    c, r_ = int(tx / ts), int(ty / ts)
    if 0 <= r_ < env.city.height and 0 <= c < env.city.width:
        rows[r_][c] = str(ti + 1)
sx, sy = path[0]
rows[int(sy / ts)][int(sx / ts)] = "S"
ex, ey = path[-1]
rows[int(ey / ts)][int(ex / ts)] = "X"
print("\n".join("".join(r_) for r_ in rows))
