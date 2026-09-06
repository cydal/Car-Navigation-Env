"""Inspect the final moments before each crash to classify the failure mode."""

import sys, os
from collections import deque, Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import CityConfig, ROAD
from env.nav_env import CarNavEnv, EnvConfig
from baselines.scripted import GapFollower

cfg = EnvConfig()
env = CarNavEnv(config=cfg, city_config=CityConfig(width=48, height=48, tile_size=4.0),
                obs_type="vector", seed=99)
drv = GapFollower(n_beams=cfg.n_beams, lidar_range=cfg.lidar_range,
                  max_speed=env.car.p.max_speed,
                  car_length=env.car.p.length, car_width=env.car.p.width)
ts = env.city.tile_size
p = env.car.p

SHOW = 3
shown = 0
kinds = Counter()

for ep in range(30):
    obs, info = env.reset(seed=1000 + ep)
    drv.reset()
    hist = deque(maxlen=14)
    done = False
    while not done:
        a = drv.act(obs)
        d = dict(drv.last)
        obs, r, te, tr, info = env.step(a)
        d.update(x=env.car.x, y=env.car.y, hdg=np.degrees(env.car.heading),
                 steer_actual=np.degrees(env.car.steer_angle), yaw=np.degrees(env.car.yaw_rate))
        hist.append(d)
        done = te or tr

    if info["reason"] != "crash":
        kinds[info["reason"]] += 1
        continue

    # Classify: was the car turning hard at the moment of impact, or going straight?
    last = hist[-1]
    turning = abs(last["steer_actual"]) > 12.0
    fast = last["speed"] > 6.0
    kinds["crash/" + ("cornering" if turning else "straight") + ("-fast" if fast else "-slow")] += 1

    if shown < SHOW:
        shown += 1
        print("=" * 78)
        print(f"CRASH  ep={ep}  after {info['step']} steps   "
              f"(turning={turning}, speed={last['speed']:.1f} m/s)")
        print("=" * 78)
        print(f"{'x':>7}{'y':>7}{'hdg':>6}{'spd':>6}{'tgtV':>6}{'steer':>7}{'wheel':>7}"
              f"{'theta':>7}{'bear':>7}{'chosen_clr':>11}{'need':>7}{'nOK':>5}{'fb':>4}")
        for h in hist:
            print(f"{h['x']:>7.1f}{h['y']:>7.1f}{h['hdg']:>6.0f}{h['speed']:>6.1f}"
                  f"{h['target_speed']:>6.1f}{h['steer']:>7.2f}{h['steer_actual']:>7.1f}"
                  f"{h['theta']:>7.0f}{h['bearing']:>7.0f}{h['chosen_clear']:>11.1f}"
                  f"{h['need']:>7.1f}{h['n_ok']:>5}{'Y' if h['fell_back'] else '.':>4}")

        # Zoomed map around the crash, with the car's footprint corners marked.
        cx, cy = last["x"], last["y"]
        c0, r0 = int(cx / ts), int(cy / ts)
        print(f"\nlocal map around tile ({r0},{c0})   car heading {last['hdg']:.0f} deg")
        for rr in range(max(0, r0 - 4), min(env.city.height, r0 + 5)):
            row = ""
            for cc in range(max(0, c0 - 6), min(env.city.width, c0 + 7)):
                ch = "." if env.city.grid[rr, cc] == ROAD else "#"
                if (rr, cc) == (r0, c0):
                    ch = "C"
                row += ch
            print("   " + row)

        # Which part of the car is inside a building?
        hl, hw = p.length / 2, p.width / 2
        h = np.radians(last["hdg"])
        names = ["front-left", "front-right", "rear-left", "rear-right"]
        local = [(hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)]
        inside = []
        for nm, (lx, ly) in zip(names, local):
            px = cx + lx * np.cos(h) - ly * np.sin(h)
            py = cy + lx * np.sin(h) + ly * np.cos(h)
            if not env.city.is_road(px, py):
                inside.append(nm)
        centre_ok = env.city.is_road(cx, cy)
        print(f"\n   centre on road: {centre_ok}   corners inside building: {inside or 'none'}")
        print()

print("=" * 78)
print("FAILURE MODE TALLY (30 episodes)")
print("=" * 78)
for k, v in kinds.most_common():
    print(f"  {k:<28} {v}")
