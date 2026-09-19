"""
Entry point for the 3D car navigation environment.

    python main.py spec          # print the observation/action contract
    python main.py serve         # live browser viewer at http://localhost:8765
    python main.py shots         # save a grid of agent-view frames to a PNG
    python main.py bench         # throughput, vector vs image

`serve` is the one to reach for when something looks wrong: seeing the LIDAR fan
and camera detections against the geometry they are measuring explains
observation bugs far faster than reading numbers does. There is no windowed
Panda3D viewer any more -- `render/panda_renderer.py` only ever produces the
64x64 image observation now; `serve`'s browser viewer is the one human-facing
view, and it works against a headless training box just as well as a laptop.
"""

import argparse
import time

import numpy as np

from env.world import CityConfig
from env.nav_env import CarNavEnv, EnvConfig
from agents import load_agent


def build_env(args, obs_type, renderer=None, image_size=64):
    city = CityConfig(width=args.map, height=args.map)
    cfg = EnvConfig(n_targets=args.targets, n_beams=args.beams,
                    pedestrians=args.pedestrians, speed_signs=args.signs)
    return CarNavEnv(config=cfg, city_config=city, obs_type=obs_type,
                     seed=args.seed, renderer=renderer, image_size=image_size)


def make_driver(env, agent_spec="scripted"):
    return load_agent(agent_spec, env)


# ----------------------------------------------------------------------
def cmd_spec(args):
    env = build_env(args, "vector")
    c = env.cfg
    p = env.car.p
    # Widths come from env.obs_slices rather than being recomputed here, so this
    # listing cannot drift from the observation it describes.
    desc = {
        "lidar": f"360 deg, {c.lidar_range:.0f} m range, normalised to [0,1]",
        "dynamics": "speed, yaw rate, steer angle, accel, slip",
        "nav": f"{c.n_lookahead} waypoints x (dist, sin, cos) of bearing",
        "traffic_light": f"{c.n_tl_obs} x (dist to stop line, sin, cos, red, yellow, "
                        f"green, steps to change)",
        "traffic": f"nearest {c.n_traffic_obs} moving vehicles x (dist, sin, cos, "
                   f"rel vx, rel vy)",
    }
    print("OBSERVATION (vector)")
    for name, sl in env.obs_slices.items():
        n = sl.stop - sl.start
        if n:
            print(f"  [{sl.start:>2}:{sl.stop:<2}] {n:>3} {name:<14} {desc[name]}")
    print(f"           {env.vector_dim:>3} {'total':<14} "
          f"ego-centric only -- no absolute position or heading")
    print()
    print("ACTION  Box(-1, 1, (3,))   throttle (signed, +fwd/-rev), brake ([0,1]), steer")
    print()
    print("CAR     kinematic bicycle model")
    print(f"  wheelbase {p.wheelbase} m, body {p.length} x {p.width} m")
    print(f"  max steer {np.degrees(p.max_steer):.0f} deg at {np.degrees(p.steer_rate):.0f} deg/s (rate limited)")
    print(f"  accel {p.max_accel} m/s2, brake {p.max_brake} m/s2, top speed {p.max_speed} m/s")
    print(f"  min turn radius {p.wheelbase / np.tan(p.max_steer):.2f} m")
    print()
    print(f"EPISODE dt={c.dt}s ({1/c.dt:.0f} Hz), max {c.max_episode_steps} steps, "
          f"{c.n_targets} waypoints, radius {c.target_radius} m")
    print(f"REWARD  progress x{c.progress_weight}, time -{c.time_penalty}/step, "
          f"crash -{c.crash_penalty}, waypoint +{c.target_bonus}")
    print()
    print(f"WORLD   {env.city.width}x{env.city.height} tiles @ {env.city.tile_size} m "
          f"= {env.city.extent[0]:.0f} x {env.city.extent[1]:.0f} m, resampled every reset")


def cmd_bench(args):
    print("throughput (headless, no graphics stack loaded for the vector case)")
    env = build_env(args, "vector")
    env.reset(seed=0)
    act = np.zeros(3, dtype=np.float32)
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        _, _, te, tr, _ = env.step(act)
        if te or tr:
            env.reset()
    t_vec = (time.perf_counter() - t0) / N
    print(f"  vector       : {t_vec*1e6:6.0f} us/step   {1/t_vec:>9,.0f} steps/s")

    try:
        from render.panda_renderer import PandaRenderer
    except ImportError:
        print("  image        : panda3d not installed")
        return
    # Panda3D allows one ShowBase per process and the buffer size is fixed when
    # it is created, so only one resolution can be timed per run.
    size = args.image_size
    r = PandaRenderer(offscreen=True, size=size)
    e = build_env(args, "both", renderer=r, image_size=size)
    e.reset(seed=0)
    M = 3000
    t0 = time.perf_counter()
    for _ in range(M):
        _, _, te, tr, _ = e.step(act)
        if te or tr:
            e.reset()
    t = (time.perf_counter() - t0) / M
    print(f"  +image {size:>3}   : {t*1e6:6.0f} us/step   {1/t:>9,.0f} steps/s")
    print(f"  rendering costs {(t-t_vec)*1e6:.0f} us/step at {size}x{size}")
    print(f"\n(--image-size N to time another resolution)")


def cmd_shots(args):
    from render.panda_renderer import PandaRenderer
    from PIL import Image

    r = PandaRenderer(offscreen=True, size=args.image_size)
    env = build_env(args, "both", renderer=r, image_size=args.image_size)
    driver = make_driver(env, args.agent)
    frames = []
    ep = 0
    while len(frames) < 8 and ep < 12:
        obs, _ = env.reset(seed=args.seed + ep)
        driver.reset()
        ep += 1
        k = 0
        while True:
            obs, _, te, tr, _ = env.step(driver.act(obs["vector"]))
            k += 1
            if k % args.every == 0 and len(frames) < 8:
                frames.append(obs["image"])
            if te or tr:
                break
    while len(frames) < 8:
        frames.append(np.zeros_like(frames[0]))
    grid = np.concatenate([np.concatenate(frames[0:4], axis=1),
                           np.concatenate(frames[4:8], axis=1)], axis=0)
    img = Image.fromarray(grid)
    if args.scale != 1:
        img = img.resize((grid.shape[1] * args.scale, grid.shape[0] * args.scale),
                         Image.NEAREST)
    img.save(args.out)
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]} at "
          f"{args.image_size}px per frame, scaled {args.scale}x)")


def cmd_serve(args):
    from serve.server import main as serve_main
    serve_main(host=args.host, port=args.port, seed=args.seed, map_size=args.map,
               n_targets=args.targets, traffic=args.traffic, world=args.world,
               agent=args.agent, manual=args.keys, rate=args.rate,
               pedestrians=args.pedestrians, speed_signs=args.signs)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["spec", "shots", "bench", "serve"])
    ap.add_argument("--host", default="127.0.0.1",
                    help="serve: bind address; 0.0.0.0 to reach it from another machine")
    ap.add_argument("--port", type=int, default=8765, help="serve: port")
    ap.add_argument("--traffic", default="normal", choices=["none", "light", "normal", "dense"],
                    help="serve: traffic preset")
    ap.add_argument("--world", default="city", choices=["city", "suburbs", "rural", "industrial"],
                    help="serve: world/terrain preset")
    ap.add_argument("--pedestrians", action="store_true",
                    help="zebra crossings with pedestrians (adds a 20-wide observation block)")
    ap.add_argument("--signs", action="store_true",
                    help="30/50 km/h zones and signs (adds a 3-wide observation block)")
    ap.add_argument("--agent", default="scripted",
                    help="shots/serve: 'scripted', 'random', 'manual', or a path to "
                         "a custom agent's JSON config (see agents/loader.py)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="serve: simulation speed relative to real time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", type=int, default=48, help="city size in tiles")
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--beams", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--keys", action="store_true",
                    help="serve: start in manual driving mode (arrow keys)")
    ap.add_argument("--every", type=int, default=70, help="shots: steps between frames")
    ap.add_argument("--scale", type=int, default=3, help="shots: upscale factor")
    ap.add_argument("--out", default="agent_view.png")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
