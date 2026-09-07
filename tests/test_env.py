"""Env contract checks, throughput benchmark, and scripted-policy validation."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import CityConfig
from env.nav_env import CarNavEnv, EnvConfig
from baselines.scripted import GapFollower

CITY = CityConfig(width=48, height=48, tile_size=4.0)


def make_env(seed=0, **kw):
    cfg = EnvConfig(**kw)
    return CarNavEnv(config=cfg, city_config=CITY, obs_type="vector", seed=seed)


print("=" * 62)
print("1. SPACES & OBSERVATION CONTRACT")
print("=" * 62)
env = make_env(seed=1)
obs, info = env.reset(seed=1)
print(f"observation_space : {env.observation_space}")
print(f"action_space      : {env.action_space}")
print(f"vector dim        : {env.vector_dim}  (= {env.cfg.n_beams} lidar + 5 dyn + {3*env.cfg.n_lookahead} nav)")
print(f"obs shape/dtype   : {obs.shape} {obs.dtype}")
assert obs.shape == (env.vector_dim,) and obs.dtype == np.float32

# Observation must stay inside the declared box, otherwise normalisation is a lie.
for i in range(300):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (env.vector_dim,)
    assert np.all(np.isfinite(obs)), "non-finite observation"
    assert obs.min() >= -1.0 - 1e-5 and obs.max() <= 1.0 + 1e-5, \
        f"obs out of box: [{obs.min():.3f}, {obs.max():.3f}]"
    if term or trunc:
        obs, info = env.reset()
print("obs stays in [-1, 1] over 300 random steps : OK")

# Determinism: same seed must reproduce the same trajectory.
def rollout(seed, n=120):
    e = make_env(seed=seed)
    o, _ = e.reset(seed=seed)
    rng = np.random.default_rng(123)
    trace = [o.copy()]
    for _ in range(n):
        o, r, te, tr, _ = e.step(rng.uniform(-1, 1, 3))
        trace.append(o.copy())
        if te or tr:
            break
    return np.concatenate(trace)

assert np.allclose(rollout(42), rollout(42)), "env is not deterministic under a fixed seed"
print("determinism (same seed -> same trajectory) : OK")

print()
print("=" * 62)
print("2. MAP RANDOMISATION")
print("=" * 62)
env = make_env(seed=3)
grids = []
for _ in range(5):
    env.reset()
    grids.append(env.city.grid.copy())
uniq = len({g.tobytes() for g in grids})
print(f"distinct layouts over 5 resets : {uniq}/5")
assert uniq == 5, "map is not being resampled"
env.cfg.randomize_map = False
env.reset(); g0 = env.city.grid.copy()
env.reset(); g1 = env.city.grid.copy()
assert np.array_equal(g0, g1)
print("randomize_map=False keeps the layout fixed : OK")

print()
print("=" * 62)
print("3. THROUGHPUT (vector obs, headless)")
print("=" * 62)
env = make_env(seed=5)
env.reset()
N = 20000
act = np.zeros(3, dtype=np.float32)
t0 = time.perf_counter()
resets = 0
for _ in range(N):
    _, _, te, tr, _ = env.step(act)
    if te or tr:
        env.reset(); resets += 1
el = time.perf_counter() - t0
print(f"{N} steps ({resets} resets) in {el:.2f}s")
print(f"  -> {N/el:,.0f} steps/s   ({el/N*1e6:.0f} us/step)")

# Break down where the time goes.
env.reset()
t0 = time.perf_counter()
for _ in range(5000):
    env.vector_obs()
t_obs = (time.perf_counter() - t0) / 5000 * 1e6
t0 = time.perf_counter()
for _ in range(5000):
    env.car.step(1.0, 0.0, 0.2, 0.05)
t_phys = (time.perf_counter() - t0) / 5000 * 1e6
t0 = time.perf_counter()
for _ in range(5000):
    env.city.collides(env.car.x, env.car.y, env.car.heading, 4.4, 1.9)
t_coll = (time.perf_counter() - t0) / 5000 * 1e6
t0 = time.perf_counter()
for _ in range(20):
    env.city.generate()
t_gen = (time.perf_counter() - t0) / 20 * 1e3
print(f"  breakdown: obs(lidar) {t_obs:.0f}us | physics {t_phys:.1f}us | collision {t_coll:.1f}us")
print(f"  map generation: {t_gen:.1f} ms per reset")

print()
print("=" * 62)
print("4. REWARD SANITY")
print("=" * 62)
# Crashing must be heavily penalised and terminate.
# Drive straight, not at full lock: the minimum turn radius (4 m) fits inside a
# 12 m corridor, so a full-lock car circles indefinitely without ever hitting
# anything. Straight ahead reaches the end of a street.
env = make_env(seed=11)
env.reset(seed=11)
crash_r = None
for _ in range(4000):
    _, r, te, tr, info = env.step(np.array([1.0, -1.0, 0.0]))  # full throttle, straight
    if te and info["reason"] == "crash":
        crash_r = r
        break
    if te or tr:
        env.reset()
print(f"crash step reward : {crash_r:.1f}  (expect around -{env.cfg.crash_penalty})")
assert crash_r is not None and crash_r < -50

# Idling for a whole episode should cost about as much as one crash, so that
# "stop and do nothing" is never a safe hiding strategy.
env = make_env(seed=12)
env.reset(seed=12)
total = 0.0
for _ in range(env.cfg.max_episode_steps):
    _, r, te, tr, info = env.step(np.array([-1.0, 1.0, 0.0]))  # no throttle, full brake
    total += r
    if te or tr:
        break
print(f"full-brake episode: reward={total:.1f} reason={info['reason']} steps={info['step']}")
assert total < -env.cfg.crash_penalty * 0.7, "idling is too cheap -- agent will learn to park"
print("idling is not cheaper than crashing : OK")

print()
print("=" * 62)
print("5. SCRIPTED BASELINE (obs-only follow-the-gap + pure pursuit)")
print("=" * 62)
print("Validates the task is solvable AND that the vector obs alone suffices.\n")
env = make_env(seed=99)
p = env.car.p
driver = GapFollower.for_env(env)

EPISODES = 40
reasons, rewards, reached, steps, speeds = {}, [], [], [], []
for ep in range(EPISODES):
    obs, info = env.reset(seed=1000 + ep)
    driver.reset()
    done = False
    while not done:
        obs, r, te, tr, info = env.step(driver.act(obs))
        speeds.append(info["speed"])
        done = te or tr
    reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
    rewards.append(info["episode_reward"])
    reached.append(info["targets_reached"])
    steps.append(info["step"])

n_t = env.cfg.n_targets
success = reasons.get("success", 0)
print(f"episodes            : {EPISODES}  ({n_t} waypoints each)")
print(f"outcomes            : {reasons}")
print(f"full-route success  : {success}/{EPISODES} = {success/EPISODES*100:.0f}%")
print(f"waypoints reached   : mean {np.mean(reached):.2f}/{n_t}  "
      f"({np.sum(reached)}/{EPISODES*n_t} = {np.sum(reached)/(EPISODES*n_t)*100:.0f}%)")
print(f"episode reward      : mean {np.mean(rewards):8.1f}   median {np.median(rewards):8.1f}")
print(f"episode length      : mean {np.mean(steps):.0f} steps")
print(f"mean speed          : {np.mean(speeds):.1f} m/s ({np.mean(speeds)*3.6:.0f} km/h)")

# Compare against a random policy to confirm the reward signal discriminates.
rnd_rewards, rnd_reached = [], []
rng = np.random.default_rng(0)
for ep in range(EPISODES):
    obs, info = env.reset(seed=1000 + ep)
    done = False
    while not done:
        obs, r, te, tr, info = env.step(rng.uniform(-1, 1, 3))
        done = te or tr
    rnd_rewards.append(info["episode_reward"])
    rnd_reached.append(info["targets_reached"])

print()
print(f"{'policy':<12}{'mean reward':>14}{'waypoints':>12}")
print(f"{'-'*38}")
print(f"{'random':<12}{np.mean(rnd_rewards):>14.1f}{np.mean(rnd_reached):>12.2f}")
print(f"{'scripted':<12}{np.mean(rewards):>14.1f}{np.mean(reached):>12.2f}")
print(f"{'gap':<12}{np.mean(rewards)-np.mean(rnd_rewards):>14.1f}")

assert np.mean(reached) > np.mean(rnd_reached) + 1.0, "scripted policy is not beating random"
assert np.mean(rewards) > np.mean(rnd_rewards) + 200, "reward does not separate good from bad driving"
assert success >= EPISODES * 0.3, "task looks too hard -- baseline should complete some routes"
assert success <= EPISODES * 0.9, "task looks too easy -- leaves no headroom for a learned policy"
print("\nscripted >> random on both reward and waypoints : OK")
print("difficulty is in the useful band (30-90% baseline success) : OK")

print()
print("=" * 62)
print("ALL ENV CHECKS PASSED")
print("=" * 62)
