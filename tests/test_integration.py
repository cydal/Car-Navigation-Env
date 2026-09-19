"""The contract an *outside* learner depends on.

`test_core` checks the simulation and `test_env` checks that the task is
learnable. Neither covers the seam an algorithm actually sits on: seeding,
reward decomposition, termination semantics, and whether the objects handed out
survive a replay buffer or a worker process.

Everything here was written against a real defect. The headline one: `reset(seed=s)`
reseeded the city and the LIDAR but not the traffic, so calling it twice on one env
gave two different episodes. `test_env`'s determinism check missed it because it
built a fresh env per rollout -- which is not how a training loop is written.
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import carnav
from baselines.scripted import GapFollower
from wrappers import RewardOverrideWrapper

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def mk(**kw):
    return carnav.make(width=48, height=48, **kw)


print("=" * 70)
print("1. SEEDING CONTRACT")
print("=" * 70)
print("A run must be reproducible from the seed alone, however the env was built.\n")

# The regression: reset(seed) twice on ONE long-lived env.
e = mk(seed=0)
o1, _ = e.reset(seed=7)
snap1 = (e.city.grid.copy(), e.traffic.x.copy(), (e.car.x, e.car.y))
o2, _ = e.reset(seed=7)
snap2 = (e.city.grid.copy(), e.traffic.x.copy(), (e.car.x, e.car.y))
check("reset(seed) twice on one env -> identical observation", np.array_equal(o1, o2),
      f"max|diff| {np.abs(o1 - o2).max():.2e}")
check("  ... identical map, traffic and spawn",
      np.array_equal(snap1[0], snap2[0]) and np.array_equal(snap1[1], snap2[1])
      and snap1[2] == snap2[2])


def rollout(env, seed, n=200):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(1)
    frames, rs = [obs.copy()], []
    for _ in range(n):
        obs, r, te, tr, _ = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
        frames.append(obs.copy())
        rs.append(r)
        if te or tr:
            break
    return np.concatenate(frames), np.array(rs)


e = mk(seed=0)
a, ra = rollout(e, 7)
b, rb = rollout(e, 7)
check("reset(seed) twice -> identical rollout and rewards",
      a.shape == b.shape and np.array_equal(a, b) and np.allclose(ra, rb),
      f"{ra.size} steps")

# The two ways to seed must agree, or a config file and a CLI flag disagree.
p, _ = mk(seed=7).reset()
q, _ = mk(seed=0).reset(seed=7)
check("CarNavEnv(seed=s).reset() == CarNavEnv().reset(seed=s)", np.array_equal(p, q))

r1, _ = mk(seed=0).reset(seed=55)
r2, _ = mk(seed=999).reset(seed=55)
check("two independently built envs, same reset(seed) -> same episode",
      np.array_equal(r1, r2))

# Independent streams: changing how much one subsystem draws must not move the
# others. Proxy for that: with traffic switched off the city layout is unchanged.
g_on = mk(seed=0).reset(seed=3)[0]
e_on, e_off = mk(seed=0), mk(seed=0, traffic=False)
e_on.reset(seed=3); e_off.reset(seed=3)
check("traffic draws do not perturb the city stream",
      np.array_equal(e_on.city.grid, e_off.city.grid)
      and (e_on.car.x, e_on.car.y) == (e_off.car.x, e_off.car.y),
      "same map and spawn with traffic on and off")

s1 = [mk(seed=4).action_space.sample() for _ in range(3)]
s2 = [mk(seed=4).action_space.sample() for _ in range(3)]
check("action_space.sample() is reproducible from the env seed",
      all(np.array_equal(x, y) for x, y in zip(s1, s2)))

print()
print("=" * 70)
print("2. REWARD DECOMPOSITION")
print("=" * 70)
print("Reward must be exactly time + progress + waypoint + red-light + crash,")
print("reconstructible from info alone -- otherwise no one can debug a return.\n")

e = mk(seed=0)
c = e.cfg
drv = GapFollower.for_env(e)
worst, worst_ctx = 0.0, None
n_steps = n_wp = n_red = n_crash = n_stuck = 0
for ep in range(40):
    obs, info = e.reset(seed=5000 + ep)
    drv.reset()
    targets = list(e.targets)
    idx = 0
    prev_d = float(np.hypot(targets[0][0] - e.car.x, targets[0][1] - e.car.y))
    prev_v = 0
    while True:
        obs, r, te, tr, info = e.step(drv.act(obs))
        dv = info["red_light_violations"] - prev_v
        prev_v = info["red_light_violations"]
        n_red += dv
        exp = -c.time_penalty * c.action_repeat - c.red_light_penalty * dv
        if info["crashed"]:
            # A crash short-circuits before the progress term: you are not paid
            # for the metre in which you hit the wall.
            exp -= c.crash_penalty
            n_crash += 1
        elif info["reason"] == "stuck":
            exp -= c.time_penalty * c.action_repeat * max(
                0, c.max_episode_steps - info["step"])
            n_stuck += 1
        else:
            d = float(np.hypot(targets[idx][0] - info["x"], targets[idx][1] - info["y"]))
            exp += (prev_d - d) * c.progress_weight
            prev_d = d
            if d < c.target_radius:
                exp += c.target_bonus
                n_wp += 1
                idx += 1
                if idx < len(targets):
                    prev_d = float(np.hypot(targets[idx][0] - info["x"],
                                            targets[idx][1] - info["y"]))
            n_steps += 1
        if abs(exp - r) > worst:
            worst, worst_ctx = abs(exp - r), dict(ep=ep, step=info["step"],
                                                  reason=info["reason"],
                                                  got=round(r, 6), want=round(exp, 6))
        if te or tr:
            break
check("every step's reward matches its declared components", worst < 1e-6,
      f"max residual {worst:.2e} over {n_steps} steps "
      f"({n_wp} waypoints, {n_red} reds, {n_crash} crashes, {n_stuck} stuck)"
      + (f"\n         worst: {worst_ctx}" if worst >= 1e-6 else ""))
assert n_wp > 0 and n_red > 0 and n_crash > 0, \
    "decomposition check never exercised the waypoint/red-light/crash branches"

# episode_reward must be the running sum, or logged returns are fiction.
e = mk(seed=0)
off = 0
for ep in range(20):
    obs, info = e.reset(seed=90 + ep)
    tot = 0.0
    while True:
        obs, r, te, tr, info = e.step(np.array([0.6, -1.0, 0.3], dtype=np.float32))
        tot += r
        if te or tr:
            break
    off += abs(tot - info["episode_reward"]) > 1e-6
check("info['episode_reward'] == sum of step rewards", off == 0, f"{off}/20 episodes off")

print()
print("=" * 70)
print("3. TERMINATION SEMANTICS")
print("=" * 70)
print("terminated vs truncated decides whether a learner bootstraps V(s').\n")

# Each of the four endings is provoked deliberately rather than hoped for: an
# absent case would otherwise pass its assertion by default.
seen, both = {}, 0
PARK = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # zero throttle, full brake: held at 0
FLOOR = np.array([1.0, -1.0, 0.0], dtype=np.float32)


def collect(env, policy, seeds):
    global both
    for s in seeds:
        obs, info = env.reset(seed=s)
        while True:
            obs, r, te, tr, info = env.step(policy(obs))
            both += te and tr
            if te or tr:
                seen.setdefault(info["reason"], (te, tr))
                break


collect(mk(seed=0, stuck_steps=40), lambda o: PARK, [1])          # stuck
collect(mk(seed=0, stuck_steps=0, max_episode_steps=40), lambda o: PARK, [1])  # timeout
collect(mk(seed=0), lambda o: FLOOR, range(2, 8))                 # crash
_e = mk(seed=0)
_d = GapFollower.for_env(_e)
collect(_e, lambda o: _d.act(o), range(5000, 5040))               # success

for reason, (te, tr) in sorted(seen.items()):
    print(f"         {reason:<8} terminated={str(te):<5} truncated={str(tr)}")
check("all four endings are reachable", set(seen) == {"crash", "success", "stuck", "timeout"},
      f"saw {sorted(seen)}")
check("terminated and truncated are never both true", both == 0)
check("crash and success are terminated", seen["crash"] == seen["success"] == (True, False))
check("timeout is truncated, not terminated", seen["timeout"] == (False, True))
# `stuck` is terminated on purpose: the lump-sum penalty already accounts for the
# rest of the episode, so bootstrapping V(s') as well would double-count it.
check("stuck is terminated (lump sum replaces the bootstrap)", seen["stuck"] == (True, False))

a = mk(seed=0, stuck_steps=60)
b = mk(seed=0, stuck_steps=0)
tots = []
for env in (a, b):
    obs, info = env.reset(seed=31)
    t = 0.0
    while True:
        obs, r, te, tr, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        t += r
        if te or tr:
            break
    tots.append((t, info["step"], info["reason"]))
check("early 'stuck' exit is undiscounted-return-neutral vs idling to the limit",
      abs(tots[0][0] - tots[1][0]) < 1e-6,
      f"{tots[0][0]:+.1f} after {tots[0][1]} steps ({tots[0][2]}) vs "
      f"{tots[1][0]:+.1f} after {tots[1][1]} ({tots[1][2]})")

print()
print("=" * 70)
print("4. OBSERVATION CONTRACT")
print("=" * 70)

for kw, dim in ((dict(), 73), (dict(traffic_lights=False), 66),
                (dict(traffic=False), 53), (dict(traffic=False, traffic_lights=False), 46)):
    e = mk(seed=0, **kw)
    sl = e.obs_slices
    widths = sum(s.stop - s.start for s in sl.values())
    label = ", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults"
    check(f"obs_slices tile the vector exactly ({label})",
          widths == e.vector_dim == dim and e.observation_space.shape == (dim,),
          f"{dim}-D, blocks "
          + " ".join(f"{k}[{s.start}:{s.stop}]" for k, s in sl.items() if s.stop > s.start))

# Bounds under a policy that tries to saturate every channel.
e = mk(seed=0)
lo, hi, nonfinite, peak = np.inf, -np.inf, 0, 0.0
for ep in range(25):
    obs, _ = e.reset(seed=700 + ep)
    while True:
        obs, r, te, tr, info = e.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        nonfinite += not np.all(np.isfinite(obs))
        lo, hi = min(lo, float(obs.min())), max(hi, float(obs.max()))
        peak = max(peak, info["speed"])
        if te or tr:
            break
check("obs stays finite and inside the declared box at full throttle",
      nonfinite == 0 and lo >= -1 - 1e-5 and hi <= 1 + 1e-5,
      f"[{lo:.4f}, {hi:.4f}] at up to {peak:.1f} m/s")

# A replay buffer keeps the array it is handed; it must not be a live view.
e = mk(seed=0)
o0, i0 = e.reset(seed=1)
o1, _, _, _, i1 = e.step(np.zeros(3, dtype=np.float32))
check("observations are fresh arrays, safe to store in a replay buffer",
      not np.shares_memory(o0, o1) and not np.array_equal(o0, o1))
check("info is a fresh dict each step", i0 is not i1)

print()
print("=" * 70)
print("5. PROCESS AND LIFECYCLE")
print("=" * 70)

e = mk(seed=0)
e.reset(seed=1)
blob = pickle.dumps(e)
clone = pickle.loads(blob)
o_a = e.step(np.zeros(3, dtype=np.float32))[0]
o_b = clone.step(np.zeros(3, dtype=np.float32))[0]
check("env pickles, and the clone steps identically", np.array_equal(o_a, o_b),
      f"{len(blob)/1024:.0f} KB -- fine for SubprocVecEnv")

f = carnav.make_factory(traffic=False)
check("make_factory is picklable by the stdlib and builds an env",
      pickle.loads(pickle.dumps(f))().vector_dim == 53)

e = mk(seed=0)
e.reset(seed=1)
e.close()
e.close()
e.reset(seed=1)
e.step(np.zeros(3, dtype=np.float32))
check("close() is idempotent and the env is reusable after it", True)

e = mk(seed=0)
e.reset(seed=1)
e.step(np.array([50.0, -50.0, 900.0], dtype=np.float32))
check("out-of-range actions are clipped rather than fatal", True)
try:
    e.step(np.zeros(5, dtype=np.float32))
    check("a wrong-shape action raises", False, "a (5,) action was accepted")
except Exception as ex:
    check("a wrong-shape action raises", True, type(ex).__name__)

try:
    carnav.make(traffic_light=False)     # note the missing 's'
    check("carnav.make rejects a typo'd setting", False,
          "traffic_light=False was silently ignored, leaving lights on")
except TypeError:
    check("carnav.make rejects a typo'd setting", True, "traffic_light -> TypeError")

print()
print("=" * 70)
print("6. GYMNASIUM CONFORMANCE")
print("=" * 70)

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
except ImportError:
    # Reported rather than passed silently: without gymnasium installed the
    # `gym.Env` branch of nav_env.py never executes, so the "Gymnasium-compatible"
    # claim is unverified here. It is verified in an env with the `gym` extra.
    print("  [SKIP] gymnasium is not installed -- the gym.Env branch was not exercised.")
    print("         Run this suite in an env with `pip install -e '.[gym]'` to cover it.")
else:
    import warnings
    from env.nav_env import CarNavEnv
    check("CarNavEnv subclasses gymnasium.Env", issubclass(CarNavEnv, gym.Env))

    for kw in (dict(), dict(traffic=False, traffic_lights=False)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                check_env(mk(seed=0, **kw), skip_render_check=True)
                ok, detail = True, f"{len(w)} warnings"
            except Exception as ex:
                ok, detail = False, f"{type(ex).__name__}: {ex}"
        check(f"gymnasium env_checker passes ({', '.join(f'{k}={v}' for k, v in kw.items()) or 'defaults'})",
              ok, detail)

    # `env.np_random` is contract, not decoration: env_checker fails outright if a
    # seeded reset leaves it unset, which it did until it got its own stream.
    a, b = mk(seed=0), mk(seed=0)
    a.reset(seed=11); b.reset(seed=11)
    check("env.np_random is seeded and reproducible", a.np_random.random() == b.np_random.random())

    p, q = mk(seed=0), mk(seed=0)
    p.reset(seed=11); q.reset(seed=11)
    p.np_random.random(50)                       # stand-in for a wrapper drawing
    check("draws from env.np_random do not perturb the simulation",
          np.array_equal(p.step(np.zeros(3, np.float32))[0],
                         q.step(np.zeros(3, np.float32))[0]),
          "a wrapper cannot shift a map layout")

    from gymnasium.vector import SyncVectorEnv
    v = SyncVectorEnv([carnav.make_factory(width=48, height=48, traffic=False)
                       for _ in range(3)])
    o, _ = v.reset(seed=[1, 2, 3])
    for _ in range(20):
        o, r, te, tr, _ = v.step(v.action_space.sample())
    check("SyncVectorEnv steps", o.shape == (3, 53) and r.shape == (3,), f"obs {o.shape}")
    v.close()

    g = gym.make("CarNav-v0", width=48, height=48)
    # A registered max_episode_steps would add a TimeLimit on top of the env's own,
    # giving two limits and only one of them recorded in info.
    check("registered ids add no second TimeLimit", g.spec.max_episode_steps is None,
          f"spec.max_episode_steps={g.spec.max_episode_steps}")
    g.close()
    d = gym.make("CarNavBoth-v0", width=48, height=48)
    check("CarNavBoth-v0 exposes a Dict observation space",
          sorted(d.unwrapped.observation_space.spaces) == ["image", "vector"])
    d.close()

print()
print("=" * 70)
print("7. REWARD OVERRIDE (wrappers.RewardOverrideWrapper)")
print("=" * 70)
print("A training codebase must be able to substitute its own reward for the")
print("env's own, with env/ changing nothing and info['reward_components']")
print("still carrying the original breakdown underneath the override.\n")


def sparse_only(obs, action, reward, terminated, truncated, info):
    return info["reward_components"]["target_bonus"]


env_w = RewardOverrideWrapper(mk(seed=0), sparse_only)
drv = GapFollower.for_env(env_w)
mismatches = n_hits = n_steps2 = 0
for ep in range(20):
    obs, info = env_w.reset(seed=6000 + ep)
    drv.reset()
    while True:
        obs, r, te, tr, info = env_w.step(drv.act(obs))
        n_steps2 += 1
        want = info["reward_components"]["target_bonus"]
        if abs(r - want) > 1e-9:
            mismatches += 1
        if want > 0:
            n_hits += 1
        if te or tr:
            break
check("reward_fn's return value is exactly what step() hands back",
      mismatches == 0, f"{mismatches}/{n_steps2} steps mismatched")
check("the override actually took effect (reward is sparse, not the dense default)",
      n_hits > 0, f"{n_hits} waypoint-bonus steps out of {n_steps2}")

# The env's own accounting is untouched by the override -- the documented gotcha.
plain, env_w2 = mk(seed=0), RewardOverrideWrapper(mk(seed=0), sparse_only)
plain.reset(seed=6000)
env_w2.reset(seed=6000)
info_p = info_w = None
for _ in range(60):
    a = np.array([0.6, -1.0, 0.1], dtype=np.float32)
    _, _, te_p, tr_p, info_p = plain.step(a)
    _, _, te_w, tr_w, info_w = env_w2.step(a)
    if te_p or tr_p or te_w or tr_w:
        break
check("info['episode_reward'] still reflects the env's own formula, not the override",
      abs(info_p["episode_reward"] - info_w["episode_reward"]) < 1e-9,
      f"plain={info_p['episode_reward']:.4f} wrapped={info_w['episode_reward']:.4f}")

check("RewardOverrideWrapper passes unrelated attributes through to the env",
      env_w.cfg is env_w.env.cfg and env_w.action_space is not None)

print()
print("=" * 70)
if FAILED:
    print(f"{len(FAILED)} INTEGRATION CHECK(S) FAILED")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("ALL INTEGRATION CHECKS PASSED")
print("=" * 70)
