# Using this environment from a learning algorithm

Everything an outside implementation — PPO, SAC, DreamerV3, a custom world model —
needs in order to treat `CarNavEnv` as a fixed, trustworthy substrate. Nothing here
requires reading the simulation code.

The contract below is machine-checked by `tests/test_integration.py`. If a claim on
this page is wrong, that suite should fail; if it does not, the suite has a gap.

- [Getting the env](#getting-the-env)
- [Observation](#observation)
- [Action](#action)
- [Reward](#reward)
- [Overriding the reward](#overriding-the-reward)
- [What to beat](#what-to-beat)
- [Episode endings and bootstrapping](#episode-endings-and-bootstrapping)
- [Seeding and reproducibility](#seeding-and-reproducibility)
- [The `info` dict](#the-info-dict)
- [Running many envs](#running-many-envs)
- [Image observations](#image-observations)
- [Notes for world models](#notes-for-world-models)
- [Agents, replay, and structured sensing](#agents-replay-and-structured-sensing)
- [Visualising a world model's imagination](#visualising-a-world-models-imagination)
- [Things that will bite you](#things-that-will-bite-you)

---

## Getting the env

Three ways in, in order of preference.

**1. Editable install.** `env` is a generic top-level package name, so do this in a
virtualenv dedicated to the project.

```bash
pip install -e .              # numpy only
pip install -e '.[gym]'       # + gymnasium: gym.make ids, wrappers, vector envs
pip install -e '.[render]'    # + panda3d, pillow: image observations
```

**2. No install.** Put the repo root on the path; this is what `tests/` does.

```bash
export PYTHONPATH=/path/to/rl-env3d
```

**3. Vendored.** The env has one hard dependency (numpy) and imports nothing from
`render/`, `baselines/` or `main.py`, so `env/` plus `carnav.py` can be copied
wholesale.

### Constructing one

```python
import carnav

env = carnav.make()                        # 73-D vector obs, traffic + lights
env = carnav.make(traffic=False)           # 53-D, lights only
env = carnav.make(obs_type="both", image_size=64)
env = carnav.make(n_beams=64, road_width=2, max_speed=15.0, width=64, height=64)
```

`carnav.make` takes a **flat** keyword namespace and routes each setting to the
config object that owns it. This matters because the settings that must move
together live in three different places, and a training script that builds them by
hand gets one wrong eventually — silently, because a mismatched config still
produces a perfectly valid episode.

| keyword group | goes to | examples |
|---|---|---|
| task, episode, reward, sensor widths | `EnvConfig` | `traffic`, `traffic_lights`, `n_beams`, `dt`, `action_repeat`, `max_episode_steps`, `crash_penalty`, `n_targets` |
| map | `CityConfig` | `width`, `height`, `tile_size`, `road_width`, `max_signals` |
| vehicle | `CarParams` | `max_speed`, `max_accel`, `max_brake`, `max_steer`, `wheelbase`, `car_length`, `car_width` |

`width`/`height` are the **map** in tiles; the car's body is `car_width` /
`car_length`. Anything unrecognised raises `TypeError` rather than being dropped —
a typo'd `traffic_light=False` that quietly leaves lights on invalidates a run
without ever failing.

`python main.py spec` prints the resulting contract for any configuration.

### Via gymnasium

`import carnav` registers three ids (a no-op without gymnasium installed):

```python
import carnav, gymnasium as gym
env = gym.make("CarNav-v0")                          # vector
env = gym.make("CarNavImage-v0")                     # image
env = gym.make("CarNavBoth-v0", traffic=False)       # dict obs, kwargs pass through
```

No `max_episode_steps` is registered, deliberately — see
[Episode endings](#episode-endings-and-bootstrapping).

### The loop

```python
import numpy as np, carnav

env = carnav.make()
obs, info = env.reset(seed=0)
while True:
    action = policy(obs)                    # np.float32, shape (3,), in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info["reason"], info["episode_reward"], info["targets_reached"])
        obs, info = env.reset()
        break
```

Standard Gymnasium 5-tuple. `reset` returns `(obs, info)`. The env subclasses
`gymnasium.Env` when gymnasium is importable and falls back to `object` with a
minimal `Box` shim otherwise, so **if your algorithm relies on
`isinstance(env, gym.Env)` — as Stable-Baselines3 does — install the `gym` extra.**

`gymnasium.utils.env_checker.check_env` passes with **zero warnings** on both the
default and the no-traffic-no-lights configuration (checked against gymnasium
1.3.0), as do `SyncVectorEnv`, `AsyncVectorEnv`, `RecordEpisodeStatistics` and
`FrameStackObservation`. Section 6 of `tests/test_integration.py` runs those and
prints a visible `[SKIP]` when gymnasium is absent rather than passing quietly, so
an uncovered branch is never mistaken for a covered one.

---

## Observation

`obs_type` selects `"vector"`, `"image"` or `"both"` (a dict with both keys).

### Vector

`float32`, shape `(env.vector_dim,)`, **every element in [-1, 1]**, entirely
ego-centric — no absolute position or heading anywhere. Verified inside the
declared box under a policy that saturates the car at its 22 m/s top speed.

Never recompute the block offsets. Ask the env:

```python
sl = env.obs_slices          # {'lidar': slice(0,32), 'dynamics': slice(32,37), ...}
lidar = obs[sl["lidar"]]
```

With the defaults (`n_beams=32`, `n_lookahead=3`, `n_tl_obs=1`, `n_traffic_obs=4`):

| block | slice | dims | contents |
|---|---|---|---|
| `lidar` | `0:32` | 32 | 360° ranges / 60 m. Index `n_beams // 2` (= 16) is straight ahead; index 0 is straight back, so the array's wrap-around discontinuity sits at the rear. `1.0` = nothing in range. Vehicles are in this scan as geometry. |
| `dynamics` | `32:37` | 5 | speed / 22 (slightly negative in reverse, down to -0.27), yaw rate / 2, steer angle / lock, accel / max_brake, slip / lock |
| `nav` | `37:46` | 9 | next 3 waypoints × (distance / 150 m clipped to 1, sin, cos of bearing) |
| `traffic_light` | `46:53` | 7 | nearest signal: distance **to the stop line** / 60 m, sin, cos of bearing, then one-hot red / yellow / green **for this car's approach axis**, then steps until the phase changes / 100 |
| `traffic` | `53:73` | 20 | nearest 4 **moving** vehicles × (distance / 60 m, sin, cos of bearing, relative velocity x and y in the ego frame / 22) |
| `pedestrians` | `73:93` | 20 | **opt-in** (`pedestrians=True`): nearest 4 people × the same 5 features as a traffic slot. People are also in the LIDAR scan as 0.3 m circles. |
| `signs` | `93:96` | 3 | **opt-in** (`speed_signs=True`): active speed limit / 22, next sign ahead's distance / 60 m (1 if none), its limit / 22 (0 if none). The *active* limit is given directly, so no sign memory is needed to stay Markov. |

Blocks a switch removes become **empty slices**, not missing keys, so
`obs[sl["traffic"]]` is always valid. The two opt-in blocks are appended
*behind* everything else, so switching them on never moves an existing index:

| config | `vector_dim` | vehicles |
|---|---|---|
| defaults | 73 | 26 (8 moving, 18 parked) |
| `traffic_lights=False` | 66 | 26 |
| `traffic=False` | 53 | 0 |
| both off | 46 | 0 — the original task |
| `pedestrians=True` | 93 | 26, plus ~1–3 people at each of 6 zebra crossings |
| `speed_signs=True` | 76 | 26, plus 30 km/h zones and signs around 6 crossings |
| both on | 96 | the fullest task |

### Padding conventions

Read these before writing an encoder; they are the only way to tell "absent" from
"real".

- **Absent waypoint** (sequence exhausted): `(1.0, 0.0, 0.0)`. `sin = cos = 0` is
  not a unit vector, so it cannot collide with any real bearing.
- **Absent or out-of-range signal**: `(1.0, 0, 0, 0, 0, 1, 1)` — green at full
  distance with a full phase remaining, which no approaching red can produce.
- **Absent traffic slot**: `(1.0, 0, 0, 0, 0)`, again with `sin = cos = 0`.

### Where the encoding saturates

Two channels clip, and an encoder should know it rather than discover it:

- **Relative velocity** is normalised by the ego's `max_speed` (22 m/s), but a
  head-on closing rate reaches ego speed + 9 m/s of traffic. It clips at
  ±1 on roughly **1% of (step, channel) pairs** under a full-throttle policy — so
  the fastest head-on approaches are compressed. Raise `traffic_obs_range` or
  rescale in a wrapper if your method is sensitive to that tail.
- **`accel`** is divided by `max_brake` (9 m/s²) while full throttle is only
  4.5 m/s², so the channel occupies roughly [-1, +0.5] rather than [-1, 1].

Everything else uses its full range. There is no reason to normalise the vector
again; do consider scaling **reward** (below).

### Auxiliary sensors (not in the observation)

`env.radar` is a per-sector (default 8) nearest-*moving*-vehicle range +
closing-speed sensor, built for the live viewer's radar readout — not for
training. It updates every `reset`/`step` but is never concatenated into the
vector observation, so `vector_dim`/`obs_slices` above are unaffected by it
regardless of `EnvConfig(radar=..., n_radar_sectors=..., radar_range=...)`.
Read `env.radar.last_distances` / `.last_closing_speed` / `.sector_bearing(i)`
directly if you want it in a wrapper's observation.

### Image

`(image_size, image_size, 3)` `uint8`, third-person chase camera, default 64×64.
See [Image observations](#image-observations).

---

## Action

`Box(-1, 1, (3,))`, `float32`:

| index | channel | mapping |
|---|---|---|
| 0 | throttle | **signed**: passed straight through. Positive drives forward, negative reverses. |
| 1 | brake | rescaled to [0, 1] |
| 2 | steer | a **command**, not an angle; passes through a rate limiter |

`-1` therefore means "full reverse" for throttle, but "no brake" for brake, so
the zero action `[0, 0, 0]` is coasting (no drive, no brake) -- unlike brake,
throttle needed no rescale since it was already the signed quantity the car
wants.

Out-of-range actions are clipped, not rejected. A wrong **shape** raises.
Braking alone can never reverse the car -- it only ever decelerates whichever
direction the car is already moving in, and stops exactly at zero. Only a
negative throttle command drives it backward, up to `max_reverse_speed`
(6 m/s by default, slower than the forward top speed) at `max_reverse_accel`
(3 m/s², gentler than the forward 4.5 m/s²).

Steer is rate limited at 150°/s toward a 32° lock, and the resulting angle is
hidden state — which is why it is in the observation. Do not assume the commanded
steer took effect this step.

---

## Reward

Every step's reward is **exactly**:

```
reward = -time_penalty * action_repeat
       + progress_weight * (dist_to_target_before - dist_to_target_after)
       + target_bonus      per waypoint reached this step
       - red_light_penalty per red-light entry this step
       - crash_penalty     if this step crashed into a building or vehicle
       - pedestrian_penalty  if this step hit a person            (pedestrians=True only)
       - speeding_penalty * max(0, speed - limit)  per physics step (speed_signs=True only)
       - time_penalty * action_repeat * steps_remaining   if this step ended as "stuck"
```

| term | default | notes |
|---|---|---|
| `time_penalty` | 0.1 / physics step | charged inside the action-repeat loop, so a step costs `0.1 × action_repeat` |
| `progress_weight` | 1.0 | metres closed toward the **current** waypoint |
| `target_bonus` | 100.0 | per waypoint |
| `crash_penalty` | 100.0 | building or vehicle alike; `info["crash_with"]` says which |
| `red_light_penalty` | 50.0 | on **entry only**, never per step of occupancy |
| `pedestrian_penalty` | 300.0 | **opt-in**; terminates with `crash_with="pedestrian"`; deliberately larger than a vehicle crash and its own `reward_components` entry, so an agent has a reason to treat a person differently from a parked van |
| `speeding_penalty` | 0.05 per m/s over, per physics step | **opt-in**; 5 m/s over the limit for 5 s costs 25. The limit is 50 km/h, or 30 within a zone around a crossing |

Reconstructed from `info` alone and checked against the env's own reward to a **max
residual of 7.1e-15** — over 18,363 steps on every `test_integration` run (51
waypoints, 21 red-light violations, 31 crashes) and 59,254 steps in the wider audit
that established it. `info["episode_reward"]` is the exact running sum, so a logged
return is never fiction.

Nothing else is in there. There is no lane-keeping term, no comfort term, no speed
term, no shaping on the traffic block. If you want any of those, add them in a
wrapper — the env deliberately does not, so that a published number says which
objective it optimised.

Two behaviours worth knowing:

- **Progress telescopes.** Over a stretch with no waypoint hit, the progress terms
  sum to the net distance closed, so there is no free reward in driving out and
  back. On the step a waypoint is reached, progress is credited against the
  **old** waypoint and the reference is then re-anchored to the new one — you are
  not charged for the 40 m jump to the next target.
- **A crash credits no progress.** The step short-circuits at the collision check,
  so up to ~1 m of movement goes unpaid (measured mean −0.22 m, max 1.09 m — ~1%
  of the crash penalty). You are not paid for the metre in which you hit the wall.

**Scale.** Episode returns run roughly −100 (random policy) to +400 (three
waypoints). A single step can be +100 or −100 against a typical −0.1, so the
reward distribution is extremely heavy-tailed. Value-based methods generally want
reward scaling, return normalisation, or a smaller `target_bonus`; that is a
learner-side decision and the env does not do it for you.

---

## Overriding the reward

Everything above is the reward `CarNavEnv.step()` computes internally, and it
never changes — that fixed formula is what makes it comparable across runs and
methods. A training codebase that wants a *different* reward (sparse-only,
additive shaping on top of it, a different scale, a completely custom scheme)
does not fork the env; it wraps it, the same way `replay.RecordingWrapper`
does:

```python
import carnav
from wrappers import RewardOverrideWrapper

def sparse_only(obs, action, reward, terminated, truncated, info):
    return info["reward_components"]["target_bonus"]      # ignore everything else

def plus_lane_bonus(obs, action, reward, terminated, truncated, info):
    return reward + my_lane_center_bonus(info)             # additive shaping

env = RewardOverrideWrapper(carnav.make(), sparse_only)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)   # reward is sparse_only's
```

`reward_fn(obs, action, reward, terminated, truncated, info)` is called on
every `step()`; whatever it returns is what the caller sees instead of the
env's own reward. The env still computes its usual reward internally first —
unchanged, still exactly the formula above, and still what `info[
"reward_components"]` breaks down — so `reward_fn` can build additively on
`reward`/`info` (as `plus_lane_bonus` does) or ignore both for a fully custom
scheme (as `sparse_only` does). `env/nav_env.py` has zero lines of override-
related code, checked the same way the Agent/replay boundary is: nothing here
had to change for this to work.

**`RewardOverrideWrapper` composes with `RecordingWrapper`.** Order controls
what gets recorded:

```python
from replay import ReplayBuffer, RecordingWrapper

# Records the OVERRIDDEN reward:
env = RecordingWrapper(RewardOverrideWrapper(carnav.make(), reward_fn), ReplayBuffer(200_000))
# Records the ENV'S OWN reward, override applied only after step() returns:
env = RewardOverrideWrapper(RecordingWrapper(carnav.make(), ReplayBuffer(200_000)), reward_fn)
```

**The one gotcha:** `info["episode_reward"]` is the *env's own* running sum of
its own formula, computed with no idea this wrapper exists — it will not match
a running sum of what `reward_fn` returns. This is why the mechanism is a
wrapper and not an `EnvConfig` flag: the env's internal accounting stays
correct and auditable no matter what a wrapper on top of it does, at the cost
of `episode_reward` no longer being *your* number once you override. Track
your own running return if you override the reward and want to log it.

Checked in `tests/test_integration.py` §7: `reward_fn`'s return value is
exactly what `step()` hands back, the override actually changes the reward
(not just wraps a no-op), `info["episode_reward"]` is provably unaffected by
the override, and unrelated attributes (`.cfg`, `.action_space`, ...) still
reach the wrapped env through `__getattr__`.

---

## What to beat

Reference points from `baselines/scripted.py` — a follow-the-gap + pure-pursuit
controller that reads **only the observation vector**, with no privileged access to
the world, the car object or the target list. Over 300 pooled episodes on seeds
5000–5299, 3 waypoints each:

| task | dims | mean reward | waypoints | full route |
|---|---|---|---|---|
| random policy | 73 | −103 | 0.00 / 3 | 0% |
| `traffic=False, traffic_lights=False` | 46 | +222 | 1.90 / 3 | 44.0% |
| `traffic=False` | 53 | +179 | 1.86 / 3 | 42.7% |
| `traffic_lights=False` | 66 | +137 | 1.42 / 3 | 26.7% |
| defaults | 73 | +129 | 1.58 / 3 | 31.3% |

Because the arms are paired (same maps, spawns and waypoints), the differences carry
small standard errors: signals cost 42.3 ±5.4 reward but only 1.3 ±1.6 points of
completion, while moving traffic costs 84.9 ±10.1 reward and 17.3 ±2.4 points.

Two things to take from this if you are choosing a task to train on:

- **`traffic=False, traffic_lights=False` (46-D) is the sane first target.** It is
  the original navigation problem, roughly 2× the step rate, and the 44% baseline
  sits in the middle of the band where a success rate can actually move.
- **The default 73-D task has real headroom, and the baseline's ceiling is
  structural.** Its residual failure is dominated by *building* crashes (131 of 300)
  rather than vehicle ones (60), and it sees the vehicle it eventually hits for a
  median of 66 steps beforehand — so the gap is not sensing, it is the absence of
  any concept of a lane, of right of way, or of another driver's intent. That is
  exactly the headroom a learned policy is meant to claim.

Success rate is a per-episode Bernoulli: a 40-episode block has a ~7 point standard
error. Quote pooled runs of a few hundred episodes, and use the same seed list for
every arm.

## Episode endings and bootstrapping

| `info["reason"]` | `terminated` | `truncated` | what it means | bootstrap `V(s')`? |
|---|---|---|---|---|
| `success` | `True` | `False` | all waypoints reached | no |
| `crash` | `True` | `False` | hit a building or a vehicle | no |
| `stuck` | `True` | `False` | idled `stuck_steps` (150) steps below 0.5 m/s | **no** — see below |
| `timeout` | `False` | `True` | hit `max_episode_steps` (1000) | **yes** |

They are never both true, and `info["reason"]` is `None` until the episode ends.
If you handle the 5-tuple in the standard way — bootstrap on `truncated`, not on
`terminated` — the semantics above are already what you want.

**`stuck` is terminated on purpose, and it is not an MDP-terminal state.** The env
charges the *entire remaining* time penalty as a lump sum at that step, so ending
early is neutral in undiscounted return against idling to the time limit
(verified: identical −100.0 either way). Bootstrapping `V(s')` **as well** would
double-count the future, which is why it is reported as terminated.

The honest caveat: neutrality holds **undiscounted**. With γ < 1 a lump sum paid
now is worth more than the same penalties spread over 900 future steps, so under
discounting getting stuck is *strictly worse* than crawling. That is a defensible
shaping choice — it discourages parking — but it is shaping, and if your method is
sensitive to it, `stuck_steps=0` disables the detector entirely and lets episodes
run to `timeout`.

**Do not add a second time limit.** The env applies its own from
`max_episode_steps` and reports it as `truncated`. Wrapping it in
`gymnasium.wrappers.TimeLimit`, or registering an id with `max_episode_steps`, adds
an outer limit that fires without the inner bookkeeping. The registered ids
deliberately omit it.

The **final observation is returned** on both termination and truncation, so a
wrapper has what it needs to bootstrap from it.

---

## Seeding and reproducibility

```python
env = carnav.make(seed=0)      # seed at construction
obs, info = env.reset(seed=7)  # or per-episode; this is the authority
```

Guaranteed, and asserted in `tests/test_integration.py`:

- `env.reset(seed=s)` twice **on the same long-lived env** reproduces the episode
  exactly — map, spawn, waypoints, traffic, signal phases, rewards.
- `carnav.make(seed=s).reset()` == `carnav.make().reset(seed=s)`.
- Two separately constructed envs given the same `reset(seed=s)` produce the same
  episode.
- `env.action_space.sample()` is reproducible from the env's seed (the gymnasium
  space is seeded, and the no-gymnasium shim carries its own generator rather than
  drawing from the global `np.random`).
- `env.np_random` — gymnasium's own generator, which wrappers draw from — is seeded
  from the same seed and is reproducible. Drawing from it does **not** perturb the
  simulation.

Each subsystem — city, LIDAR, traffic, spaces, `np_random` — gets an **independent
stream** spawned from the seed via `SeedSequence`, rather than sharing one
generator. Three consequences you can rely on:

- **Config ablations are paired.** All four traffic/lights combinations produce
  the *same* map, spawn pose and waypoints for a given seed, so a difference
  between two configurations is a difference between the tasks, not between two
  resamples of the map distribution. Use the same seed list for every arm.
- **Adding a draw in one subsystem does not perturb the others.** A change to the
  traffic spawner cannot move city layouts, so a result stays reproducible across
  a code change that had nothing to do with it.
- **A wrapper cannot move the task.** `np.random` at module scope and
  `env.np_random` are both outside the simulation's streams, so a stochastic
  wrapper or an exploration schedule cannot change which cities you are evaluated
  on.

`reset()` with no seed continues the current streams — normal training. Pass a
seed only when you want a specific episode back.

One caveat: with `randomize_map=False` the layout is **not** resampled per episode,
so it comes from the seed given at *construction*; `reset(seed=s)` then only moves
the spawn and waypoints.

---

## The `info` dict

Returned by both `reset` and `step`; a fresh dict each time, safe to keep.

| key | type | notes |
|---|---|---|
| `reason` | `str \| None` | `success` / `crash` / `stuck` / `timeout`; `None` while running |
| `is_success` | `bool` | `reason == "success"` — what SB3's success-rate logging wants |
| `episode_reward` | `float` | exact running sum of step rewards |
| `step` | `int` | env steps taken this episode |
| `targets_reached` | `int` | waypoints completed |
| `n_targets` | `int` | waypoints in this episode |
| `targets_reached_this_step` | `int` | 0 or 1 in practice |
| `crashed` | `bool` | this step ended in a collision |
| `crash_with` | `str \| None` | `"building"` or `"vehicle"` |
| `red_light_violations` | `int` | cumulative this episode (**not** per step; diff it) |
| `dist_to_target` | `float` | metres to the current waypoint |
| `target_bearing` | `float` | radians, body frame |
| `x`, `y`, `heading`, `speed`, `steer_angle` | `float` | privileged ground truth, **for logging and diagnostics only** |
| `reward_components` | `dict \| absent` | `step`-only; `{"time", "crash", "progress", "target_bonus", "red_light", "pedestrian", "speeding"}`, sums exactly to that step's `reward` -- for logging/dashboards, not present on `reset`'s info |
| `speed_limit` | `float \| None` | active limit in m/s at the car's position when `speed_signs=True`, else `None` |
| `pedestrians_on_road` | `int` | people currently on a zebra (0 unless `pedestrians=True`) |

`reward_components` is purely additive bookkeeping: it does not change the
`reward` scalar, `observation_space`, or any existing `info` key.

`x`, `y` and `heading` are absolute world state and are deliberately *not* in the
observation. Feeding them to a policy defeats the whole design — the map is
resampled every episode precisely so that absolute coordinates carry no
transferable information.

---

## Running many envs

The vector path is pure numpy with no global state, so it parallelises freely.

```python
import carnav
from gymnasium.vector import AsyncVectorEnv
venv = AsyncVectorEnv([carnav.make_factory(traffic=False) for _ in range(16)])
```

`make_factory(**kwargs)` returns a picklable zero-argument builder — picklable by
the **standard library**, not just cloudpickle, so plain `multiprocessing` works
too. A built env also pickles (~260 KB), though passing factories is the right
pattern.

Seed the arms apart: give worker `i` seed `base + i * 10_000` so their episode
streams do not overlap.

Throughput and where the time goes are in the README's Performance section.
`traffic=False` is roughly 2× the step rate of `traffic=True`, and the cost is the
moving-vehicle controller rather than the sensing — so if you are throughput-bound
before you are difficulty-bound, `n_traffic` is the knob.

---

## Image observations

The renderer is **injected, never imported by the simulation**. A headless
vector-observation run loads no graphics stack at all — verified by installing an
import hook that makes `panda3d`, `direct` and `PIL` raise `ImportError`, then
running a full episode: it completes, and `sys.modules` ends with zero graphics
modules in it. So `pip install -e .` with no extras is a working training install.

```python
from render.panda_renderer import PandaRenderer
r = PandaRenderer(offscreen=True, size=64)
env = carnav.make(obs_type="both", renderer=r, image_size=64)
```

Four constraints, all of them load-bearing:

- **One Panda3D `ShowBase` per process.** You cannot run N image envs in one
  process; use subprocess workers, one renderer each. `make_factory` refuses a
  `renderer=` keyword for exactly this reason — a renderer cannot cross a process
  boundary.
- **The buffer size is fixed when the renderer is built.** `capture` raises rather
  than silently rescaling if asked for a different size, so build the renderer at
  the resolution you intend to train on.
- **The camera is stateless** — `capture` is a pure function of car pose, and
  `test_render` asserts byte-identical frames across two runs of a seed. There
  is no camera-smoothing option to accidentally leave on; smoothing would make
  the image depend on history, which breaks both determinism and the Markov
  property the vector state works hard to preserve.
- **There is no HUD, minimap or camera picture-in-picture to suppress.**
  `PandaRenderer` has no human-facing UI at all any more — that job belongs to
  the live browser viewer (`serve/`, `web/`), a separate process watching the
  simulation over a websocket. `capture()` renders exactly what it always did:
  the chase-camera view, nothing composited on top of it.

Each frame is a fresh array (`.copy()` internally), safe to put straight into a
replay buffer.

---

## Notes for world models

This env exists because rendering made DreamerV3 impractical on its 2D
predecessor, so the vector state is built to stand alone. Specifics that matter
for a latent dynamics model:

- **The vector observation is Markov by construction.** `steer_angle` is in it
  because the front wheels lag the command through a rate limiter; relative
  velocity is in it because a single LIDAR scan cannot say whether the car ahead
  is stopped or pulling away. Both were added to close a non-Markov gap, not for
  completeness.
- **Timing.** `dt = 0.05` (20 Hz), `action_repeat = 1` by default, episodes up to
  1000 steps. Raising `action_repeat` shortens the horizon and cuts cost
  proportionally; traffic steps *inside* the repeat loop, on the same clock, so
  vehicles cannot tunnel through the car between collision checks.
- **The genuinely predictable latents** are `traffic_light[6]` (steps until the
  phase changes) and the traffic block's relative velocities. The signal phase is
  a deterministic finite-state machine — a full cycle is 280 steps (14 s), an
  episode sees ~3.6 cycles — so "it will be green before I arrive" is learnable
  from the observation rather than a surprise. That is the cleanest thing in here
  for a latent to roll forward.
- **Stochasticity is concentrated at reset**, not within an episode: the map,
  spawn, waypoints and traffic population are all sampled at `reset`, and the
  dynamics are then deterministic given the actions (with `lidar_noise=0`, the
  default). A model conditioned on a whole episode faces a deterministic system;
  the uncertainty is about *which* city it is in.
- **Observations are fresh arrays**, never views into a reused buffer, so storing
  them in a replay buffer cannot silently alias.
- **Use `obs_slices` for per-block heads or losses.** Reconstruction error on 32
  LIDAR channels will otherwise dominate the 7 traffic-light channels that
  actually carry the decision.

---

## Agents, replay, and structured sensing

Everything above is `CarNavEnv` itself: observation, action, reward, seeding.
None of it changes depending on *how* you decide what action to take. That
split is deliberate — an RL policy, a world-model planner, a hand-written
controller, and a human at the keyboard should all be able to drive this env
through the exact same `reset()`/`step()` calls, with the env never importing
or knowing about any of them.

### The `Agent` interface

`agents/base.py` documents the contract every decision-maker in this repo
follows — `reset()`, `act(obs, info=None)`, and an optional `diagnostics()` for
a HUD. Duck typing is enough; nothing has to subclass `Agent`. It's
deliberately reward-free: `act` never receives one, since some approaches
(rule-based control, MPC over a learned world model, imitation policies at
inference) have no use for it. `env.step()` still returns a reward every
call — an RL training loop reads it from there directly, not through an
Agent.

Built-ins: `agents.load_agent("scripted", env)` (the `baselines.scripted.
GapFollower` reference controller), `"manual"` (`ManualAgent`, keyboard
control), `"random"` (`RandomAgent`, the minimal example — read its ~10 lines
before writing your own). A custom agent is described by a small JSON file
rather than a bare CLI string, since a real one usually needs more than zero
constructor arguments (a checkpoint path, a device, ...):

```json
{
  "module": "my_world_model.agent",
  "factory": "PlannerAgent",
  "kwargs": {"checkpoint": "runs/ckpt_500.pt", "device": "cpu"}
}
```

`factory` is looked up on `module` and always called as `factory(env,
**kwargs)` — a class or a function both work, as long as `env` is the first
positional argument (even if unused). `python main.py serve --agent
configs/my_agent.json` and `python main.py shots --agent ...` both take the
same spec.

### Replay buffers, decoupled from the env

`replay/` (`ReplayBuffer`, `EpisodeBuffer`, `RecordingWrapper`) is a second,
independent piece of "outside the env" infrastructure — `env/nav_env.py` has
zero lines of replay-related code; a training codebase uses these directly
against ordinary `env.step()` calls:

```python
import carnav
from replay import ReplayBuffer, RecordingWrapper

env = RecordingWrapper(carnav.make(), ReplayBuffer(capacity=200_000))
obs, info = env.reset()
...                                    # buffer fills itself as you step
batch = env.buffer.sample(256)
```

`ReplayBuffer` is a ring buffer of raw transitions with uniform sampling —
the off-policy (DQN/SAC-style) case. `EpisodeBuffer` stores whole episodes and
samples contiguous multi-step windows (`sample_sequences(batch_size,
length)`) — what a sequence/world model actually needs for BPTT, which
`ReplayBuffer`'s i.i.d. sampling can't give you. Both work unchanged whether
`obs` is the vector, the image, or (`obs_type="both"`) a dict of both, and
both round-trip through `save(path)`/`load(path)` (stdlib `pickle`, no new
dependency) for the collect-now-train-later-in-a-different-process workflow.
`RecordingWrapper` is sugar for calling `buffer.add(...)` by hand every step;
use whichever fits your loop.

**Banking frames for later, without training on them now.** Both buffer
classes take an eighth, optional field on `add(..., frame=None)`, separate
from `obs`/`next_obs`, precisely so it never has to change what a policy
trains on:

```python
import carnav
from replay import ReplayBuffer, RecordingWrapper

env = carnav.make(render_mode="rgb_array")             # obs_type stays "vector"
env = RecordingWrapper(env, ReplayBuffer(200_000), capture_frames=True)
obs, info = env.reset()                                # obs is still the vector
...
batch = env.buffer.sample(256)
batch["obs"]      # (256, vector_dim) -- what a vector-only policy trains on today
batch["frame"]    # (256, H, W, 3) uint8 -- banked in case an image experiment wants it later
```

`capture_frames=True` calls `env.render()` every step and stores the result
in `frame`, which needs `render_mode="rgb_array"` set when the env was built
(`RecordingWrapper` raises immediately if it isn't, rather than silently
storing `None` for every step). Leave `capture_frames` at its default `False`
and this costs nothing: no renderer is built, no frame is captured, and
`sample()`/`sample_sequences()` hand back `batch["frame"] is None` rather
than a key you have to work around. `env/nav_env.py` has zero lines of
frame-capture code either way — `render()` (used here) and `image_obs()`
(used by `obs_type="image"`/`"both"`) were already both there; this just
calls the existing one from outside instead of adding a new mechanism.
Checked in `tests/test_replay.py` §6.

### `env.perception` — structured camera sensing (auxiliary, like `env.radar`)

`env/perception.py` adds occlusion-aware front/left/right/rear camera
detections and left/right blind-spot zones, computed every `reset()`/`step()`
and exposed as `env.perception` — same discipline as `env.radar` (see above):
never concatenated into the observation vector, gated by `EnvConfig(
perception=True)`, no RNG stream. `env.perception.cameras[name]` is a list of
`(vehicle_index, distance_m, bearing_rad)` tuples for whichever *other
vehicles* (not buildings — see the module docstring for why) are in that
camera's field of view and range and not hidden behind another vehicle;
`env.perception.blind_spots["left"/"right"]` is `{"occupied": bool,
"vehicle": index}`. This is what the live viewer's camera-feed badges, FOV
wedges, and detection outlines actually read — a real answer to "what does
the car's camera see", not a rendering approximation.

---

## Visualising a world model's imagination

Not every method plans by imagining, and this does not try to generalise
across all the ones that do — it standardises on the one thing that is
renderable regardless of what is underneath it: a sequence of predicted
future `(x, y)` positions. DreamerV3 rolling out its latent dynamics and
decoding to a position is the motivating case; anything else that can produce
candidate future poses (an MPC planner's rollouts, a sampling-based planner's
particles) qualifies the same way. Agents that never imagine anything — the
large majority — simply never set this and are completely unaffected.

Add an `imagined_trajectories` key to `Agent.diagnostics()`:

```python
[{"x": [x1, x2, ...], "y": [y1, y2, ...]}, ...]   # one dict per rollout
```

- **Absolute world-frame metres**, not ego-frame — the same frame `info["x"]`/
  `info["y"]` already hand the agent every step, so anchoring an imagined
  rollout to the current pose needs no extra bookkeeping and no coordinate
  transform the viewer would have to reverse.
- **Step order starting from the *next* predicted step**, not the current
  pose (the viewer already knows where the car is).
- **Any number of rollouts.** One dict for a single mean/expected trajectory,
  or several for stochastic samples — Dreamer's latent transitions are
  stochastic, so a handful of samples is the honest way to show what the
  model actually thinks might happen, not just its mean. The viewer fans them
  out as separate lines and lowers per-line opacity as the count grows, so
  many samples read as a spread rather than a solid blob.
- **Any horizon length**, and it may change between calls — nothing here
  requires a fixed planning horizon.
- **Omit the key entirely** (`{}`, the `diagnostics()` default) when there is
  nothing to imagine. This is the common case; the viewer treats its absence
  as "nothing to draw," never as an error.

Worked example — decoding a world model's imagined latents into a handful of
diagnostic rollouts, alongside whatever action the model actually chose:

```python
import numpy as np

class DreamerAgent:
    def __init__(self, env, checkpoint, n_samples=3, horizon=16):
        self.model = load_world_model(checkpoint)
        self.n_samples, self.horizon = n_samples, horizon
        self.dt = env.cfg.dt
        self._latent = None
        self._pose = None                      # (x, y, heading) anchor for this step

    def reset(self):
        self._latent = self.model.initial_state()

    def act(self, obs, info=None):
        self._pose = (info["x"], info["y"], info["heading"]) if info else (0.0, 0.0, 0.0)
        self._latent, action = self.model.step(self._latent, obs)
        return action

    def diagnostics(self):
        # Roll the model's dynamics forward from the current latent, `n_samples`
        # times, and decode each imagined state to a position. However this
        # decoding works internally, the output handed to the viewer is always
        # just world-frame (x, y) -- the viewer never sees a latent.
        rollouts = self.model.imagine(self._latent, self.horizon, n=self.n_samples)
        x0, y0, heading0 = self._pose
        trajs = []
        for rollout in rollouts:                # rollout: (horizon, 2) local (dx, dy) offsets
            xs, ys = [], []
            x, y = x0, y0
            for dx, dy in rollout:
                x += dx * np.cos(heading0) - dy * np.sin(heading0)
                y += dx * np.sin(heading0) + dy * np.cos(heading0)
                xs.append(float(x)); ys.append(float(y))
            trajs.append({"x": xs, "y": ys})
        return {"imagined_trajectories": trajs}
```

Point `main.py serve` at it (`--agent configs/dreamer.json`, the same JSON-
config mechanism every custom agent uses) and the rollouts show up
immediately: a trail of small violet dots ahead of the car, shrinking and
fading toward the end of the horizon (dots rather than a line so it stays
legible viewed nearly end-on, and reads as distinct from the pulsing LIDAR
wave around the car), plus an `imagining ×N · H steps` chip next to the agent's
name in the driver panel — both gated behind the existing `sensors` overlay
toggle, so turning overlays off hides them along with LIDAR and detection
boxes. No server-side code has to change for a new
agent to use this: `serve/server.py` forwards whatever `diagnostics()`
returns without inspecting its shape (see `agent_diagnostics()` and `_py()`'s
`np.ndarray` handling in `serve/server.py` — a bare numpy array in the dict is
fine, it does not have to be pre-converted to a list). Checked in
`tests/test_viewer.py` §6.

---

## Things that will bite you

1. **The zero action *is* coasting** -- zero throttle, zero brake. (This
   flipped when throttle became signed: it used to be half throttle and half
   brake, with `[-1, -1, 0]` for coasting; check any code written against the
   old convention.)
2. **`stuck` is `terminated`, not `truncated`,** and its lump-sum penalty is only
   return-neutral undiscounted. Set `stuck_steps=0` if that shaping is in your way.
3. **Do not wrap in `TimeLimit`.** The env has its own limit and reports it as
   `truncated`.
4. **`red_light_violations` is cumulative.** Diff consecutive steps for a per-step
   signal.
5. **Never put `info["x"]`, `["y"]` or `["heading"]` in an observation.** They are
   absolute world state; the map is resampled every episode specifically so those
   carry nothing transferable.
6. **Install the `gym` extra if your library needs `isinstance(env, gym.Env)`.**
   Without gymnasium the env falls back to a plain object with a `Box` shim.
7. **One `ShowBase` per process** for image observations — parallelism must be
   process-level.
8. **Scale the reward, not the observation.** The vector is already in [-1, 1];
   the reward has ±100 spikes against a typical −0.1.
9. **Keep configuration in one place.** `carnav.make` raises on unknown
   keywords; hand-building `EnvConfig` + `CityConfig` + `CarParams` does not.
10. **`traffic=False` and `traffic_lights=False` remove the simulation entity, the
    observation block and the rendered geometry together.** There is no way to
    have an agent penalised for a signal it cannot see.
11. **`info["episode_reward"]` is not your number under `RewardOverrideWrapper`.**
    It is the env's own running sum of its own formula, computed with no idea the
    wrapper exists. Track your own running return if you override the reward.
