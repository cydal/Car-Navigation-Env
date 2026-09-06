# rl-env3d — a 3D car navigation environment for RL and world models

A procedurally generated city that a car has to navigate through a sequence of
waypoints. Built as a successor to the PNG-bitmap 2D environment in `../RL-t3d`,
with the specific goal of supporting **both** vector and image observations so
world-model experiments are not forced to choose.

```
python main.py spec     # observation / action contract
python main.py demo     # live window, scripted driver, LIDAR overlay
python main.py bench    # throughput, vector vs image
python main.py shots    # save a grid of 64x64 agent-view frames
```

## Why this exists

The old environment gave a 9-dimensional state: seven binary "is there road at
this one pixel" sensors, plus a target angle and distance. That is enough to
learn *something*, but it cannot express how far away a wall is, so a policy can
only ever react once it is already on top of one. Its collision test was a single
centre pixel, so the car could clip a building corner and drive on.

The design brief here was: **the vector observation must be dense enough to
replace the image entirely**, because rendering was what made DreamerV3
impractical on the old env — while still being able to render properly when you
want pixels.

Both paths are live, and the cost of each is measured (see Performance).

## Layout

| file | role |
|---|---|
| `env/world.py` | procedural city grid, ray casting, collision, spawn sampling |
| `env/car.py` | kinematic bicycle model with a rate-limited steering actuator |
| `env/sensors.py` | 360° LIDAR, ego-centric nav and proprioception features |
| `env/nav_env.py` | Gymnasium-style env: observations, reward, termination |
| `render/panda_renderer.py` | Panda3D third-person renderer (offscreen or windowed) |
| `baselines/scripted.py` | follow-the-gap + pure-pursuit driver, obs-only |
| `tests/` | core, env and render suites, plus two diagnostic scripts |

The renderer is **injected**, never imported by the simulation. Headless training
never loads a graphics stack, and `test_core` / `test_env` run without panda3d
installed at all. Gymnasium is optional too — there is a small `Box` shim so the
env works without it.

## Observation

`obs_type` selects `"vector"`, `"image"` or `"both"`.

**Vector — 46 dimensions, all in [-1, 1], entirely ego-centric:**

| block | dims | contents |
|---|---|---|
| LIDAR | 32 | full 360°, 60 m range, normalised. Beam `n/2` is straight ahead, so the array's wrap-around discontinuity sits at the rear where it carries least information. |
| dynamics | 5 | speed, yaw rate, **steer angle**, acceleration, slip angle |
| navigation | 9 | next 3 waypoints × (distance, sin, cos) of bearing |

There is no absolute position or heading anywhere in the observation, which is
what lets a policy trained on one procedural layout transfer to the next instead
of memorising coordinates.

Three details that matter more than they look:

- **`steer_angle` is included** because the front wheels lag the command through
  a rate-limited actuator. Without it the observation is not Markov, and the
  agent cannot learn to anticipate turn-in delay.
- **Angles are sin/cos pairs**, never raw radians, so the network never sees the
  discontinuity at ±π.
- **Absent waypoints are padded with `sin = cos = 0`** — not a valid unit vector,
  so "no waypoint" is unambiguously distinguishable from every real bearing.

**Image** — 64×64×3 uint8 by default, third-person chase camera.

## Action

`Box(-1, 1, (3,))` → throttle, brake, steer. Throttle and brake are rescaled to
[0, 1]; steer is a *command*, not an angle, and passes through the rate limiter.
Braking cannot reverse the car.

## Reward

`progress × 1.0` toward the current waypoint, `−0.1` per step, `+100` per
waypoint reached, `−100` for a crash.

Early termination on "stuck" charges the *remaining* time penalty as a lump sum,
so ending an episode early is reward-neutral. Otherwise stopping dead would be a
cheap way to escape the per-step cost, and the agent would learn to park.

## Performance

Measured on this machine, 48×48 tile map, 32 beams:

| | µs/step | steps/s |
|---|---|---|
| vector only | ~205 | **~4,900** |
| vector + 64×64 image | ~565 | **~1,750** |

Ray casting is the bulk of the vector step (~158 µs of ~205 µs). Map generation
is 1.2 ms per reset and rebuilding the render mesh is 2.9 ms.

## Design decisions worth knowing

These were all forced by measurement rather than chosen up front.

**Corridor width is set by the car's turn radius.** `road_width=3` (12 m), not 2
(8 m). The minimum turn radius is 4 m, so a U-turn needs about 9.9 m of width
including the body. At 8 m, any episode whose target spawned behind the car was
geometrically unwinnable. Set it to 2 for a deliberately much harder env.

**Spawn poses must be able to drive away, not merely not collide.** The heading
is aligned with the roomiest probed direction and then jittered — but the probe
is a centre-line ray that ignores the car's 1.9 m width, and the jitter is
applied *after* it. So the accepted pose is additionally required to have 6.6 m
of *swept* forward room. Before this check, 1% of episodes crashed within 1 m of
the spawn no matter what the policy did, which poisons training with
unavoidable negative returns.

**The fast ray caster is corner-safe.** Sampling a beam every 0.25 m is 2.7×
faster than exact DDA traversal, but a beam grazing a convex building corner
crosses only a few centimetres of that tile, so no sample lands inside and the
wall is missed entirely — 0.2% of beams reported ~20 m of clearance where the
true range was 6 m. Since the tile is a full building, a car driving down that
beam crashes; the sensor would have been punishing the agent for trusting it.
The sampler now identifies the skipped cell the same way DDA does, by comparing
distance-to-vertical-boundary against distance-to-horizontal. Agreement with
exact DDA is within ±step/2 over 57,600 rays, asserted in `test_core`.

**Collision is an oriented bounding box**, 4 corners + 4 edge midpoints. On 4,000
poses whose centre sits on road, the OBB catches ~1,000 that a centre-point test
would have driven straight through.

**The chase camera is stateless.** No smoothing, so `capture` is a pure function
of car pose. Smoothing looks better but makes the image depend on history, which
silently breaks both determinism under a fixed seed and the Markov property the
vector state works hard to preserve. `--smooth` exists for watching, not
training; `test_render` asserts byte-identical frames across two runs of a seed.

**The road is drawn as one quad per tile, not one big plane.** A featureless
plane is cheaper but gives a vision policy almost no optical flow, which makes
speed nearly unobservable from pixels — exactly the signal an image-based world
model needs. Per-tile shading costs ~1k extra quads.

**Every road tile is reachable.** Generation flood-fills, keeps the largest
connected component and walls off the rest, so a sampled target is never
unreachable from a sampled spawn.

## Baseline

`baselines/scripted.py` drives using **only the observation vector** — no
privileged access to the world, the car object or the target list. That makes it
a test of the observation design itself: if classical control can drive from
these 46 numbers, the vector state carries enough to learn from, and an RL agent
that underperforms it has a learning problem rather than a sensing one.

It combines follow-the-gap heading selection, pure-pursuit steering, corridor
centering from left/right LIDAR asymmetry, and braking-distance speed control.

Over 40 episodes of 3 waypoints each:

| policy | mean reward | waypoints |
|---|---|---|
| random | −100 | 0.00 / 3 |
| scripted | **+214** | **1.85 / 3** |

Full-route success ~42–45%. `test_env` asserts this stays inside a 30–90% band:
below that the task is broken, above it there is no headroom left for a learned
policy to show anything.

Two lessons from tuning it that apply to reward shaping generally:

- Scoring clearance *linearly* made a long open street outscore the turn the car
  actually needed, so it sailed straight past its waypoint. Clearance has to
  **saturate** — past "comfortably enough room", extra distance is worthless.
- A proportional gain on heading error commanded near-full lock for a 10°
  correction and oscillated. Pure-pursuit geometry scales the response to how far
  away the aim point is, which is what actually holds a line.

## Tests

```
python tests/test_core.py     # city, ray casters vs brute force, car model, OBB
python tests/test_env.py      # obs contract, determinism, reward, baseline vs random
python tests/test_render.py   # image obs, render determinism, overlay, cost
python tests/diagnose.py      # single-episode step trace + ASCII trajectory map
python tests/diagnose_crash.py# last 14 steps before each crash, failure-mode tally
```

The two `diagnose` scripts earned their place: every controller bug listed above
was found by dumping state before a crash, not by reasoning about the code.

## Roadmap

The task is meant to be enriched in place, reusing the same observation layout:

1. **now** — reach a sequence of waypoints
2. collectible items (reward shaping with optional detours)
3. static parked cars — `Lidar.scan()` already accepts an `obstacles` array of
   circles and `_ray_circle_ranges` is written and vectorised, just unused
4. rule-based moving traffic
5. traffic lights and right-of-way
6. multi-agent

`tasks/` is reserved for pulling reward and termination out of `nav_env.py` once
there is more than one task to share them.
