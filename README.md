# rl-env3d — a 3D car navigation environment for RL and world models

A procedurally generated city that a car has to navigate through a sequence of
waypoints. Built as a successor to the PNG-bitmap 2D environment in `../RL-t3d`,
with the specific goal of supporting **both** vector and image observations so
world-model experiments are not forced to choose.

```
python main.py spec           # observation / action contract
python main.py demo           # live window, scripted driver, LIDAR overlay
python main.py demo --keys    # same but drive it yourself (arrow keys + brake)
python main.py bench          # throughput, vector vs image
python main.py shots          # save a grid of 64x64 agent-view frames
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

| path | role |
|---|---|
| `env/world.py` | procedural city grid, ray casting, collision, spawn sampling, signal placement |
| `env/car.py` | kinematic bicycle model with a rate-limited steering actuator |
| `env/sensors.py` | 360° LIDAR, ego-centric nav and proprioception features |
| `env/nav_env.py` | Gymnasium-style env: observations, reward, termination, traffic-light state |
| `render/panda_renderer.py` | Panda3D third-person renderer (offscreen or windowed) |
| `baselines/scripted.py` | follow-the-gap + pure-pursuit driver, obs-only |
| `tests/` | core, env and render suites, plus two diagnostic scripts |
| `kenney_car-kit/` | CC0 low-poly vehicle assets (Kenney.nl); only used by the renderer |

The renderer is **injected**, never imported by the simulation. Headless training
never loads a graphics stack, and `test_core` / `test_env` run without panda3d
installed at all. Gymnasium is optional too — there is a small `Box` shim so the
env works without it.

## Observation

`obs_type` selects `"vector"`, `"image"` or `"both"`.

**Vector — 53 dimensions, all in [-1, 1], entirely ego-centric:**

| block | dims | contents |
|---|---|---|
| LIDAR | 32 | full 360°, 60 m range, normalised. Beam `n/2` is straight ahead, so the array's wrap-around discontinuity sits at the rear where it carries least information. |
| dynamics | 5 | speed, yaw rate, **steer angle**, acceleration, slip angle |
| navigation | 9 | next 3 waypoints × (distance, sin, cos) of bearing |
| traffic light | 7 | nearest signal: distance **to the stop line**, sin/cos bearing, red/yellow/green for *this car's approach axis*, steps until the phase changes |

Set `EnvConfig(traffic_lights=False)` to drop the last block and get the original
46-D vector; that switch also disables the red-light penalty, because penalising
a signal the agent cannot see is not a rule it can learn.

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
waypoint reached, `−100` for a crash, `−50` for entering an intersection against
a red.

Early termination on "stuck" charges the *remaining* time penalty as a lump sum,
so ending an episode early is reward-neutral. Otherwise stopping dead would be a
cheap way to escape the per-step cost, and the agent would learn to park.

The red-light penalty fires **on entry only, never per step of occupancy**. A
light that turns red while you are already in the box is not a violation, and
charging per step would make a single mistake unboundedly expensive — the agent
would learn to fear intersections rather than to read signals.

Which signal you are judged against is decided by **the stop line you crossed**,
not by where you were pointing (see Design decisions).

## Visualisation

`main.py demo` opens a Panda3D window with three layers:

**3D chase camera** — third-person view following the car.  The car mesh is a
Kenney CC0 sedan scaled to match the physics car (≈ 4.4 m long).  The green
beacon posts mark the next waypoints; the brightest one is the current target.
Fog hides the map boundary and gives depth cues.

**Traffic signals** — six crossings per map carry lights, and each signalised
crossing gets **four separate signal heads, one per approach**, hung from a mast
arm that reaches out over the corridor so the head sits above the stop line
facing oncoming traffic. A single pole on one corner (the first attempt) shows
its edge or its back from most approaches and cannot be read at all from two of
them, which made the light unreadable exactly when it mattered. The lens stands
proud of its backplate so the driver sees a full lit face rather than a sliver.

**Minimap (top-right corner)** — a top-down view of the full city.  Buildings
are warm grey, road is dark blue-grey.  Elements:

| element | meaning |
|---|---|
| orange triangle | car — tip = nose, tail = rear |
| bright green square | current target waypoint |
| yellow squares | upcoming waypoints (in order) |

**Proximity ring (3D, default on in windowed mode)** — a ring of radius 3 m
around the car in the 3D scene.  Each segment is coloured by the LIDAR
distance in that direction: **green** = clear, **red** = obstacle close.
This gives an at-a-glance danger signal without cluttering the minimap.
The LIDAR ray lines are reserved for future use (e.g. visualising a world
model's imagined scene in a separate overlay).  Set `show_rays=False` to
suppress the ring; it is off by default in offscreen (training) mode so it
never enters image observations.

**Keyboard controls** (`--keys` flag):

| key | action |
|---|---|
| ↑ arrow | throttle |
| ↓ arrow | brake |
| ← / → arrow | steer left / right |

The scripted driver runs when `--keys` is not set.  Both use the same 53-D
observation vector — no privileged simulator access.

**Offscreen vs windowed** — when `offscreen=True` (training), the minimap
overlay is suppressed so it never enters the image observation.

## Performance

Measured on this machine, 48×48 tile map, 32 beams:

| | µs/step | steps/s |
|---|---|---|
| vector only | ~205 | **~4,900** |
| vector + 64×64 image | ~565 | **~1,750** |

Ray casting is the bulk of the vector step (~158 µs of ~205 µs). Map generation
is ~2.5 ms per reset and rebuilding the render mesh is ~5.5 ms, rising to ~8.2 ms
with signal geometry.

**Traffic lights cost the vector path nothing measurable**: 239 µs/step with them
off versus 233 µs/step with them on, i.e. inside run-to-run noise. This is
checked rather than assumed, because the whole premise of the env is that the
vector path stays fast enough to train on — a throughput regression there is a
correctness problem, not a performance nit. The signal *geometry* costs ~2.7 ms
per reset, but only when a renderer is attached, and it is skipped entirely when
`traffic_lights=False`.

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

**Right-of-way is decided by the stop line crossed, not by heading.** The obvious
implementation asks which way the car is pointing (`|sin h| > |cos h|`) and reads
the N-S or E-W phase accordingly. On a 12 m undivided corridor with no lane
markings and no drive-on-side convention there is no well-defined "lane", so a
car cutting a corner sits near 45° and that test becomes a coin flip: measured at
**7% of real entries, with 11% within 20° of the boundary**. A penalty that is
noise on 7% of the events it fires for is not a rule anything can learn. The
axis is now taken from *which stop line the car crossed* — whether it was
previously outside the box in x or in y — which is unambiguous and is fixed at
exactly the moment the penalty is evaluated. The two rules disagree on 8% of
entries, and the observation now agrees with the reward on **100% of measured
violations**.

**Yellow is as long as it takes to stop, not as long as looks right.** Stopping
from the 10 m/s cruise speed at the controller's 6 m/s² takes 1.67 s and 8.3 m.
Yellow was originally 12 steps (0.6 s) — 2.8× too short — which means the warning
was one the vehicle *physically could not honour*, so yellow was decoration and
every driver arriving on one was guaranteed a −50. It is now 40 steps (2.0 s).
Timings in an RL env are set by the vehicle's physics, not by realism.

**Waiting at a red is exempt from the stuck detector.** A red lasts
`GREEN + YELLOW` steps, so without the exemption the signal cycle and the
stuck timeout are silently coupled and a long enough cycle terminates the episode
**for obeying the law**. The original settings had only 58 steps of margin — not
a bug yet, but one retune away from being one.

**Only six crossings per map are signalised**, chosen greedily farthest-first out
of the ~31 true 4-way junctions. Every 4-way sits ~29 m from the next, so
lighting them all against a 14 s cycle makes a drive stop-go every 3 s with no
rhythm to learn — constant friction rather than a skill. Sparse placement raises
the mean spacing to ~72 m and makes each signal an event.

**Only true 4-way crossings qualify.** Junctions where `_block_segments` walled
off an arm are T-junctions or dead ends, which do not warrant a light; requiring
all four arms be road cut the candidates from 64 to ~31.

## Baseline

`baselines/scripted.py` drives using **only the observation vector** — no
privileged access to the world, the car object or the target list. That makes it
a test of the observation design itself: if classical control can drive from
these 53 numbers, the vector state carries enough to learn from, and an RL agent
that underperforms it has a learning problem rather than a sensing one.

It combines follow-the-gap heading selection, pure-pursuit steering, corridor
centering from left/right LIDAR asymmetry, and braking-distance speed control.

Over 160 episodes of 3 waypoints each:

| policy | mean reward | waypoints | red-light violations |
|---|---|---|---|
| random | −103 | 0.00 / 3 | 0.05 |
| scripted | **+213** | **1.93 / 3** | 0.56 |

Full-route success ~49%; 40-episode blocks range from 32% to 55%, so quote
pooled runs rather than single blocks. `test_env` asserts this stays inside a
30–90% band: below that the task is broken, above it there is no headroom left
for a learned policy to show anything.

**The baseline had to be taught to stop.** It has a `min_speed = 2.0` floor to
stop it dithering in a corridor, which meant it structurally *could not* hold at
a red — it was paying ~1.7 violations per episode, about −84, for a manoeuvre it
was incapable of performing. A benchmark that a rule-following agent beats only
by breaking rules is not measuring the task, so the red-light case now bypasses
the floor; violations fell to 0.56 per episode with zero stuck terminations. The
rest is control imperfection, which is exactly the headroom a learned policy
should be able to claim. Turning lights off scores +257 against +213, so the
signals cost the baseline ~44 per episode and are a real part of the task rather
than a decoration.

The controller gates on **the traffic-light slice being present in the
observation**, not on a constructor flag, so it adapts to an env with or without
lights without retuning.

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

1. **done** — reach a sequence of waypoints
2. **done** — traffic lights and right-of-way (`traffic_lights=False` restores
   the original 46-D task)
3. collectible items (reward shaping with optional detours)
4. static parked cars — `Lidar.scan()` already accepts an `obstacles` array of
   circles and `_ray_circle_ranges` is written and vectorised, just unused
5. rule-based moving traffic
6. multi-agent

`tasks/` is reserved for pulling reward and termination out of `nav_env.py` once
there is more than one task to share them.
