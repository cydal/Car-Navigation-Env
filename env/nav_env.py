"""
CarNavEnv -- Gymnasium-compatible car navigation on a procedural city.

Design notes
------------
* The renderer is *injected*, never imported here. Simulation runs at full speed
  with no graphics stack loaded unless you ask for image observations.
* Observations are ego-centric and the map is resampled per episode, so the only
  way to score well is to actually learn driving + navigation rather than to
  memorise one layout.
* `obs_type` selects 'vector', 'image' or 'both'. The vector state is built to be
  dense enough to stand alone, so world-model experiments can drop the image
  path when rendering becomes the bottleneck.
"""

import copy

import numpy as np

from .world import ProceduralCity, CityConfig
from .car import Car, CarParams
from .sensors import Lidar, Radar, nav_features, dynamics_features, relative_bearing
from .perception import Perception
from .crossings import Street, PED_FEATURES, SIGN_FEATURES
from .traffic import Traffic, TRAFFIC_FEATURES

try:                                    # gymnasium is optional
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
    _HAS_GYM = True
except ImportError:                     # pragma: no cover
    gym, gym_spaces = None, None
    _HAS_GYM = False


class _Box:
    """Minimal Box stand-in so the env is usable without gymnasium installed.

    It carries its own generator rather than drawing from the global
    `np.random`, because `action_space.sample()` is what a random-policy
    reference run uses and an unseeded global makes that run unreproducible --
    the one property the rest of this env works hardest to guarantee.
    """

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def sample(self):
        if np.issubdtype(self.dtype, np.integer):
            return self._rng.integers(self.low, np.int64(self.high) + 1,
                                      size=self.shape).astype(self.dtype)
        return self._rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return (x.shape == self.shape and np.all(x >= self.low - 1e-5)
                and np.all(x <= self.high + 1e-5))

    __contains__ = contains

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.__name__})"


def _box(low, high, shape, dtype=np.float32):
    if _HAS_GYM:
        return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _Box(low, high, shape, dtype)


_BaseEnv = gym.Env if _HAS_GYM else object


TL_FEATURES = 7   # per signal exposed in the observation; see _tl_features


class TrafficLight:
    """Two-phase signal at one crossing: N-S green, then E-W green, with yellows.

    Timings are set by the car's braking physics, not by realism. Yellow must be
    at least as long as it takes to stop from cruising speed (10 m/s at the
    controller's 6 m/s^2 is 1.67 s), or the warning is one a vehicle physically
    cannot honour and yellow is decoration. 40 steps at dt=0.05 is 2.0 s.

    Green is deliberately short for an RL env: a full cycle is 280 steps = 14 s,
    so a 1000-step episode sees ~3.6 cycles and the agent actually encounters
    every phase instead of meeting one light state per episode.
    """

    GREEN_STEPS  = 100  # 5.0 s at dt=0.05
    YELLOW_STEPS = 40   # 2.0 s -- exceeds the 1.67 s stop from cruise speed

    # phase 0 = NS green, 1 = NS yellow, 2 = EW green, 3 = EW yellow
    _NS = ('green', 'yellow', 'red', 'red')
    _EW = ('red', 'red', 'green', 'yellow')

    def __init__(self, x, y, phase=0, timer=0):
        self.x = x
        self.y = y
        self.phase = phase
        self.timer = timer

    @property
    def phase_limit(self):
        return self.YELLOW_STEPS if self.phase % 2 == 1 else self.GREEN_STEPS

    def tick(self):
        self.timer += 1
        if self.timer >= self.phase_limit:
            self.phase = (self.phase + 1) % 4
            self.timer = 0

    def state(self, axis):
        """Signal shown to traffic travelling along `axis` ('ns' or 'ew')."""
        return (self._NS if axis == 'ns' else self._EW)[self.phase]

    @property
    def steps_remaining(self):
        """Steps until this phase ends -- what makes the light anticipatable."""
        return self.phase_limit - self.timer

    @property
    def ns_state(self):
        return self._NS[self.phase]

    @property
    def ew_state(self):
        return self._EW[self.phase]


class EnvConfig:
    """Task and episode settings, kept separate from world/vehicle config."""

    def __init__(
        self,
        n_targets=3,
        n_lookahead=3,
        target_radius=6.0,
        target_min_dist=40.0,
        target_max_dist=110.0,
        nav_range=150.0,
        n_beams=32,
        lidar_range=60.0,
        lidar_noise=0.0,
        dt=0.05,
        action_repeat=1,
        max_episode_steps=1000,
        randomize_map=True,
        # --- reward shaping
        time_penalty=0.1,
        progress_weight=1.0,
        crash_penalty=100.0,
        target_bonus=100.0,
        stuck_speed=0.5,
        stuck_steps=150,
        # --- traffic lights
        traffic_lights=True,
        red_light_penalty=50.0,
        n_tl_obs=None,          # None = 1 when lights are on, else 0
        tl_obs_range=60.0,
        # --- traffic
        traffic=True,
        n_traffic=8,            # vehicles driving the grid
        n_parked=18,            # vehicles parked at the kerb
        traffic_speed=9.0,      # cruise speed, jittered per vehicle
        n_traffic_obs=None,     # None = 4 when traffic is on, else 0
        traffic_obs_range=60.0,
        # --- radar (auxiliary sensor; never enters the observation vector)
        radar=True,
        n_radar_sectors=8,
        radar_range=60.0,
        radar_noise=0.0,
        # --- perception (auxiliary sensor; never enters the observation vector)
        perception=True,
        # --- pedestrians at zebra crossings (opt-in; appends a 5*n_ped_obs block)
        pedestrians=False,
        n_crossings=6,
        crossing_min_sep=40.0,
        crossing_offset=2.0,        # metres beyond the junction box edge
        peds_per_crossing=(1, 3),
        n_ped_obs=None,             # None = 4 when pedestrians are on, else 0
        ped_obs_range=60.0,
        pedestrian_penalty=300.0,
        # --- speed signs (opt-in; appends a 3-wide block)
        speed_signs=False,
        speed_limit_kmh=50.0,
        zone_limit_kmh=30.0,        # within zone_len of a crossing
        zone_len=30.0,
        sign_range=60.0,
        speeding_penalty=0.05,      # per m/s over the limit, per physics step
    ):
        self.n_targets = n_targets              # waypoints per episode
        self.n_lookahead = n_lookahead          # waypoints exposed in the observation
        self.target_radius = target_radius      # metres to count as reached
        self.target_min_dist = target_min_dist  # waypoint spacing (annulus)
        self.target_max_dist = target_max_dist
        self.nav_range = nav_range              # normaliser for target distance
        self.n_beams = n_beams
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        self.dt = dt                            # physics timestep (20 Hz default)
        self.action_repeat = action_repeat
        self.max_episode_steps = max_episode_steps
        self.randomize_map = randomize_map      # new layout every reset
        self.time_penalty = time_penalty
        self.progress_weight = progress_weight
        self.crash_penalty = crash_penalty
        self.target_bonus = target_bonus
        self.stuck_speed = stuck_speed
        self.stuck_steps = stuck_steps
        # Penalising a light the agent cannot see would be unlearnable, so the
        # penalty and the observation block are driven by one switch.
        self.traffic_lights = traffic_lights
        self.red_light_penalty = red_light_penalty
        self.n_tl_obs = (1 if traffic_lights else 0) if n_tl_obs is None else n_tl_obs
        self.tl_obs_range = tl_obs_range
        # Same one-switch discipline as the lights: `traffic=False` removes the
        # vehicles, the observation block and the rendered meshes together, so the
        # agent is never asked about something it cannot see or crash into.
        self.traffic = traffic
        self.n_traffic = n_traffic if traffic else 0
        self.n_parked = n_parked if traffic else 0
        self.traffic_speed = traffic_speed
        self.n_traffic_obs = (4 if traffic else 0) if n_traffic_obs is None else n_traffic_obs
        self.traffic_obs_range = traffic_obs_range
        # Radar is purely auxiliary (HUD/telemetry), so unlike the lights/traffic
        # one-switch discipline above it has no observation-block width to keep in
        # sync -- turning it off just stops the renderer/HUD from reading it.
        self.radar = radar
        self.n_radar_sectors = n_radar_sectors
        self.radar_range = radar_range
        self.radar_noise = radar_noise
        # Same auxiliary discipline as radar: no noise model, no RNG stream,
        # no observation-block width -- turning it off just stops the
        # renderer/HUD from reading it.
        self.perception = perception
        # Pedestrians and signs follow the lights/traffic one-switch rule: each
        # flag adds its people/signs, its penalty and its observation block
        # together. Both are off by default so the 73-D contract, its tests and
        # the published baseline numbers are exactly what they were.
        self.pedestrians = pedestrians
        self.n_crossings = n_crossings if (pedestrians or speed_signs) else 0
        self.crossing_min_sep = crossing_min_sep
        self.crossing_offset = crossing_offset
        self.peds_per_crossing = tuple(peds_per_crossing)
        self.n_ped_obs = (4 if pedestrians else 0) if n_ped_obs is None else n_ped_obs
        self.ped_obs_range = ped_obs_range
        self.pedestrian_penalty = pedestrian_penalty
        self.speed_signs = speed_signs
        self.n_sign_obs = 1 if speed_signs else 0
        self.speed_limit_kmh = speed_limit_kmh
        self.zone_limit_kmh = zone_limit_kmh
        self.zone_len = zone_len
        self.sign_range = sign_range
        self.speeding_penalty = speeding_penalty


class CarNavEnv(_BaseEnv):
    """Drive a car through a sequence of waypoints in a procedural city.

    Action (3,) in [-1, 1]:
        0  throttle  signed: positive drives forward, negative reverses
        1  brake     rescaled to [0, 1]; opposes current motion, never itself reverses the car
        2  steer     steering command, scaled to the steering lock

    Throttle and brake are separate channels (matching real vehicle interfaces)
    even though the current kinematic model lets them partially cancel; they
    become independent once a load-transfer model is added.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        config=None,
        city_config=None,
        car_params=None,
        obs_type="vector",
        renderer=None,
        image_size=64,
        render_mode=None,
        seed=None,
    ):
        if obs_type not in ("vector", "image", "both"):
            raise ValueError(f"obs_type must be vector|image|both, got {obs_type!r}")

        self.cfg = config or EnvConfig()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.image_size = image_size
        self._renderer = renderer

        # Provisional: `seed_all` at the end of __init__ replaces this, but Traffic
        # is handed a generator at construction so one has to exist by then.
        self.rng = np.random.default_rng(seed)
        city_cfg = city_config or CityConfig()
        # The renderer draws signal masts from `city.signals`, so if the city kept
        # signals while the task had them switched off, the pixel observation would
        # show lights that never change and that the vector observation and reward
        # know nothing about. One switch, one city.
        if not self.cfg.traffic_lights and city_cfg.max_signals:
            city_cfg = copy.copy(city_cfg)   # never mutate a caller's config
            city_cfg.max_signals = 0
        self.city = ProceduralCity(city_cfg, seed=seed)
        self.car = Car(car_params or CarParams())
        self.traffic = Traffic(self.city, self.cfg, self.rng)
        self.lidar = Lidar(
            n_beams=self.cfg.n_beams,
            max_range=self.cfg.lidar_range,
            noise_std=self.cfg.lidar_noise,
            rng=self.rng,
        )
        self.radar = Radar(
            n_sectors=self.cfg.n_radar_sectors,
            max_range=self.cfg.radar_range,
            noise_std=self.cfg.radar_noise,
            rng=self.rng,
        )
        self.perception = Perception()
        self.street = Street(self.cfg, self.rng)

        # --- episode state
        self.targets = []
        self.traffic_lights = []
        self.target_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist = None
        self.stuck_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.terminated_reason = None
        self._in_intersection = set()   # indices of TLs the car is currently inside
        self.red_light_violations = 0
        self.crash_with = None

        # --- spaces
        self.action_space = _box(-1.0, 1.0, (3,))
        self.vector_dim = (self.cfg.n_beams + 5 + 3 * self.cfg.n_lookahead
                           + TL_FEATURES * self.cfg.n_tl_obs
                           + TRAFFIC_FEATURES * self.cfg.n_traffic_obs
                           + PED_FEATURES * self.cfg.n_ped_obs
                           + SIGN_FEATURES * self.cfg.n_sign_obs)
        vec_space = _box(-1.0, 1.0, (self.vector_dim,))
        img_space = _box(0, 255, (image_size, image_size, 3), np.uint8)
        if obs_type == "vector":
            self.observation_space = vec_space
        elif obs_type == "image":
            self.observation_space = img_space
        else:
            if _HAS_GYM:
                self.observation_space = gym_spaces.Dict({"vector": vec_space, "image": img_space})
            else:
                self.observation_space = {"vector": vec_space, "image": img_space}

        # Seeding is centralised *after* the spaces exist, because the spaces draw
        # too. Doing it here as well as in `reset` is what makes the two ways of
        # seeding an env agree; see `seed_all`.
        self.seed_all(seed)
        if not self.cfg.randomize_map:
            # A fixed layout has no reset-time `generate` to produce it, so it has
            # to be drawn from the seeded stream here. When the map *is* resampled,
            # regenerating here instead advances the city stream by one draw, which
            # is what made `CarNavEnv(seed=s).reset()` and
            # `CarNavEnv().reset(seed=s)` return two different episodes.
            self.city.generate()

    # ------------------------------------------------------------------
    # Observation layout
    # ------------------------------------------------------------------
    @property
    def obs_slices(self):
        """Name -> slice into the vector observation, for the configured widths.

        Exposed because every consumer outside this file needs it and each one
        would otherwise recompute the offsets from `n_beams`, `n_lookahead`,
        `n_tl_obs` and `n_traffic_obs`. That has already gone wrong once: the
        scripted baseline took "the rest of the vector" as the traffic-light
        block, which kept working until traffic was appended behind it and then
        fed vehicle features to the signal logic. A world model that wants to
        weight the LIDAR block differently from the nav block, or decode them with
        separate heads, needs the same numbers -- so there is one source for them.

        Blocks that are switched off are present as empty slices rather than
        absent, so `obs[sl["traffic"]]` is always valid and returns nothing when
        there is no traffic.
        """
        c = self.cfg
        w = [("lidar", c.n_beams), ("dynamics", 5), ("nav", 3 * c.n_lookahead),
             ("traffic_light", TL_FEATURES * c.n_tl_obs),
             ("traffic", TRAFFIC_FEATURES * c.n_traffic_obs),
             ("pedestrians", PED_FEATURES * c.n_ped_obs),
             ("signs", SIGN_FEATURES * c.n_sign_obs)]
        out, i = {}, 0
        for name, n in w:
            out[name] = slice(i, i + n)
            i += n
        assert i == self.vector_dim, f"{i} != {self.vector_dim}"
        return out

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed_all(self, seed=None):
        """Single entry point for randomness. Reseeds every subsystem that draws.

        Two things here are deliberate and both were bugs first.

        *Every* subsystem is reseeded, not just some. `reset(seed=s)` used to
        rebind the env's generator and forward it to the city and the LIDAR but
        not to `Traffic`, which kept the generator it was handed at construction.
        The effect was that `reset(seed=7)` twice on one env gave two different
        episodes -- same map, same spawn, different traffic -- so a run was only
        reproducible if you also rebuilt the env, which is not how any training
        loop is written. The existing determinism test missed it because it
        constructed a fresh env per rollout.

        Each subsystem gets an *independent* stream spawned from the seed rather
        than a shared generator. Sharing one couples them through draw order:
        adding a single `rng.uniform` to the traffic spawner would then silently
        change every city layout, so an experiment could not be reproduced across
        a code change that had nothing to do with it.
        """
        ss = np.random.SeedSequence(seed)
        # `radar_ss` is appended as a 7th, trailing stream rather than inserted
        # among the existing six: SeedSequence.spawn(N) is a prefix of spawn(M)
        # for M > N, so adding a stream at the end leaves every existing
        # subsystem's draws (city, lidar, traffic, ...) exactly as they were --
        # inserting it earlier would silently reshuffle all of them and break
        # every previously-seeded rollout's reproducibility.
        (env_ss, city_ss, lidar_ss, traffic_ss, space_ss, gym_ss, radar_ss,
         street_ss) = ss.spawn(8)                  # street_ss: trailing, same rule
        self.rng = np.random.default_rng(env_ss)
        self.city.rng = np.random.default_rng(city_ss)
        self.lidar.rng = np.random.default_rng(lidar_ss)
        self.radar.rng = np.random.default_rng(radar_ss)
        self.traffic.rng = np.random.default_rng(traffic_ss)
        self.street.rng = np.random.default_rng(street_ss)
        space_seed = int(space_ss.generate_state(1, dtype=np.uint32)[0])
        for space in (self.action_space, self.observation_space):
            if hasattr(space, "seed"):
                space.seed(space_seed)
        if _HAS_GYM:
            # `env.np_random` is part of the gymnasium Env contract: wrappers read
            # it and `utils.env_checker` fails the env outright if a seeded reset
            # leaves it unset, which it did. The simulation itself never draws from
            # it -- it has its own streams above -- so it gets a sixth spawned
            # stream rather than sharing one, keeping the guarantee that nothing a
            # wrapper draws can shift a map layout.
            self._np_random = np.random.default_rng(gym_ss)
            self._np_random_seed = -1 if seed is None else int(seed)
        self._seed = seed

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_all(seed)

        if self.cfg.randomize_map:
            self.city.generate()

        # Only the sparse signalised subset gets lights, and phases are staggered
        # so the agent meets different states rather than one synchronised city.
        self.traffic_lights = [
            TrafficLight(x, y, phase=i % 4, timer=(i * 23) % TrafficLight.YELLOW_STEPS)
            for i, (x, y) in enumerate(self.city.signals)
        ] if self.cfg.traffic_lights else []

        x, y, heading = self.city.sample_free_pose(self.car.p.length, self.car.p.width)
        self.car.reset(x, y, heading)
        self._sample_targets()

        # After the ego pose, so traffic can be kept clear of it. `sample_free_pose`
        # only checks buildings, so a vehicle spawned on the spawn point would put
        # back the unwinnable start that the swept-clearance check removes.
        self.traffic.reset(x, y)
        self.street.reset(self.city, x, y)
        if self.cfg.radar:
            self.radar.scan(self.traffic, self.car)
        if self.cfg.perception:
            self.perception.scan(self.traffic, self.car)

        self.target_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.terminated_reason = None
        self._in_intersection = set()
        self.red_light_violations = 0
        self.crash_with = None
        self.prev_dist = self._dist_to_target()

        if self._renderer is not None:
            self._renderer.build_scene(self.city, self.traffic, self.street)

        return self._observe(), self._info()

    def _sample_targets(self):
        """Chain waypoints so each is a reachable hop from the previous one."""
        self.targets = []
        px, py = self.car.x, self.car.y
        for _ in range(self.cfg.n_targets):
            pt = self.city.sample_point_near(
                px, py, self.cfg.target_min_dist, self.cfg.target_max_dist
            )
            if pt is None:
                pt = self.city.sample_road_point()
            self.targets.append(pt)
            px, py = pt

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        self.last_action = action
        throttle = float(action[0])              # signed: +forward, -reverse
        brake = (action[1] + 1.0) * 0.5
        steer = action[2]

        reward = 0.0
        comp_time = 0.0
        comp_crash = 0.0
        comp_progress = 0.0
        comp_target = 0.0
        comp_red_light = 0.0
        comp_pedestrian = 0.0
        comp_speeding = 0.0
        terminated = False
        crashed = False
        reached = 0
        prev_x, prev_y = self.car.x, self.car.y   # for entry-arm resolution below

        for _ in range(cfg.action_repeat):
            self.car.step(throttle, brake, steer, cfg.dt)
            # Traffic moves inside the repeat loop, on the same clock as the ego.
            # Stepping it once per env step instead would let a vehicle jump
            # action_repeat * dt * v metres between collision checks and pass
            # clean through the car.
            self.traffic.step(cfg.dt, self.car, self.traffic_lights, self.street)
            self.street.step(cfg.dt, self.car)
            reward -= cfg.time_penalty
            comp_time -= cfg.time_penalty

            hit_building = self.city.collides(
                self.car.x, self.car.y, self.car.heading,
                self.car.p.length, self.car.p.width)
            hit_ped = self.street.hits(self.car)
            if hit_building or hit_ped or self.traffic.hits(self.car):
                # A person is categorically worse than a kerb or a parked van:
                # its own, larger penalty and its own reward component.
                if hit_ped:
                    reward -= cfg.pedestrian_penalty
                    comp_pedestrian -= cfg.pedestrian_penalty
                else:
                    reward -= cfg.crash_penalty
                    comp_crash -= cfg.crash_penalty
                terminated = crashed = True
                self.terminated_reason = "crash"
                self.crash_with = ("pedestrian" if hit_ped
                                   else "building" if hit_building else "vehicle")
                break

            if cfg.speed_signs:
                excess = self.car.speed - self.street.limit_at(self.car.x, self.car.y)
                if excess > 0.0:
                    fine = cfg.speeding_penalty * excess
                    reward -= fine
                    comp_speeding -= fine

            dist = self._dist_to_target()
            step_progress = (self.prev_dist - dist) * cfg.progress_weight
            reward += step_progress
            comp_progress += step_progress
            self.prev_dist = dist

            if dist < cfg.target_radius:
                reward += cfg.target_bonus
                comp_target += cfg.target_bonus
                reached += 1
                self.target_idx += 1
                if self.target_idx >= len(self.targets):
                    terminated = True
                    self.terminated_reason = "success"
                    break
                self.prev_dist = self._dist_to_target()

        # Red-light violation: one-shot penalty when the car crosses the stop line
        # into an intersection while the signal for its approach is red. Checked
        # before ticking so the car is judged on the phase it actually saw.
        #
        # Only entry is penalised, never occupancy: a light turning red while you
        # are already in the box is not a violation, and charging per-step would
        # make one mistake unboundedly expensive.
        if not terminated and self.traffic_lights:
            half = self._zone_half
            for i, tl in enumerate(self.traffic_lights):
                in_zone = (abs(self.car.x - tl.x) < half and
                           abs(self.car.y - tl.y) < half)
                if in_zone and i not in self._in_intersection:
                    axis = self._entry_axis(tl, prev_x, prev_y, half)
                    if tl.state(axis) == 'red':
                        reward -= cfg.red_light_penalty
                        comp_red_light -= cfg.red_light_penalty
                        self.red_light_violations += 1
                if in_zone:
                    self._in_intersection.add(i)
                else:
                    self._in_intersection.discard(i)

        self.step_count += 1
        for tl in self.traffic_lights:
            tl.tick()

        # Idling is discouraged by the time penalty; cutting the episode short
        # here only saves compute. The remaining penalty is charged as a lump sum
        # so early termination carries no reward-shaping side effect.
        # Waiting at a red is not being stuck. Without this exemption the stuck
        # detector and the signal timings are silently coupled: a red lasts
        # GREEN+YELLOW steps, so any cycle longer than stuck_steps would terminate
        # the episode *for obeying the law*.
        # Queueing behind stopped traffic is exempt for the same reason: the
        # episode must not end because the agent correctly declined to drive into
        # the back of a car. Early termination is reward-neutral, so being generous
        # here costs nothing but compute.
        if not terminated and cfg.stuck_steps:
            idle = self.car.speed < cfg.stuck_speed
            held = idle and (self._waiting_at_red() or self._blocked_by_traffic()
                             or self.street.yielding(self.car))
            self.stuck_count = self.stuck_count + 1 if (idle and not held) else 0
            if self.stuck_count >= cfg.stuck_steps:
                remaining = max(0, cfg.max_episode_steps - self.step_count)
                stuck_penalty = cfg.time_penalty * remaining * cfg.action_repeat
                reward -= stuck_penalty
                comp_time -= stuck_penalty
                terminated = True
                self.terminated_reason = "stuck"

        truncated = (not terminated) and self.step_count >= cfg.max_episode_steps
        if truncated:
            self.terminated_reason = "timeout"

        if self.cfg.radar:
            self.radar.scan(self.traffic, self.car)
        if self.cfg.perception:
            self.perception.scan(self.traffic, self.car)

        self.episode_reward += reward
        info = self._info()
        info["crashed"] = crashed
        info["targets_reached_this_step"] = reached
        info["red_light_violations"] = self.red_light_violations
        info["reward_components"] = {
            "time": comp_time,
            "crash": comp_crash,
            "progress": comp_progress,
            "target_bonus": comp_target,
            "red_light": comp_red_light,
            "pedestrian": comp_pedestrian,
            "speeding": comp_speeding,
        }
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _dist_to_target(self):
        if self.target_idx >= len(self.targets):
            return 0.0
        tx, ty = self.targets[self.target_idx]
        return float(np.hypot(tx - self.car.x, ty - self.car.y))

    # ------------------------------------------------------------------
    # Traffic lights
    # ------------------------------------------------------------------
    @property
    def _zone_half(self):
        """Half-width of an intersection box; its edges are the stop lines."""
        return (self.city.cfg.road_width / 2.0) * self.city.tile_size

    @staticmethod
    def _entry_axis(tl, prev_x, prev_y, half):
        """Which axis of traffic the car joined, from the arm it entered through.

        Deciding this from instantaneous heading instead (|sin h| > |cos h|) is a
        coin flip whenever the car is near 45 degrees -- measured at 7% of real
        entries, with 11% within 20 degrees. That turns the penalty itself into
        noise the agent cannot learn from. The arm the car came *through* is
        unambiguous and is fixed at the moment the penalty is evaluated.
        """
        was_out_y = abs(prev_y - tl.y) >= half
        was_out_x = abs(prev_x - tl.x) >= half
        if was_out_y and not was_out_x:
            return 'ns'                    # crossed a north/south stop line
        if was_out_x and not was_out_y:
            return 'ew'
        # Corner case (both/neither outside): fall back to travel direction.
        return 'ns' if abs(prev_y - tl.y) > abs(prev_x - tl.x) else 'ew'

    @staticmethod
    def _approach_axis(tl, x, y):
        """Axis the car will enter on, from where the crossing lies relative to it.

        Used for the observation, where entry has not happened yet. Displacement
        to the crossing is far steadier than heading -- approaching down a
        corridor, the crossing is squarely ahead on that corridor's axis -- and it
        agrees with `_entry_axis` by construction.
        """
        return 'ns' if abs(y - tl.y) > abs(x - tl.x) else 'ew'

    def _waiting_at_red(self):
        """True if the car is legitimately held at a red it is approaching."""
        if not self.traffic_lights:
            return False
        half = self._zone_half
        x, y = self.car.x, self.car.y
        for tl in self.traffic_lights:
            # Near the stop line, on the approach, and facing a red.
            if (abs(x - tl.x) < half + 12.0 and abs(y - tl.y) < half + 12.0
                    and tl.state(self._approach_axis(tl, x, y)) == 'red'):
                return True
        return False

    def _blocked_by_traffic(self):
        """True if a stopped moving vehicle is directly ahead within a car length.

        Restricted to the moving population and to vehicles that are themselves
        stopped, so this cannot be gamed by idling next to a parked car.
        """
        t = self.traffic
        m = t.n_moving
        if m == 0:
            return False
        car = self.car
        c, s = np.cos(car.heading), np.sin(car.heading)
        dx, dy = t.x[:m] - car.x, t.y[:m] - car.y
        fwd = dx * c + dy * s
        lat = np.abs(-dx * s + dy * c)
        gap = fwd - (car.p.length + t.length[:m]) / 2.0
        return bool(((t.speed[:m] < 1.0) & (fwd > 0.0) & (gap < 5.0)
                     & (lat < (car.p.width + t.width[:m]) / 2.0 + 0.6)).any())

    def vector_obs(self):
        # Vehicles are not part of the tile grid, so they reach LIDAR through the
        # circle-obstacle path. Same circles as the collision test, so a range the
        # agent is shown always matches the shape it would hit.
        obstacles = self.traffic.circles(self.car.x, self.car.y, self.cfg.lidar_range)
        if self.cfg.pedestrians:
            # People are geometry to the LIDAR as much as vehicles are.
            people = self.street.circles(self.car.x, self.car.y, self.cfg.lidar_range)
            if len(people):
                obstacles = np.concatenate([obstacles, people]) if len(obstacles) else people
        scan = self.lidar.scan(self.city, self.car.x, self.car.y, self.car.heading,
                               obstacles=obstacles)
        dyn = dynamics_features(self.car)
        nav = nav_features(self.car, self.targets, self.target_idx,
                           self.cfg.n_lookahead, self.cfg.nav_range)
        traf = self.traffic.obs_features(self.car, self.cfg.n_traffic_obs,
                                        self.cfg.traffic_obs_range,
                                        self.car.p.max_speed)
        ped = self.street.obs_features(self.car, self.cfg.n_ped_obs,
                                       self.cfg.ped_obs_range, self.car.p.max_speed)
        signs = self.street.sign_features(self.car, self.cfg.n_sign_obs,
                                          self.cfg.sign_range, self.car.p.max_speed)
        return np.concatenate([scan, dyn, nav, self._tl_features(), traf, ped, signs]).astype(np.float32)

    def _tl_features(self):
        """Encode the n_tl_obs nearest signals, ego-centrically, in [-1, 1].

        Per signal (TL_FEATURES values):
            dist_norm    distance to the stop line / tl_obs_range, 0 at the line
            sin_bearing  ego-centric bearing to the crossing
            cos_bearing
            red          one-hot state for *this car's approach axis*
            yellow
            green
            time_norm    steps until the phase changes, / GREEN_STEPS

        `time_norm` is what makes the light anticipatable rather than a surprise:
        it is the difference between "stop" and "it will be green before you get
        there", and it is the latent a world model should learn to roll forward.

        Distance is measured to the *stop line*, not the crossing centre, because
        that is where the decision has to be taken.

        Absent or out-of-range signals pad with green at full distance, which no
        real approaching red can produce, so "no signal" stays distinguishable.
        """
        n = self.cfg.n_tl_obs
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        car = self.car
        r = self.cfg.tl_obs_range
        half = self._zone_half
        pad = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        ranked = sorted(self.traffic_lights,
                        key=lambda tl: (tl.x - car.x) ** 2 + (tl.y - car.y) ** 2)

        out = []
        for k in range(n):
            if k >= len(ranked):
                out += pad
                continue
            tl = ranked[k]
            dx, dy = tl.x - car.x, tl.y - car.y
            to_line = max(0.0, float(np.hypot(dx, dy)) - half)
            if to_line >= r:
                out += pad
                continue
            bearing = np.arctan2(dy, dx) - car.heading
            state = tl.state(self._approach_axis(tl, car.x, car.y))
            out += [
                float(to_line / r),
                float(np.sin(bearing)),
                float(np.cos(bearing)),
                float(state == 'red'),
                float(state == 'yellow'),
                float(state == 'green'),
                float(min(1.0, tl.steps_remaining / TrafficLight.GREEN_STEPS)),
            ]

        return np.array(out, dtype=np.float32)

    def image_obs(self):
        if self._renderer is None:
            self._renderer = self._make_default_renderer()
            self._renderer.build_scene(self.city, self.traffic)
        return self._renderer.capture(self, size=self.image_size)

    def _make_default_renderer(self):
        try:
            from render.panda_renderer import PandaRenderer
        except ImportError as e:
            raise RuntimeError(
                "Image observations need the Panda3D renderer. "
                "Install it with `pip install panda3d`, or pass obs_type='vector'."
            ) from e
        return PandaRenderer(offscreen=True, size=self.image_size)

    def _observe(self):
        if self.obs_type == "vector":
            return self.vector_obs()
        if self.obs_type == "image":
            return self.image_obs()
        return {"vector": self.vector_obs(), "image": self.image_obs()}

    def _info(self):
        tx, ty = self.targets[self.target_idx] if self.target_idx < len(self.targets) else (self.car.x, self.car.y)
        _, sin_b, cos_b = relative_bearing(self.car.x, self.car.y, self.car.heading, tx, ty)
        return {
            "x": self.car.x,
            "y": self.car.y,
            "heading": self.car.heading,
            "speed": self.car.speed,
            "steer_angle": self.car.steer_angle,
            "dist_to_target": self._dist_to_target(),
            "target_bearing": float(np.arctan2(sin_b, cos_b)),
            "targets_reached": self.target_idx,
            "n_targets": len(self.targets),
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "reason": self.terminated_reason,
            "is_success": self.terminated_reason == "success",
            "red_light_violations": self.red_light_violations,
            "crash_with": self.crash_with,
            "speed_limit": (self.street.limit_at(self.car.x, self.car.y)
                            if self.cfg.speed_signs else None),
            "pedestrians_on_road": int(self.street.on_road().sum()),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = self._make_default_renderer()
            self._renderer.build_scene(self.city, self.traffic)
        return self._renderer.capture(self, size=self.image_size)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
