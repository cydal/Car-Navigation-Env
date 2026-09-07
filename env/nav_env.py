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

import numpy as np

from .world import ProceduralCity, CityConfig
from .car import Car, CarParams
from .sensors import Lidar, nav_features, dynamics_features, relative_bearing

try:                                    # gymnasium is optional
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
    _HAS_GYM = True
except ImportError:                     # pragma: no cover
    gym, gym_spaces = None, None
    _HAS_GYM = False


class _Box:
    """Minimal Box stand-in so the env is usable without gymnasium installed."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.__name__})"


def _box(low, high, shape, dtype=np.float32):
    if _HAS_GYM:
        return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _Box(low, high, shape, dtype)


_BaseEnv = gym.Env if _HAS_GYM else object


class TrafficLight:
    """Two-phase signal: N-S green then E-W green, with yellow transitions."""
    GREEN_STEPS  = 80   # simulation steps per green phase
    YELLOW_STEPS = 12   # simulation steps per yellow phase

    def __init__(self, x, y, phase=0, timer=0):
        self.x = x
        self.y = y
        self.phase = phase  # 0=NS-green, 1=NS-yellow, 2=EW-green, 3=EW-yellow
        self.timer = timer

    def tick(self):
        limit = self.YELLOW_STEPS if self.phase % 2 == 1 else self.GREEN_STEPS
        self.timer += 1
        if self.timer >= limit:
            self.phase = (self.phase + 1) % 4
            self.timer = 0

    @property
    def ns_state(self):
        return ('green', 'yellow', 'red', 'red')[self.phase]

    @property
    def ew_state(self):
        return ('red', 'red', 'green', 'yellow')[self.phase]


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
        red_light_penalty=50.0,
        # --- traffic-light observation (0 = off, keeps the default 46-dim vector)
        n_tl_obs=0,
        tl_obs_range=60.0,
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
        self.red_light_penalty = red_light_penalty
        self.n_tl_obs = n_tl_obs                # nearest TLs in obs; 0 = off
        self.tl_obs_range = tl_obs_range


class CarNavEnv(_BaseEnv):
    """Drive a car through a sequence of waypoints in a procedural city.

    Action (3,) in [-1, 1]:
        0  throttle  rescaled to [0, 1]
        1  brake     rescaled to [0, 1]
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

        self.rng = np.random.default_rng(seed)
        self.city = ProceduralCity(city_config or CityConfig(), seed=seed)
        self.car = Car(car_params or CarParams())
        self.lidar = Lidar(
            n_beams=self.cfg.n_beams,
            max_range=self.cfg.lidar_range,
            noise_std=self.cfg.lidar_noise,
            rng=self.rng,
        )

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

        # --- spaces
        self.action_space = _box(-1.0, 1.0, (3,))
        self.vector_dim = (self.cfg.n_beams + 5 + 3 * self.cfg.n_lookahead
                           + 6 * self.cfg.n_tl_obs)
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

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.city.rng = self.rng
            self.lidar.rng = self.rng

        if self.cfg.randomize_map:
            self.city.generate()

        # Stagger phases so not every intersection turns green simultaneously.
        self.traffic_lights = [
            TrafficLight(x, y, phase=i % 4, timer=(i * 23) % TrafficLight.GREEN_STEPS)
            for i, (x, y) in enumerate(self.city.intersections)
        ]

        x, y, heading = self.city.sample_free_pose(self.car.p.length, self.car.p.width)
        self.car.reset(x, y, heading)
        self._sample_targets()

        self.target_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.terminated_reason = None
        self._in_intersection = set()
        self.red_light_violations = 0
        self.prev_dist = self._dist_to_target()

        if self._renderer is not None:
            self._renderer.build_scene(self.city)

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
        throttle = (action[0] + 1.0) * 0.5
        brake = (action[1] + 1.0) * 0.5
        steer = action[2]

        reward = 0.0
        terminated = False
        crashed = False
        reached = 0

        for _ in range(cfg.action_repeat):
            self.car.step(throttle, brake, steer, cfg.dt)
            reward -= cfg.time_penalty

            if self.city.collides(self.car.x, self.car.y, self.car.heading,
                                  self.car.p.length, self.car.p.width):
                reward -= cfg.crash_penalty
                terminated = crashed = True
                self.terminated_reason = "crash"
                break

            dist = self._dist_to_target()
            reward += (self.prev_dist - dist) * cfg.progress_weight
            self.prev_dist = dist

            if dist < cfg.target_radius:
                reward += cfg.target_bonus
                reached += 1
                self.target_idx += 1
                if self.target_idx >= len(self.targets):
                    terminated = True
                    self.terminated_reason = "success"
                    break
                self.prev_dist = self._dist_to_target()

        # Red-light violation: one-shot penalty when the car enters an intersection
        # zone while the relevant signal is red.  Checked before ticking so the car
        # experiences the phase it sees this step, not the next one.
        if not terminated and self.traffic_lights:
            half_road = (self.city.cfg.road_width / 2.0) * self.city.tile_size
            h = self.car.heading
            ns_direction = abs(np.sin(h)) > abs(np.cos(h))
            for i, tl in enumerate(self.traffic_lights):
                in_zone = (abs(self.car.x - tl.x) < half_road and
                           abs(self.car.y - tl.y) < half_road)
                if in_zone and i not in self._in_intersection:
                    state = tl.ns_state if ns_direction else tl.ew_state
                    if state == 'red':
                        reward -= cfg.red_light_penalty
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
        if not terminated and cfg.stuck_steps:
            self.stuck_count = self.stuck_count + 1 if self.car.speed < cfg.stuck_speed else 0
            if self.stuck_count >= cfg.stuck_steps:
                remaining = max(0, cfg.max_episode_steps - self.step_count)
                reward -= cfg.time_penalty * remaining * cfg.action_repeat
                terminated = True
                self.terminated_reason = "stuck"

        truncated = (not terminated) and self.step_count >= cfg.max_episode_steps
        if truncated:
            self.terminated_reason = "timeout"

        self.episode_reward += reward
        info = self._info()
        info["crashed"] = crashed
        info["targets_reached_this_step"] = reached
        info["red_light_violations"] = self.red_light_violations
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _dist_to_target(self):
        if self.target_idx >= len(self.targets):
            return 0.0
        tx, ty = self.targets[self.target_idx]
        return float(np.hypot(tx - self.car.x, ty - self.car.y))

    def vector_obs(self):
        scan = self.lidar.scan(self.city, self.car.x, self.car.y, self.car.heading)
        dyn = dynamics_features(self.car)
        nav = nav_features(self.car, self.targets, self.target_idx,
                           self.cfg.n_lookahead, self.cfg.nav_range)
        return np.concatenate([scan, dyn, nav, self._tl_features()]).astype(np.float32)

    def _tl_features(self):
        """Encode the n_tl_obs nearest traffic lights into the observation vector.

        Per light (6 values, all in [-1, 1] or [0, 1]):
            dist_norm   normalised distance (0 = at light, 1 = at/beyond tl_obs_range)
            sin_bearing ego-centric bearing to the intersection
            cos_bearing
            red         1 if the relevant signal is red, else 0
            yellow      1 if yellow
            green       1 if green

        The relevant signal is NS if |sin(heading)| > |cos(heading)|, else EW.
        Absent / out-of-range lights are padded with (1, 0, 0, 0, 0, 1) — far, green.
        """
        n = self.cfg.n_tl_obs
        if n == 0 or not self.traffic_lights:
            return np.zeros(6 * n, dtype=np.float32)

        car = self.car
        r = self.cfg.tl_obs_range
        ns_dir = abs(np.sin(car.heading)) > abs(np.cos(car.heading))

        ranked = sorted(
            self.traffic_lights,
            key=lambda tl: (tl.x - car.x) ** 2 + (tl.y - car.y) ** 2
        )

        out = []
        for k in range(n):
            if k < len(ranked):
                tl = ranked[k]
                dist = float(np.hypot(tl.x - car.x, tl.y - car.y))
                if dist < r:
                    bearing = np.arctan2(tl.y - car.y, tl.x - car.x) - car.heading
                    state = tl.ns_state if ns_dir else tl.ew_state
                    out += [
                        float(dist / r),
                        float(np.sin(bearing)),
                        float(np.cos(bearing)),
                        float(state == 'red'),
                        float(state == 'yellow'),
                        float(state == 'green'),
                    ]
                    continue
            out += [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # pad: far, green

        return np.array(out, dtype=np.float32)

    def image_obs(self):
        if self._renderer is None:
            self._renderer = self._make_default_renderer()
            self._renderer.build_scene(self.city)
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
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = self._make_default_renderer()
            self._renderer.build_scene(self.city)
        return self._renderer.capture(self, size=self.image_size)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
