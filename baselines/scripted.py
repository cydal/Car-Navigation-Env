"""
Scripted driving baseline: follow-the-gap steering + braking-distance speed control.

This controller consumes *only the observation vector* -- no privileged access to
the world, the car object or the target list. That makes it a test of the
observation design itself: if a few lines of classical control can drive this
thing, the vector state carries enough information to learn from, and an RL agent
that underperforms it has a learning problem rather than a sensing one.

It also gives a reference score to compare learned policies against.

Why gap-following rather than goal-attraction + obstacle-repulsion: a repulsion
field steers *away* from walls, which in a street grid means it fights the goal
term and stalls facing a facade whenever the target is behind the car. Scoring
candidate *headings* by *how far you can travel along them* instead makes the
car drive to an intersection and turn there, which is what a street network
requires.
"""

import numpy as np


class GapFollower:
    """Follow-the-gap driver operating on CarNavEnv vector observations."""

    def __init__(
        self,
        n_beams=32,
        n_lookahead=3,
        lidar_range=60.0,
        max_speed=22.0,
        max_steer=np.radians(32.0),
        car_length=4.4,
        car_width=1.9,
        wheelbase=2.5,
        cone_deg=100.0,
        w_goal=1.0,
        w_clear=0.75,
        lookahead=18.0,
        lookahead_time=1.4,
        lookahead_min=6.0,
        k_center=0.8,
        corridor_thresh=11.0,
        steer_smooth=0.45,
        cruise_speed=10.0,
        min_speed=2.0,
        brake_decel=6.0,
        stop_margin=3.5,
        n_tl_obs=1,
        tl_obs_range=60.0,
        tl_stop_margin=2.0,
        n_traffic_obs=4,
        traffic_obs_range=60.0,
        follow_lat=2.5,
        follow_gap=4.0,
    ):
        self.n_beams = n_beams
        self.n_lookahead = n_lookahead
        self.n_tl_obs = n_tl_obs
        self.tl_obs_range = tl_obs_range
        self.tl_stop_margin = tl_stop_margin  # metres to stop short of the stop line
        self.n_traffic_obs = n_traffic_obs
        self.traffic_obs_range = traffic_obs_range
        self.follow_lat = follow_lat          # lateral window for "in my path"
        self.follow_gap = follow_gap          # bumper gap held in a queue
        self.car_length = car_length
        self.lidar_range = lidar_range
        self.max_speed = max_speed
        self.max_steer = max_steer
        self.wheelbase = wheelbase
        self.w_goal = w_goal
        self.w_clear = w_clear
        self.lookahead = lookahead            # clearance beyond this counts as "enough"
        self.lookahead_time = lookahead_time  # pure-pursuit lookahead = v * this
        self.lookahead_min = lookahead_min
        self.k_center = k_center              # corridor centering gain
        self.corridor_thresh = corridor_thresh
        self.steer_smooth = steer_smooth      # EMA factor on the steer command
        self.cruise_speed = cruise_speed
        self.min_speed = min_speed
        self.brake_decel = brake_decel        # conservative, below the car's limit
        self.stop_margin = stop_margin        # metres to keep in hand

        # Mirrors Lidar's beam layout: index n/2 is straight ahead.
        self.offsets = -np.pi + 2.0 * np.pi * np.arange(n_beams) / n_beams
        self.cone = np.abs(self.offsets) <= np.radians(cone_deg)

        # LIDAR measures from the car's centre, so subtract the distance from the
        # centre to the bodywork in each beam direction -- the box's radial extent.
        hl, hw = car_length / 2.0, car_width / 2.0
        eps = 1e-9
        self.overhang = np.minimum(
            hl / np.maximum(np.abs(np.cos(self.offsets)), eps),
            hw / np.maximum(np.abs(np.sin(self.offsets)), eps),
        )

        self.half_width = car_width / 2.0

        # Side beams, for measuring how far off-centre we are in a street.
        deg = np.degrees(self.offsets)
        self.right_mask = (deg > 65.0) & (deg < 115.0)     # +offset == toward +y
        self.left_mask = (deg < -65.0) & (deg > -115.0)

        self.steer_prev = 0.0

    @classmethod
    def for_env(cls, env, **kw):
        """Build a driver matching an env's observation layout and car dimensions.

        Still obs-only: what is read here is the *shape* of the observation and the
        car's own geometry, never the world, the targets or the traffic. Having one
        place that does it stops four call sites from disagreeing about the layout,
        which is how the traffic-light slice came to be mis-read in the first place.
        """
        c, p = env.cfg, env.car.p
        return cls(n_beams=c.n_beams, n_lookahead=c.n_lookahead,
                   lidar_range=c.lidar_range, max_speed=p.max_speed,
                   max_steer=p.max_steer, car_length=p.length, car_width=p.width,
                   wheelbase=p.wheelbase,
                   n_tl_obs=c.n_tl_obs, tl_obs_range=c.tl_obs_range,
                   n_traffic_obs=c.n_traffic_obs,
                   traffic_obs_range=c.traffic_obs_range, **kw)

    def reset(self):
        """Clear the steering filter between episodes."""
        self.steer_prev = 0.0

    # ------------------------------------------------------------------
    def unpack(self, obs):
        """Split the flat observation into scan / dynamics / nav / lights / traffic.

        Every block is sliced by its own declared width rather than taking "the
        rest" as the traffic-light slice. That shortcut worked while lights were
        the last block, but appending traffic behind them made it hand vehicle
        features to the signal logic, which reads index 3 as `red` -- a driver
        braking for a phantom light whenever a car happened to be nearby. The
        length check turns any future layout change into an error instead.
        """
        n = self.n_beams
        nav_end = n + 5 + 3 * self.n_lookahead
        tl_end = nav_end + 7 * self.n_tl_obs
        traf_end = tl_end + 5 * self.n_traffic_obs
        if traf_end != len(obs):
            raise ValueError(
                f"observation is {len(obs)}-D but this driver expects {traf_end}-D "
                f"(n_beams={n}, n_lookahead={self.n_lookahead}, "
                f"n_tl_obs={self.n_tl_obs}, n_traffic_obs={self.n_traffic_obs})")
        return (obs[:n], obs[n:n + 5], obs[n + 5:nav_end],
                obs[nav_end:tl_end], obs[tl_end:traf_end])

    def _red_light_stop_speed(self, tl):
        """Target speed to hold for the nearest signal, or None if it is not ours.

        Only reds and yellows ahead of the car matter. A signal that is already
        behind us (cos_bearing <= 0) or that we are inside the box of has no claim
        on us -- braking to a halt mid-intersection would be worse than clearing it.
        """
        if tl.size < 7:
            return None
        to_line = float(tl[0]) * self.tl_obs_range
        cos_b = float(tl[2])
        red, yellow = float(tl[3]), float(tl[4])
        if (red < 0.5 and yellow < 0.5) or cos_b <= 0.3:
            return None
        # Already committed: past the stop line, so keep going and clear the box.
        if to_line <= 0.2:
            return None
        room = max(0.0, to_line - self.tl_stop_margin)
        return float(np.sqrt(2.0 * self.brake_decel * room))

    def _traffic_follow_speed(self, traf, speed):
        """Speed to hold behind the nearest vehicle in our own path, or None.

        LIDAR already slows the car for traffic -- the vehicles are in the scan --
        but the result is floored at `min_speed`, so the car creeps into whatever
        it has stopped behind. Queueing needs a genuine zero, exactly like a red
        light, and for the same reason: the floor exists to stop dithering in a
        corridor, not to forbid stopping.

        Only vehicles travelling *with* us count. An oncoming car sits inside the
        lateral window too on a 12 m undivided corridor, and halting for one would
        freeze the baseline at every passing car; both parties keeping right is
        what resolves that, which the corridor centering term already does.
        """
        if traf.size < 5:
            return None
        best = None
        for i in range(0, traf.size - 4, 5):
            sin_b, cos_b = float(traf[i + 1]), float(traf[i + 2])
            if sin_b == 0.0 and cos_b == 0.0:
                continue                        # padded empty slot, not a vehicle
            d = float(traf[i]) * self.traffic_obs_range
            fwd, lat = d * cos_b, d * sin_b
            if fwd <= 0.0 or abs(lat) > self.follow_lat:
                continue
            # The block carries velocity *relative to us*; add our own back to get
            # the leader's, which is what sets how slowly we have to end up going.
            v_lead = float(traf[i + 3]) * self.max_speed + speed
            if v_lead < -0.5:
                continue                        # closing head-on: not a leader
            room = max(0.0, fwd - self.car_length - self.follow_gap)
            v = float(np.sqrt(max(0.0, v_lead) ** 2 + 2.0 * self.brake_decel * room))
            best = v if best is None else min(best, v)
        return best

    def _forward_room(self, eff):
        """How far the car can carry straight on before its body meets something.

        A fixed forward cone is wrong at both ends: at 20 m a 25 deg cone spans
        wider than a 12 m street and calls the far kerb an obstacle, while a single
        beam misses a car sitting half a lane over. Project each beam's hit into the
        car's frame instead and keep the ones landing inside a body-width corridor
        ahead -- the same swept test `world.sample_free_pose` uses for spawn poses.
        """
        fwd = eff * np.cos(self.offsets)
        lat = np.abs(eff * np.sin(self.offsets))
        blocking = (fwd > 0.0) & (lat < self.half_width + 0.3)
        return float(fwd[blocking].min()) if blocking.any() else self.lidar_range

    def _windowed_clearance(self, eff):
        """Shrink each beam's range to the min of itself and its neighbours.

        Stops the car aiming at a gap that is only one beam wide and therefore
        narrower than the car.
        """
        return np.minimum(eff, np.minimum(np.roll(eff, 1), np.roll(eff, -1)))

    def _clearance_at(self, clear, angle):
        idx = int(round((angle + np.pi) / (2.0 * np.pi) * self.n_beams)) % self.n_beams
        return float(clear[idx])

    def act(self, obs):
        scan, dyn, nav, tl, traf = self.unpack(obs)
        speed = float(dyn[0]) * self.max_speed
        bearing = float(np.arctan2(nav[1], nav[2]))

        # Travellable distance from the bodywork, per beam.
        eff = np.maximum(scan * self.lidar_range - self.overhang, 0.0)
        clear = self._windowed_clearance(eff)

        # --- candidate headings: every in-cone beam, plus the exact goal bearing
        cand = list(self.offsets[self.cone])
        if abs(bearing) <= np.radians(100.0):
            cand.append(bearing)
        cand = np.asarray(cand)

        clears = np.array([self._clearance_at(clear, a) for a in cand])
        align = 0.5 * (1.0 + np.cos(cand - bearing))            # 1 = straight at goal

        # Clearance *saturates*: past the point where there is comfortably enough
        # room, extra distance is worthless. Scoring it linearly instead makes a
        # long open street outscore the turn the car actually needs, so the car
        # sails past its waypoint down the roomiest road it can find.
        need = self.stop_margin + speed * speed / (2.0 * self.brake_decel)
        enough = max(self.lookahead, need)
        clear_term = np.clip(clears / enough, 0.0, 1.0)
        score = self.w_goal * align + self.w_clear * clear_term

        # Reject anything we could not stop within, unless nothing qualifies.
        ok = clears >= need
        best = int(np.argmax(np.where(ok, score, -np.inf))) if ok.any() else int(np.argmax(clears))

        theta = float(cand[best])

        # --- pure-pursuit geometry: curvature to a point `L` ahead at angle theta.
        # A plain proportional gain on theta commands near-full lock for a 10 deg
        # correction and oscillates; this scales the response to how far away the
        # aim point is, which is what actually keeps the car on line.
        L = max(self.lookahead_min, speed * self.lookahead_time)
        curvature = 2.0 * np.sin(theta) / L
        delta = np.arctan(curvature * self.wheelbase)
        steer = float(np.clip(delta / self.max_steer, -1.0, 1.0))

        # --- corridor centering: balance the room either side.
        # Without this the goal term drags the car diagonally across a street
        # until a corner clips a facade -- the centre stays on road the whole way,
        # so nothing else in the controller ever objects.
        room_r = float(eff[self.right_mask].min()) if self.right_mask.any() else np.inf
        room_l = float(eff[self.left_mask].min()) if self.left_mask.any() else np.inf
        in_corridor = room_r < self.corridor_thresh and room_l < self.corridor_thresh
        centering = 0.0
        if in_corridor:
            centering = (room_r - room_l) / max(1.0, room_r + room_l)
            steer = float(np.clip(steer + self.k_center * centering, -1.0, 1.0))

        # Mild low-pass: the discrete beam set makes the argmax jump between
        # adjacent directions on alternating steps.
        steer = float((1.0 - self.steer_smooth) * steer + self.steer_smooth * self.steer_prev)
        self.steer_prev = steer

        # --- speed: whatever we can still stop from, softened for hard steering
        room = max(0.0, clears[best] - self.stop_margin)
        v_safe = np.sqrt(2.0 * self.brake_decel * room)
        v_turn = self.cruise_speed * (1.0 - 0.45 * abs(steer))

        # The `min_speed` floor exists to stop the car dithering to a halt in an
        # open corridor, so it applies to the *comfort* terms only. Every constraint
        # that comes from something real -- LIDAR clearance, a red light, a queue
        # ahead -- has to be able to command an actual zero. With the floor applied
        # after them the car rolled into whatever it had stopped for at 2 m/s, which
        # is how 20 of 35 vehicle collisions happened to a target it had been
        # tracking straight ahead for 15+ steps.
        target = float(np.clip(min(self.cruise_speed, v_turn),
                               self.min_speed, self.cruise_speed))
        target = min(target, float(v_safe))

        # Speed has to respect where the car is actually *going*, not only where it
        # has decided to aim. Those coincide closely enough among static buildings,
        # which is why this was never noticed, but they diverge by more than 5 m on
        # 44% of steps: a vehicle stopped straight ahead while the controller aims
        # 40 deg off to take a turn left the speed term reading open road. Six of
        # eight sampled collisions were at 7-9 m/s with 1-3 m of clearance in front
        # of the bumper and 20-35 m along the chosen heading.
        room_f = max(0.0, self._forward_room(eff) - self.stop_margin)
        target = min(target, float(np.sqrt(2.0 * self.brake_decel * room_f)))

        # Both of these gate on their slice being present in the observation rather
        # than on a constructor flag, so the controller adapts to an env with or
        # without lights and traffic without retuning.
        v_light = self._red_light_stop_speed(tl)
        if v_light is not None:
            target = min(target, v_light)

        v_follow = self._traffic_follow_speed(traf, speed)
        if v_follow is not None:
            target = min(target, v_follow)

        err = target - speed
        if err > 0.25:
            throttle, brake = min(1.0, err / 2.5), 0.0
        elif err < -0.6:
            throttle, brake = 0.0, min(1.0, -err / 5.0)
        else:
            throttle, brake = 0.2, 0.0

        # Expose the decision for debugging / diagnostics.
        self.last = {
            "theta": np.degrees(theta),
            "bearing": np.degrees(bearing),
            "chosen_clear": float(clears[best]),
            "need": float(need),
            "n_ok": int(ok.sum()),
            "fell_back": not bool(ok.any()),
            "target_speed": target,
            "speed": speed,
            "steer": steer,
            "min_clear": float(clear.min()),
            "centering": centering,
            "in_corridor": in_corridor,
            "v_light": v_light,
            "v_follow": v_follow,
        }

        # Env expects every channel in [-1, 1]; throttle/brake are rescaled to [0, 1].
        return np.array([throttle * 2.0 - 1.0, brake * 2.0 - 1.0, steer], dtype=np.float32)
