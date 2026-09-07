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
        n_tl_obs=0,
        tl_obs_range=60.0,
        tl_stop_margin=2.0,
    ):
        self.n_beams = n_beams
        self.n_lookahead = n_lookahead
        self.n_tl_obs = n_tl_obs
        self.tl_obs_range = tl_obs_range
        self.tl_stop_margin = tl_stop_margin  # metres to stop short of the stop line
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

        # Side beams, for measuring how far off-centre we are in a street.
        deg = np.degrees(self.offsets)
        self.right_mask = (deg > 65.0) & (deg < 115.0)     # +offset == toward +y
        self.left_mask = (deg < -65.0) & (deg > -115.0)

        self.steer_prev = 0.0

    def reset(self):
        """Clear the steering filter between episodes."""
        self.steer_prev = 0.0

    # ------------------------------------------------------------------
    def unpack(self, obs):
        """Split the flat observation into scan / dynamics / nav / traffic-light."""
        n = self.n_beams
        nav_end = n + 5 + 3 * self.n_lookahead
        return obs[:n], obs[n:n + 5], obs[n + 5:nav_end], obs[nav_end:]

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
        scan, dyn, nav, tl = self.unpack(obs)
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
        target = float(np.clip(min(self.cruise_speed, v_safe, v_turn), self.min_speed, self.cruise_speed))

        # A red light is the one case where a full stop is correct, so it has to
        # bypass the min_speed floor -- that floor exists to stop the car dithering
        # in a corridor, and applying it here would make obeying a red impossible.
        # Gated on the slice actually being present, not on a constructor flag, so
        # the controller adapts to an env with or without lights without retuning.
        v_light = self._red_light_stop_speed(tl)
        if v_light is not None:
            target = min(target, v_light)

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
        }

        # Env expects every channel in [-1, 1]; throttle/brake are rescaled to [0, 1].
        return np.array([throttle * 2.0 - 1.0, brake * 2.0 - 1.0, steer], dtype=np.float32)
