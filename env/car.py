"""
Kinematic bicycle model.

Chosen over a point-mass model because it gives a finite turning radius that
tightens as the car slows -- the car cannot pivot in place, and taking a corner
fast requires braking first. That coupling is what makes driving a control
problem rather than a grid-walk.

The steering actuator is rate limited, so the commanded steer angle is a target
the front wheels move toward rather than snap to. The resulting steer angle is
hidden state: the agent must anticipate its own turn-in lag.
"""

import numpy as np


class CarParams:
    """Vehicle geometry and powertrain limits (SI units)."""

    def __init__(
        self,
        wheelbase=2.5,
        length=4.4,
        width=1.9,
        max_steer=np.radians(32.0),
        steer_rate=np.radians(150.0),
        max_accel=4.5,
        max_brake=9.0,
        rolling_resist=0.4,
        drag=0.0035,
        max_speed=22.0,
    ):
        self.wheelbase = wheelbase          # front-to-rear axle distance
        self.length = length                # collision footprint
        self.width = width
        self.max_steer = max_steer          # steering lock
        self.steer_rate = steer_rate        # rad/s the wheels can be turned
        self.max_accel = max_accel          # m/s^2 at full throttle
        self.max_brake = max_brake          # m/s^2 at full brake
        self.rolling_resist = rolling_resist  # constant coast-down decel
        self.drag = drag                    # quadratic air drag coefficient
        self.max_speed = max_speed          # ~80 km/h


class Car:
    """A single vehicle integrated with a kinematic bicycle model."""

    def __init__(self, params=None):
        self.p = params or CarParams()
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, heading, speed=0.0):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.steer_angle = 0.0      # actual front wheel angle (rad)
        self.yaw_rate = 0.0         # rad/s, for observations
        self.slip = 0.0             # body slip angle at centre of mass
        self.accel = 0.0            # last longitudinal acceleration

    def step(self, throttle, brake, steer_cmd, dt):
        """Advance one timestep.

        throttle, brake in [0, 1]; steer_cmd in [-1, 1] scaled to steering lock.
        """
        p = self.p
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        # --- steering actuator: move toward the commanded angle at a bounded rate
        target = steer_cmd * p.max_steer
        max_delta = p.steer_rate * dt
        self.steer_angle += np.clip(target - self.steer_angle, -max_delta, max_delta)

        # --- longitudinal dynamics
        v0 = self.speed
        drive = throttle * p.max_accel
        resist = brake * p.max_brake + p.rolling_resist + p.drag * v0 * v0
        v = v0 + drive * dt
        # Braking and friction decelerate but never push the car backwards.
        v = max(0.0, v - resist * dt)
        v = min(v, p.max_speed)
        self.accel = (v - v0) / dt
        self.speed = v

        # --- bicycle kinematics about the centre of mass
        # slip angle for a CoM at the wheelbase midpoint
        self.slip = np.arctan(0.5 * np.tan(self.steer_angle))
        self.yaw_rate = v * np.cos(self.slip) * np.tan(self.steer_angle) / p.wheelbase

        self.x += v * np.cos(self.heading + self.slip) * dt
        self.y += v * np.sin(self.heading + self.slip) * dt
        self.heading = (self.heading + self.yaw_rate * dt) % (2.0 * np.pi)

    @property
    def pose(self):
        return self.x, self.y, self.heading

    @property
    def turn_radius(self):
        """Current turning radius in metres (inf when going straight)."""
        if abs(self.steer_angle) < 1e-6:
            return np.inf
        return self.p.wheelbase / np.tan(abs(self.steer_angle))
