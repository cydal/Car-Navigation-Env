"""
2D cockpit-dashboard HUD for the live demo window.

Owns only informational panels driven by `env`/`info` -- speed, reward
breakdown, a radar-style nearby-vehicle readout, LIDAR closest-obstacle
summary, and a traffic-light countdown. Camera picture-in-picture feeds and
the minimap stay owned by `PandaRenderer` (they need scene-graph/camera
access this module doesn't have).

Never touched by the training/offscreen path: `main.py` only constructs a
`Hud` for the onscreen demo, so nothing here can affect `capture()`.
"""

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from . import theme

_COMPASS = ("FRONT", "FRONT-R", "RIGHT", "REAR-R", "REAR", "REAR-L", "LEFT", "FRONT-L")


def _compass_label(bearing_rad):
    """Nearest-of-8 compass label for an ego-frame bearing (0 = dead ahead,
    positive = toward the driver's right -- see panda_renderer._panda_dir)."""
    deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    idx = int(round(deg / 45.0)) % 8
    return _COMPASS[idx]


class Hud:
    """Builds its panels once; `update()` only rewrites text/bar content."""

    # (name, x0, x1, z0, z1) in render2d NDC. Deliberately avoids the
    # minimap's top-right corner and the camera PiPs' bottom row.
    _STATUS_FRAME  = (-0.98, -0.56,  0.66, 0.98)
    _SENSOR_FRAME  = (-0.98, -0.56,  0.06, 0.60)
    _SPEED_FRAME   = (-0.24,  0.24, -0.64, -0.42)
    _REWARD_FRAME  = ( 0.56,  0.98,  0.02,  0.50)

    def __init__(self, base):
        self.base = base
        r2d = base.render2d

        self._status_bg, self._status_border = theme.panel_card(
            r2d, *self._STATUS_FRAME, name="status")
        self._status_text = OnscreenText(
            text="", parent=r2d, pos=(self._STATUS_FRAME[0] + 0.02, self._STATUS_FRAME[3] - 0.06),
            scale=0.032, fg=theme.TEXT, shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft, mayChange=True)

        self._sensor_bg, self._sensor_border = theme.panel_card(
            r2d, *self._SENSOR_FRAME, name="sensors")
        self._sensor_text = OnscreenText(
            text="", parent=r2d, pos=(self._SENSOR_FRAME[0] + 0.02, self._SENSOR_FRAME[3] - 0.06),
            scale=0.030, fg=theme.TEXT, shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft, mayChange=True)

        self._speed_bg, self._speed_border = theme.panel_card(
            r2d, *self._SPEED_FRAME, name="speed")
        self._speed_text = OnscreenText(
            text="", parent=r2d,
            pos=((self._SPEED_FRAME[0] + self._SPEED_FRAME[1]) / 2.0,
                 (self._SPEED_FRAME[2] + self._SPEED_FRAME[3]) / 2.0 - 0.03),
            scale=0.075, fg=theme.ACCENT, shadow=(0, 0, 0, 0.8),
            align=TextNode.ACenter, mayChange=True)

        self._reward_bg, self._reward_border = theme.panel_card(
            r2d, *self._REWARD_FRAME, name="reward")
        self._reward_text = OnscreenText(
            text="", parent=r2d, pos=(self._REWARD_FRAME[0] + 0.02, self._REWARD_FRAME[3] - 0.06),
            scale=0.030, fg=theme.TEXT, shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft, mayChange=True)

    # ------------------------------------------------------------------
    def update(self, env, info, mode_label="", controls_text=""):
        """Pure function of current `env`/`info` state -- no history kept here,
        the same discipline `capture()` enforces on the renderer."""
        self._update_status(info, mode_label, controls_text)
        self._update_sensors(env)
        self._update_speed(env, info)
        self._update_reward(info)

    def _update_status(self, info, mode_label, controls_text):
        lines = [
            "CARNAV DRIVE",
            f"step {info.get('step', 0):>5d}   waypoint {info.get('targets_reached', 0)}/"
            f"{info.get('n_targets', 0)}",
            f"dist to target {info.get('dist_to_target', 0):6.1f} m",
            f"mode: {mode_label}" if mode_label else "",
            controls_text,
        ]
        self._status_text.setText("\n".join(l for l in lines if l))

    def _update_sensors(self, env):
        lines = ["SENSORS"]

        lidar = getattr(env, "lidar", None)
        if lidar is not None and len(lidar.last_distances):
            idx = int(np.argmin(lidar.last_distances))
            dist = float(lidar.last_distances[idx])
            bearing = float(lidar.offsets[idx])
            lines.append(f"lidar closest {dist:5.1f} m  {_compass_label(bearing)}")

        radar = getattr(env, "radar", None)
        if radar is not None:
            lines.append("radar:")
            for i in range(radar.n_sectors):
                dist = float(radar.last_distances[i])
                if dist >= radar.max_range - 1e-3:
                    continue
                closing = float(radar.last_closing_speed[i])
                label = _compass_label(radar.sector_bearing(i))
                lines.append(f"  {label:<8s} {dist:5.1f} m  {closing*3.6:+5.1f} km/h")
            if len(lines) == 2:
                lines.append("  (clear)")

        lights = getattr(env, "traffic_lights", None)
        car = getattr(env, "car", None)
        if lights and car is not None:
            nearest = min(lights, key=lambda tl: (tl.x - car.x) ** 2 + (tl.y - car.y) ** 2)
            axis = "ns" if abs(car.y - nearest.y) > abs(car.x - nearest.x) else "ew"
            state = nearest.state(axis)
            secs = nearest.steps_remaining * env.cfg.dt
            lines.append(f"signal {state:<6s} {secs:4.1f}s to change")

        self._sensor_text.setText("\n".join(lines))

    def _update_speed(self, env, info):
        speed_ms = info.get("speed", 0.0)
        self._speed_text.setText(f"{speed_ms * 3.6:5.0f}\nkm/h")

    def _update_reward(self, info):
        comp = info.get("reward_components", {})
        step_reward = sum(comp.values()) if comp else 0.0
        lines = [
            "REWARD",
            f"episode {info.get('episode_reward', 0):8.1f}",
            f"this step {step_reward:+7.2f}",
        ]
        if comp:
            lines.append("-- breakdown --")
            for name in ("progress", "time", "target_bonus", "red_light", "crash"):
                v = comp.get(name, 0.0)
                if abs(v) > 1e-6:
                    lines.append(f"  {name:<12s} {v:+7.2f}")
        self._reward_text.setText("\n".join(lines))

    # ------------------------------------------------------------------
    def close(self):
        for np_ in (self._status_bg, self._status_border, self._sensor_bg,
                    self._sensor_border, self._speed_bg, self._speed_border,
                    self._reward_bg, self._reward_border):
            if np_ is not None:
                np_.removeNode()
        for text in (self._status_text, self._sensor_text, self._speed_text,
                     self._reward_text):
            text.destroy()
