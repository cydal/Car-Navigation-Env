"""
Live viewer server: one CarNavEnv stepped in real time, streamed to browsers.

    python main.py serve                    # http://localhost:8765
    python main.py serve --host 0.0.0.0     # reachable from other machines
    ssh -L 8765:localhost:8765 gpu-box      # or tunnel instead of exposing it

The simulation runs wherever training runs -- usually a headless GPU box --
and the person watching is somewhere else. So the server sends *numbers*: a few
kilobytes per tick (ego pose, vehicle poses, signal phases, LIDAR and radar
returns, reward and its components, what the driver decided and why), and the
browser renders them with Three.js on the viewer's own GPU. Nothing graphical
is imported here; this runs on a box with no display and no Panda3D.

Every message is derived from env state after the fact. The only way back in
is the explicit command set in `Server.handle` (pause, reset, manual keys ...),
so watching an env cannot change what it does.
"""

import asyncio
import json
import mimetypes
import random
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from baselines.scripted import GapFollower
from env.nav_env import CarNavEnv, EnvConfig
from env.traffic import VEHICLE_KINDS
from env.world import CityConfig

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
CAR_DIR = ROOT / "kenney_car-kit" / "Models" / "GLB format"

# (moving, parked). Every preset keeps the same observation layout -- the
# traffic block pads empty slots -- so the driver sees the same 73-D vector.
TRAFFIC_PRESETS = {"none": (0, 0), "light": (4, 10), "normal": (8, 18), "dense": (14, 30)}
RESTART_DELAY_S = 3.0
INFO_KEYS = ("episode_reward", "dist_to_target", "target_bearing", "targets_reached",
             "n_targets", "reason", "red_light_violations", "crash_with", "is_success",
             "reward_components")

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("model/gltf-binary", ".glb")


def _py(v):
    """JSON-safe copy: numpy scalars to Python, floats rounded to 4 dp."""
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, 4)
    return v


def _arr(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


class Session:
    """One env + driver, stepped in real time, serialised for the browser."""

    def __init__(self, seed=0, map_size=48, n_targets=3, traffic="normal",
                 manual=False, rate=1.0):
        self.map_size = map_size
        self.n_targets = n_targets
        self.manual = manual
        self.rate = rate
        self.paused = False
        self.step_once = False
        self.auto_restart = True
        self.keys = {"up": False, "down": False, "left": False, "right": False}
        self.episode = 0
        self.seed = int(seed)
        self.build(self.seed, traffic)

    def build(self, seed, traffic):
        n_moving, n_parked = TRAFFIC_PRESETS[traffic]
        self.traffic_preset = traffic
        cfg = EnvConfig(n_targets=self.n_targets, n_traffic=n_moving, n_parked=n_parked)
        city = CityConfig(width=self.map_size, height=self.map_size)
        self.env = CarNavEnv(config=cfg, city_config=city, obs_type="vector", seed=seed)
        self.driver = GapFollower.for_env(self.env)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.driver.reset()
        self.driver.last = None
        self.episode += 1
        self.action = np.zeros(3, dtype=np.float32)
        self.reward = 0.0
        self.done = None
        self.restart_at = None

    def manual_action(self):
        k = self.keys
        return np.array([1.0 if k["up"] else -1.0,
                         1.0 if k["down"] else -1.0,
                         (1.0 if k["right"] else 0.0) - (1.0 if k["left"] else 0.0)],
                        dtype=np.float32)

    def step(self):
        if self.done:
            return
        act = self.manual_action() if self.manual else self.driver.act(self.obs)
        self.action = act
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(act)
        if terminated or truncated:
            self.done = self.info.get("reason") or ("timeout" if truncated else "terminated")
            if self.auto_restart:
                self.restart_at = time.monotonic() + RESTART_DELAY_S

    # ------------------------------------------------------------------
    def driver_state(self):
        """What the scripted driver chose and which constraint bound it.

        GapFollower picks a heading by clearance-vs-goal score, then caps speed
        by the tightest of: cruise, stopping within the chosen clearance, a
        steering-dependent corner speed, the next red/yellow, and a predicted
        traffic conflict. `last` records the inputs; the binding cap is recovered
        here by matching the final target against each cap.
        """
        if self.manual:
            return {"mode": "manual", "intent": "manual"}
        last = getattr(self.driver, "last", None)
        if not last:
            return {"mode": "scripted", "intent": "starting"}
        d = self.driver
        target = float(last["target_speed"])
        room = max(0.0, float(last["chosen_clear"]) - d.stop_margin)
        caps = {
            "cruise": d.cruise_speed,
            "clearance": float(np.sqrt(2.0 * d.brake_decel * room)),
            "turn": d.cruise_speed * (1.0 - 0.45 * abs(float(last["steer"]))),
            "signal": last["v_light"],
            "traffic": last["v_traffic"],
        }
        # The binding cap is the one equal to the final target. Signal and traffic
        # are checked first: they are applied after the creep-floor clip, so when
        # they bind the match is exact. A target below cruise that matches no cap
        # was clipped to the creep floor by the clearance term.
        intent = "cruise"
        if last["fell_back"]:
            intent = "no_safe_heading"
        else:
            for name in ("signal", "traffic", "clearance", "turn"):
                v = caps[name]
                if v is not None and v < d.cruise_speed - 1e-3 and abs(float(v) - target) < 1e-3:
                    intent = name
                    break
            if intent == "cruise" and target < d.cruise_speed - 0.5:
                intent = "clearance"
        return _py({
            "mode": "scripted", "intent": intent,
            "target_speed": target, "speed": last["speed"],
            "theta_deg": last["theta"], "bearing_deg": last["bearing"],
            "chosen_clear": last["chosen_clear"], "need": last["need"],
            "min_clear": last["min_clear"], "steer": last["steer"],
            "in_corridor": bool(last["in_corridor"]), "centering": last["centering"],
            "cruise": d.cruise_speed, "caps": caps,
        })

    def reset_message(self):
        env, city, t = self.env, self.env.city, self.env.traffic
        car = env.car.p
        return {
            "type": "reset", "episode": self.episode, "seed": self.seed,
            "traffic": self.traffic_preset,
            "city": {
                "width": city.width, "height": city.height, "tile_size": city.tile_size,
                "grid": "".join(map(str, city.grid.ravel().tolist())),
                "road_width": city.cfg.road_width,
                "v_roads": city.v_roads, "h_roads": city.h_roads,
                "signals": _arr(np.asarray(city.signals, dtype=float).reshape(-1, 2)),
            },
            "targets": _arr(env.targets), "target_radius": env.cfg.target_radius,
            "vehicles": {
                "kind": [VEHICLE_KINDS[int(k)][0] for k in t.kind],
                "length": _arr(t.length, 2), "width": _arr(t.width, 2),
                "n_moving": int(t.n_moving),
            },
            "car": {"length": car.length, "width": car.width, "wheelbase": car.wheelbase,
                    "max_speed": car.max_speed, "max_steer": float(car.max_steer)},
            "cfg": {"dt": env.cfg.dt, "max_steps": env.cfg.max_episode_steps,
                    "n_targets": env.cfg.n_targets, "lidar_range": env.cfg.lidar_range,
                    "n_beams": env.cfg.n_beams, "radar_range": env.cfg.radar_range,
                    "n_radar_sectors": env.cfg.n_radar_sectors,
                    "cruise_speed": self.driver.cruise_speed},
            "lidar_offsets": _arr(env.lidar.offsets, 4),
        }

    def tick_message(self):
        env, car, t = self.env, self.env.car, self.env.traffic
        info = self.info
        restart_in = None
        if self.done and self.restart_at is not None:
            restart_in = round(max(0.0, self.restart_at - time.monotonic()), 2)
        return {
            "type": "tick", "step": int(env.step_count),
            "time": round(env.step_count * env.cfg.dt, 2),
            "ego": {"x": round(car.x, 3), "y": round(car.y, 3),
                    "heading": round(car.heading, 4), "speed": round(car.speed, 3),
                    "steer": round(car.steer_angle, 4), "accel": round(car.accel, 2)},
            "action": _arr(self.action, 2),
            "vehicles": {"x": _arr(t.x), "y": _arr(t.y),
                         "heading": _arr(t.heading), "speed": _arr(t.speed, 2)},
            "lights": [{"ns": tl.ns_state, "ew": tl.ew_state, "left": int(tl.steps_remaining)}
                       for tl in env.traffic_lights],
            "target_idx": int(env.target_idx),
            "lidar": _arr(env.lidar.last_distances, 2),
            "radar": {"dist": _arr(env.radar.last_distances, 1),
                      "closing": _arr(env.radar.last_closing_speed, 2),
                      "idx": env.radar.last_target_idx.tolist()},
            "reward": round(float(self.reward), 4),
            "info": _py({k: info.get(k) for k in INFO_KEYS}),
            "driver": self.driver_state(),
            "paused": self.paused, "manual": self.manual, "rate": self.rate,
            "auto_restart": self.auto_restart,
            "done": self.done, "restart_in": restart_in,
        }


class Server:
    def __init__(self, session):
        self.session = session
        self.clients = set()

    # --- websocket -----------------------------------------------------
    async def ws_handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps(self.session.reset_message()))
            await ws.send(json.dumps(self.session.tick_message()))
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(cmd, dict):
                    self.handle(cmd)
        finally:
            self.clients.discard(ws)

    def handle(self, cmd):
        s = self.session
        kind = cmd.get("cmd")
        if kind == "toggle":
            s.paused = not s.paused
        elif kind == "pause":
            s.paused = True
        elif kind == "resume":
            s.paused = False
        elif kind == "step":
            s.step_once = True
        elif kind == "reset":
            seed = cmd.get("seed", s.seed)
            s.reset(seed=int(seed) if isinstance(seed, (int, float)) else s.seed)
            s.paused = False
            self.send_all(s.reset_message())
        elif kind == "new":
            s.reset(seed=random.randrange(1, 100000))
            s.paused = False
            self.send_all(s.reset_message())
        elif kind == "traffic":
            preset = cmd.get("traffic")
            if preset in TRAFFIC_PRESETS:
                s.build(s.seed, preset)
                self.send_all(s.reset_message())
        elif kind == "mode":
            s.manual = bool(cmd.get("manual"))
            s.keys = {k: False for k in s.keys}
        elif kind == "auto":
            s.auto_restart = bool(cmd.get("on"))
            if s.done:
                s.restart_at = (time.monotonic() + RESTART_DELAY_S) if s.auto_restart else None
        elif kind == "rate":
            try:
                s.rate = float(min(8.0, max(0.1, float(cmd.get("rate", 1.0)))))
            except (TypeError, ValueError):
                pass
        elif kind == "keys":
            s.keys = {k: bool(cmd.get(k)) for k in s.keys}
        self.send_all(s.tick_message())

    def send_all(self, msg):
        data = json.dumps(msg)
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, data))

    @staticmethod
    async def _send(ws, data):
        try:
            await ws.send(data)
        except Exception:
            pass                      # the handler's finally block drops the client

    # --- sim loop --------------------------------------------------------
    async def loop(self):
        s = self.session
        nxt = time.monotonic()
        while True:
            now = time.monotonic()
            if s.done and s.restart_at is not None and now >= s.restart_at:
                s.reset(seed=random.randrange(1, 100000))
                self.send_all(s.reset_message())
            elif not s.paused or s.step_once:
                s.step_once = False
                s.step()
            self.send_all(s.tick_message())
            nxt += s.env.cfg.dt / max(0.1, s.rate)
            delay = nxt - time.monotonic()
            if delay < -1.0:          # fell far behind (machine asleep): resync, don't burst
                nxt = time.monotonic()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))

    # --- static files on the same port -------------------------------------
    def process_request(self, connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None               # let the websocket handshake proceed
        return self.static(request.path)

    def static(self, path):
        path = unquote(path.split("?", 1)[0])
        if path in ("", "/"):
            path = "/index.html"
        if path.startswith("/assets/cars/"):
            root, rel = CAR_DIR, path[len("/assets/cars/"):]
        else:
            root, rel = WEB_DIR, path.lstrip("/")
        file = (root / rel).resolve()
        if not (file.is_relative_to(root.resolve()) and file.is_file()):
            return Response(404, "Not Found", Headers({"Content-Type": "text/plain"}), b"not found")
        body = file.read_bytes()
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        return Response(200, "OK", Headers({
            "Content-Type": ctype, "Content-Length": str(len(body)),
            "Cache-Control": "no-store"}), body)


def main(host="127.0.0.1", port=8765, **session_kw):
    session = Session(**session_kw)
    server = Server(session)

    async def run():
        async with serve(server.ws_handler, host, port,
                         process_request=server.process_request, max_size=64 * 1024):
            shown = "localhost" if host in ("", "0.0.0.0", "127.0.0.1") else host
            print(f"viewer:  http://{shown}:{port}    (bound to {host}; ctrl-c stops)", flush=True)
            await server.loop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
