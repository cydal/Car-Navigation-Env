"""Viewer server contract: JSON-safe messages, known intents, safe static serving,
agent pluggability.

Runs without a browser or a network port. Skips cleanly when `websockets` is
not installed: the viewer is an optional extra and the core suites must stay
runnable without it.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import websockets  # noqa: F401
except ImportError:
    print("websockets not installed -- viewer checks skipped (pip install websockets)")
    sys.exit(0)

from serve.server import Session, Server, TRAFFIC_PRESETS, WORLD_PRESETS  # noqa: E402
from env.perception import CAMERAS  # noqa: E402
from agents import load_agent, RandomAgent, ManualAgent  # noqa: E402

INTENTS = {"cruise", "traffic", "clearance", "turn", "signal", "manual",
           "no_safe_heading", "starting"}

print("=" * 62)
print("1. RESET / TICK MESSAGES SERIALISE")
print("=" * 62)
s = Session(seed=3, map_size=32, traffic="normal")
r = json.loads(json.dumps(s.reset_message()))
assert r["type"] == "reset" and len(r["city"]["grid"]) == 32 * 32
assert set(r["city"]["grid"]) <= {"0", "1"}
assert r["vehicles"]["n_moving"] == TRAFFIC_PRESETS["normal"][0]
assert len(r["vehicles"]["kind"]) == sum(TRAFFIC_PRESETS["normal"])
assert len(r["lidar_offsets"]) == r["cfg"]["n_beams"]
assert len(r["city"]["v_roads"]) > 1 and len(r["city"]["h_roads"]) > 1
assert set(r["cameras"]) == set(CAMERAS), "camera specs must match env.perception.CAMERAS exactly"
for _ in range(40):
    s.step()
t = json.loads(json.dumps(s.tick_message()))
assert t["type"] == "tick" and t["step"] == 40
assert len(t["lidar"]) == r["cfg"]["n_beams"]
assert len(t["radar"]["dist"]) == r["cfg"]["n_radar_sectors"]
assert len(t["vehicles"]["x"]) == len(r["vehicles"]["kind"])
assert len(t["lights"]) == len(r["city"]["signals"])
assert set(t["perception"]["cameras"]) == set(CAMERAS)
assert set(t["perception"]["blind_spots"]) == {"left", "right"}
assert t["driver"]["intent"] in INTENTS
assert t["driver"]["kind"] == "GapFollower" and t["driver"]["manual"] is False
assert set(t["info"]["reward_components"]) == {"time", "crash", "progress", "target_bonus", "red_light"}
print(f"reset {len(json.dumps(r)):,} B, tick {len(json.dumps(t)):,} B, intent now '{t['driver']['intent']}' : OK")

print()
print("=" * 62)
print("2. AGENT DIAGNOSTICS OVER AN EPISODE")
print("=" * 62)
seen = set()
for _ in range(1000):
    s.step()
    seen.add(s.agent_diagnostics()["intent"])
    if s.done:
        break
assert seen <= INTENTS, seen - INTENTS
assert s.done or s.env.step_count == 1040
print(f"intents seen {sorted(seen)}, episode ended: {s.done} : OK")

print()
print("=" * 62)
print("3. COMMANDS")
print("=" * 62)
srv = Server(s)                                  # no clients: send_all is a no-op
srv.handle({"cmd": "mode", "manual": True})
assert s.manual and s.agent_diagnostics()["kind"] == "ManualAgent"
assert s.agent_diagnostics()["intent"] == "manual"
srv.handle({"cmd": "keys", "up": True, "left": True})
assert s.manual_agent.act(s.obs).tolist() == [1.0, -1.0, -1.0]
srv.handle({"cmd": "mode", "manual": False})
assert not s.manual and not any(s.manual_agent.keys.values()), "toggling mode resets the manual agent's keys"
assert s.agent_diagnostics()["kind"] == "GapFollower"
srv.handle({"cmd": "traffic", "traffic": "dense"})
assert s.traffic_preset == "dense" and s.env.cfg.n_traffic == TRAFFIC_PRESETS["dense"][0]
srv.handle({"cmd": "traffic", "traffic": "nonsense"})
assert s.traffic_preset == "dense", "unknown preset must be ignored"
srv.handle({"cmd": "reset", "seed": 11})
assert s.seed == 11 and s.done is None and s.env.step_count == 0
srv.handle({"cmd": "rate", "rate": 99})
assert s.rate == 8.0, "rate is clamped"
srv.handle({"cmd": "toggle"})
assert s.paused
srv.handle({"cmd": "step"})
assert s.step_once
print("mode / keys / traffic / reset / rate / pause / step : OK")

print()
print("=" * 62)
print("3b. WORLD PRESETS")
print("=" * 62)
for name in WORLD_PRESETS:
    s.build(seed=5, traffic="normal", world=name)
    r = json.loads(json.dumps(s.reset_message()))
    assert r["world"] == name and r["city"]["theme"] == WORLD_PRESETS[name][0]
    assert len(r["city"]["grid"]) == r["city"]["width"] * r["city"]["height"]
    s.step()
s.build(s.seed, "dense", world="city")
srv.handle({"cmd": "world", "world": "rural"})
assert s.world == "rural" and s.traffic_preset == "dense", "world switch must keep the current traffic preset"
srv.handle({"cmd": "world", "world": "not-a-world"})
assert s.world == "rural", "unknown world preset must be ignored"
print(f"presets {sorted(WORLD_PRESETS)}, switching preserves traffic preset : OK")

print()
print("=" * 62)
print("4. STATIC FILES")
print("=" * 62)
page = srv.static("/")
assert page.status_code == 200 and b"<canvas" in page.body
assert srv.static("/vendor/three.module.js").status_code == 200
assert srv.static("/assets/cars/race.glb").status_code == 200
assert srv.static("/assets/cars/Textures/colormap.png").status_code == 200
assert srv.static("/app.mjs").headers["Content-Type"].startswith("text/javascript")
for bad in ("/../serve/server.py", "/assets/cars/../../main.py", "/%2e%2e/main.py", "/nope.js"):
    assert srv.static(bad).status_code == 404, bad
print("index, vendor, models, texture served; traversal and unknown paths 404 : OK")

print()
print("=" * 62)
print("5. AGENT LOADING (Session never imports a specific controller)")
print("=" * 62)
env_for_agents = s.env
for spec, cls in (("scripted", "GapFollower"), ("manual", "ManualAgent"), ("random", "RandomAgent")):
    agent = load_agent(spec, env_for_agents)
    assert type(agent).__name__ == cls
    agent.reset()
    act = agent.act(s.obs)
    assert act.shape == (3,)
assert isinstance(load_agent("random", env_for_agents), RandomAgent)
assert isinstance(load_agent("manual", env_for_agents), ManualAgent)

with tempfile.TemporaryDirectory() as d:
    # A throwaway custom agent module + a JSON config pointing at it, exactly
    # the shape a real model codebase would hand to --agent.
    sys.path.insert(0, d)
    with open(os.path.join(d, "toy_agent.py"), "w") as f:
        f.write(
            "class ConstantAgent:\n"
            "    def __init__(self, env, value=0.5):\n"
            "        self.value = value\n"
            "    def reset(self):\n"
            "        pass\n"
            "    def act(self, obs, info=None):\n"
            "        import numpy as np\n"
            "        return np.full(3, self.value, dtype=np.float32)\n")
    cfg_path = os.path.join(d, "toy_agent.json")
    with open(cfg_path, "w") as f:
        json.dump({"module": "toy_agent", "factory": "ConstantAgent", "kwargs": {"value": 0.25}}, f)
    custom = load_agent(cfg_path, env_for_agents)
    assert custom.act(s.obs).tolist() == [0.25, 0.25, 0.25]
    sys.path.remove(d)

try:
    load_agent("not-a-real-agent-and-not-a-file.json", env_for_agents)
    raise AssertionError("load_agent must reject an unknown spec")
except ValueError:
    pass
print("built-ins (scripted/manual/random) + a custom JSON-configured agent : OK")

print()
print("=" * 62)
print("ALL VIEWER CHECKS PASSED")
print("=" * 62)
