"""Perception contract: FOV/range detection, vehicle occlusion, blind spots.

RNG-free and env-side only: builds a bare `Traffic`-shaped stand-in with a
handful of hand-placed vehicles rather than a full procedural `CarNavEnv`
world, so each case is exact and doesn't depend on spawn randomness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.car import Car
from env.perception import Perception, CAMERAS, BLIND_LATERAL_M, BLIND_HALF_WIDTH_M


class FakeTraffic:
    """Just enough of Traffic's interface for Perception.scan to use."""
    def __init__(self, rows):
        # rows: list of (x, y, heading, length, width)
        self.n_moving = len(rows)
        self.x = np.array([r[0] for r in rows], dtype=float)
        self.y = np.array([r[1] for r in rows], dtype=float)
        self.heading = np.array([r[2] for r in rows], dtype=float)
        self.length = np.array([r[3] for r in rows], dtype=float)
        self.width = np.array([r[4] for r in rows], dtype=float)


def car_at(x=0.0, y=0.0, heading=0.0):
    c = Car()
    c.reset(x, y, heading)
    return c


print("=" * 62)
print("1. FOV / RANGE")
print("=" * 62)
car = car_at(0, 0, 0.0)                       # facing world +x
p = Perception()

# Dead ahead, well inside front's 200 m range and 70 deg FOV.
p.scan(FakeTraffic([(20.0, 0.0, np.pi, 4.4, 1.9)]), car)
assert len(p.cameras["front"]) == 1 and p.cameras["front"][0][0] == 0
assert abs(p.cameras["front"][0][2]) < 1e-6, "dead ahead must be ~0 bearing"
print("front sees a car 20 m dead ahead : OK")

# Same distance, but far outside the front camera's FOV (behind, off to the side).
p.scan(FakeTraffic([(-20.0, 0.0, 0.0, 4.4, 1.9)]), car)
assert p.cameras["front"] == [], "directly behind must not appear in the front camera"
print("a car directly behind is not in the front camera : OK")

# Outside range entirely (front's range is 200 m).
fwd, right, yaw, fov, rng = CAMERAS["front"]
p.scan(FakeTraffic([(rng + 5.0, 0.0, np.pi, 4.4, 1.9)]), car)
assert p.cameras["front"] == [], "beyond max range must not be detected"
print(f"a car beyond {rng:.0f} m range is not detected : OK")

print()
print("=" * 62)
print("2. OCCLUSION (vehicles only)")
print("=" * 62)
# Two cars dead ahead on the same line: a near one directly in front of the
# camera, and a far one behind it -- the far one must be occluded.
near = (10.0, 0.0, np.pi, 4.4, 1.9)
far = (30.0, 0.0, np.pi, 4.4, 1.9)
p.scan(FakeTraffic([near, far]), car)
seen_ids = {d[0] for d in p.cameras["front"]}
assert seen_ids == {0}, f"expected only the near vehicle (0), got {seen_ids}"
print("a farther vehicle directly behind a nearer one is occluded : OK")

# Move the "blocker" off to the side: both should now be visible.
near_offset = (10.0, 3.0, np.pi, 4.4, 1.9)
p.scan(FakeTraffic([near_offset, far]), car)
seen_ids = {d[0] for d in p.cameras["front"]}
assert seen_ids == {0, 1}, f"expected both vehicles once the near one is offset, got {seen_ids}"
print("moving the near vehicle aside un-occludes the far one : OK")

print()
print("=" * 62)
print("3. BLIND SPOTS")
print("=" * 62)
car2 = car_at(0, 0, 0.0)
p.scan(FakeTraffic([]), car2)
assert p.blind_spots["left"]["occupied"] is False and p.blind_spots["right"]["occupied"] is False
print("no vehicles -> both blind spots clear : OK")

# A vehicle sitting in the right blind-spot zone (perpendicular offset to the
# right of a car facing +x is world +y, per this module's body-frame convention).
in_zone = (-2.0, BLIND_LATERAL_M - 0.2, np.pi, 4.4, 1.9)
p.scan(FakeTraffic([in_zone]), car2)
assert p.blind_spots["right"]["occupied"] is True and p.blind_spots["right"]["vehicle"] == 0
assert p.blind_spots["left"]["occupied"] is False
print("a vehicle in the right-side zone flags 'right', not 'left' : OK")

# Same lateral offset, but far ahead of the zone's span -- must not trigger.
too_far_ahead = (40.0, BLIND_LATERAL_M, np.pi, 4.4, 1.9)
p.scan(FakeTraffic([too_far_ahead]), car2)
assert p.blind_spots["right"]["occupied"] is False, "well ahead of the zone must not count"
print("a vehicle far ahead of the zone's span does not count : OK")

# Just outside the lateral half-width -- must not trigger either.
just_outside = (-2.0, BLIND_LATERAL_M + BLIND_HALF_WIDTH_M + 0.5, np.pi, 4.4, 1.9)
p.scan(FakeTraffic([just_outside]), car2)
assert p.blind_spots["right"]["occupied"] is False, "outside the zone's lateral half-width must not count"
print("a vehicle just outside the zone's width does not count : OK")

print()
print("=" * 62)
print("4. AUXILIARY, NOT AN OBSERVATION -- sanity check against a real env")
print("=" * 62)
from env.world import CityConfig
from env.nav_env import CarNavEnv, EnvConfig

env = CarNavEnv(config=EnvConfig(n_traffic=6, n_parked=4),
               city_config=CityConfig(width=40, height=40), obs_type="vector", seed=9)
obs, info = env.reset(seed=9)
assert obs.shape == (env.vector_dim,)
assert set(env.perception.cameras) == set(CAMERAS)
for _ in range(20):
    obs, *_ = env.step(np.array([1.0, -1.0, 0.05], dtype=np.float32))
assert obs.shape == (env.vector_dim,), "perception must never change vector_dim"
print(f"env.perception present, vector_dim unchanged at {env.vector_dim}-D : OK")

print()
print("=" * 62)
print("ALL PERCEPTION CHECKS PASSED")
print("=" * 62)
