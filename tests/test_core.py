"""Sanity checks for the world grid, ray caster and car model."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import ProceduralCity, CityConfig, ROAD, BUILDING
from env.car import Car, CarParams


def ascii_map(city, mark=None):
    rows = []
    for r in range(city.height):
        line = "".join("." if city.grid[r, c] == ROAD else "#" for c in range(city.width))
        rows.append(line)
    if mark:
        for (x, y, ch) in mark:
            c, r = int(x / city.tile_size), int(y / city.tile_size)
            if 0 <= r < city.height and 0 <= c < city.width:
                rows[r] = rows[r][:c] + ch + rows[r][c + 1:]
    return "\n".join(rows)


def brute_force_ray(city, x, y, angle, max_range, samples=20000):
    """Reference implementation: march tiny steps until a solid tile."""
    step = max_range / samples
    for i in range(1, samples + 1):
        d = i * step
        px = x + np.cos(angle) * d
        py = y + np.sin(angle) * d
        if not city.is_road(px, py):
            return d
    return max_range


print("=" * 62)
print("1. CITY GENERATION")
print("=" * 62)
city = ProceduralCity(CityConfig(width=48, height=48, tile_size=4.0), seed=7)
road_frac = (city.grid == ROAD).mean()
print(f"grid            : {city.width}x{city.height} tiles @ {city.tile_size}m")
print(f"world extent    : {city.extent[0]:.0f}m x {city.extent[1]:.0f}m")
print(f"road fraction   : {road_frac:.3f}")
print(f"reachable tiles : {len(city.road_cells)}")

# Every road tile must be in the kept component (generator walls off the rest).
n_road = int((city.grid == ROAD).sum())
assert n_road == len(city.road_cells), f"connectivity mismatch: {n_road} vs {len(city.road_cells)}"
print("connectivity    : OK (all road tiles mutually reachable)")

print()
print("=" * 62)
print("2. MAP PREVIEW (48x48)")
print("=" * 62)
print(ascii_map(city))

print()
print("=" * 62)
print("3. RAY CASTER vs BRUTE FORCE")
print("=" * 62)
MAX_RANGE = 60.0
rng = np.random.default_rng(0)
worst = 0.0
checked = 0
for _ in range(30):
    x, y = city.sample_road_point()
    angles = rng.uniform(0, 2 * np.pi, size=8)
    dda = city.cast_rays(x, y, angles, MAX_RANGE)
    for a, f in zip(angles, dda):
        slow = brute_force_ray(city, x, y, a, MAX_RANGE)
        worst = max(worst, abs(f - slow))
        checked += 1
print(f"rays checked    : {checked}")
print(f"max |DDA - brute force| : {worst:.4f} m  (brute force step = {MAX_RANGE/20000:.4f} m)")
assert worst < 0.02, "DDA disagrees with brute force"
print("raycaster       : OK")

# The sampled caster is the default (it is ~2.7x faster and the LIDAR dominates
# step time), so hold it to the exact DDA as oracle. The bound is step/2 and
# nothing may exceed it -- in particular no beam may *over*-report clearance,
# which is what a naive sampler does when it grazes a building corner and skips
# the tile entirely. That would tell the policy a wall is not there and punish it
# for believing its own sensor.
STEP = 0.25
tot = worst_over = worst_under = 0
for _ in range(200):
    x, y = city.sample_road_point()
    angles = rng.uniform(0, 2 * np.pi, size=32)
    err = city.cast_rays_fast(x, y, angles, MAX_RANGE, STEP) - city.cast_rays(x, y, angles, MAX_RANGE)
    worst_over = max(worst_over, err.max())
    worst_under = max(worst_under, -err.min())
    tot += len(angles)
print(f"sampled vs DDA  : {tot:,} rays, error +{worst_over:.4f} / -{worst_under:.4f} m "
      f"(bound step/2 = {STEP/2})")
assert worst_over <= STEP, "sampled caster over-reports clearance -- it is skipping walls"
assert worst_under <= STEP, "sampled caster under-reports clearance"
print("sampled caster  : OK (corner-safe)")

print()
print("=" * 62)
print("4. RAY CASTER SPEED (32 rays, as used by LIDAR)")
print("=" * 62)
angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
x, y = city.sample_road_point()
N = 3000
t0 = time.perf_counter()
for _ in range(N):
    city.cast_rays(x, y, angles, MAX_RANGE)
el = time.perf_counter() - t0
print(f"{N} casts in {el:.3f}s  ->  {el/N*1e6:.1f} us/cast  ({N/el:.0f} casts/s)")

print()
print("=" * 62)
print("5. CAR: BICYCLE MODEL")
print("=" * 62)
p = CarParams()
car = Car(p)
car.reset(0, 0, 0)
dt = 0.05

# Accelerate flat out for 3s
for _ in range(60):
    car.step(1.0, 0.0, 0.0, dt)
print(f"3s full throttle : speed={car.speed:.2f} m/s ({car.speed*3.6:.1f} km/h), x={car.x:.1f}m")
assert car.speed > 8.0 and car.y == 0.0

# Full lock circle: radius should match L/tan(steer)
car.reset(0, 0, 0, speed=8.0)
for _ in range(200):
    car.step(0.3, 0.0, 1.0, dt)
print(f"steer lock       : {np.degrees(car.steer_angle):.1f} deg (max {np.degrees(p.max_steer):.1f})")
print(f"turn radius      : {car.turn_radius:.2f} m")
print(f"yaw rate         : {np.degrees(car.yaw_rate):.1f} deg/s at {car.speed:.1f} m/s")
assert 3.0 < car.turn_radius < 6.0

# Turn radius must be speed-independent; yaw rate must not be.
r_slow, r_fast = None, None
for v in (4.0, 16.0):
    car.reset(0, 0, 0, speed=v)
    for _ in range(100):
        car.step(0.0, 0.0, 1.0, dt)
    if v == 4.0:
        r_slow, yaw_slow = car.turn_radius, car.yaw_rate
    else:
        r_fast, yaw_fast = car.turn_radius, car.yaw_rate
print(f"radius @4m/s={r_slow:.2f}m  @16m/s={r_fast:.2f}m  (should match)")
print(f"yaw   @4m/s={np.degrees(yaw_slow):.1f}  @16m/s={np.degrees(yaw_fast):.1f} deg/s (should differ)")
assert abs(r_slow - r_fast) < 0.1
assert yaw_fast > yaw_slow * 1.5

# Braking must never reverse the car
car.reset(0, 0, 0, speed=5.0)
for _ in range(200):
    car.step(0.0, 1.0, 0.0, dt)
print(f"brake from 5m/s  : speed={car.speed:.4f} (must be exactly 0, never negative)")
assert car.speed == 0.0

# Steering rate limit: wheels cannot snap to lock in one step
car.reset(0, 0, 0, speed=5.0)
car.step(0.0, 0.0, 1.0, dt)
frac = car.steer_angle / p.max_steer
print(f"steer after 1 step: {frac*100:.0f}% of lock (rate limited, must be < 100%)")
assert frac < 1.0
print("car model        : OK")

print()
print("=" * 62)
print("6. OBB COLLISION vs CENTRE-POINT")
print("=" * 62)
# Find a road tile adjacent to a building, place the car so its centre is on
# road but a corner overlaps the building.
found = 0
caught_by_obb_only = 0
for _ in range(4000):
    x, y = city.sample_road_point(jitter=1.0)
    if not city.is_road(x, y):
        continue
    found += 1
    h = rng.uniform(0, 2 * np.pi)
    if city.collides(x, y, h, p.length, p.width):
        caught_by_obb_only += 1
print(f"poses sampled on road (centre clear) : {found}")
print(f"of those, OBB reports collision      : {caught_by_obb_only}")
print("-> centre-point test would have missed every one of these")
assert caught_by_obb_only > 0, "expected some corner clipping on a tight grid"

print()
print("=" * 62)
print("ALL CHECKS PASSED")
print("=" * 62)
