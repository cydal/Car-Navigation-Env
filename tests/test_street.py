"""Pedestrians, zebra crossings and speed signs: layout, dynamics, penalties.

Both features are opt-in, so the first check is that leaving them off changes
nothing -- same seed, same first 73 observation values, same spawn -- and the
rest exercises them switched on, with a hand-positioned ego where the timing of
a crossing has to be exact.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.world import CityConfig
from env.nav_env import CarNavEnv, EnvConfig
from env.crossings import PED_FEATURES, SIGN_FEATURES, KERB_INSET
from env.traffic import DIRS, _right
from baselines.scripted import GapFollower

CITY = CityConfig(width=48, height=48)


def mk(seed=5, **kw):
    return CarNavEnv(config=EnvConfig(**kw), city_config=CITY, obs_type="vector", seed=seed)


print("=" * 62)
print("1. OPT-IN: DEFAULTS ARE UNTOUCHED")
print("=" * 62)
off = mk(); o_off, _ = off.reset(seed=5)
on = mk(pedestrians=True, speed_signs=True); o_on, _ = on.reset(seed=5)
assert off.vector_dim == 73 and off.street.n_crossings == 0 and off.street.n_pedestrians == 0
assert on.vector_dim == 73 + 4 * PED_FEATURES + SIGN_FEATURES == 96
assert np.array_equal(o_off, o_on[:73]), "flags on must not change a byte of the first 73 dims"
assert (off.car.x, off.car.y, off.car.heading) == (on.car.x, on.car.y, on.car.heading)
assert off.targets == on.targets
sl = on.obs_slices
assert sl["pedestrians"] == slice(73, 93) and sl["signs"] == slice(93, 96)
for kw, dim in ((dict(pedestrians=True), 93), (dict(speed_signs=True), 76)):
    e = mk(**kw); e.reset(seed=5)
    assert e.vector_dim == dim == len(e.reset(seed=5)[0]), kw
on2 = mk(pedestrians=True, speed_signs=True); on2.reset(seed=5)
assert np.array_equal(on.street.cx, on2.street.cx) and np.array_equal(on.street.trigger, on2.street.trigger)
print("73 -> 93 / 76 / 96; first 73 dims, spawn and targets identical; street seed-deterministic : OK")

print()
print("=" * 62)
print("2. LAYOUT: CROSSINGS, ZONES, SIGNS")
print("=" * 62)
st, city = on.street, on.city
assert st.n_crossings == on.cfg.n_crossings and st.n_pedestrians > 0
assert len(st.signs) == 4 * st.n_crossings, "two 30 signs + two 50 signs per crossing"
sig = np.asarray(city.signals, dtype=float).reshape(-1, 2)
for i in range(st.n_crossings):
    x, y = st.cx[i], st.cy[i]
    assert city.is_road(x, y), "crossing centre must be on the road"
    if len(sig):
        assert np.hypot(sig[:, 0] - x, sig[:, 1] - y).min() > city.tile_size, "never on a signalised junction"
    assert abs(st.limit_at(x, y) - 30 / 3.6) < 1e-9, "30 zone at the crossing itself"
for sx, sy, lim, fdx, fdy in st.signs:
    assert city.is_road(sx, sy), "signs stand on the kerb strip inside the corridor"
    assert abs(np.hypot(fdx, fdy) - 1.0) < 1e-9
# Somewhere far from every crossing the default limit applies.
cells = city.road_cells
far = None
for r, c in cells:
    x, y = (c + 0.5) * city.tile_size, (r + 0.5) * city.tile_size
    if np.hypot(st.cx - x, st.cy - y).min() > on.cfg.zone_len + st.half + 1.0:
        far = (x, y); break
assert far is not None and abs(st.limit_at(*far) - 50 / 3.6) < 1e-9
print(f"{st.n_crossings} crossings, {st.n_pedestrians} people, {len(st.signs)} signs; 30 at crossings, 50 elsewhere : OK")

print()
print("=" * 62)
print("3. DYNAMICS: TRIGGER, CROSS, RE-ARM, YIELD, COLLIDE")
print("=" * 62)
env = mk(pedestrians=True, speed_signs=True); env.reset(seed=5)
st, car = env.street, env.car
i = 0                                              # study crossing 0
a, p = DIRS[st.cdir[i]], DIRS[_right(st.cdir[i])]
group = np.nonzero(st.pc == i)[0]
assert group.size > 0
# Park the ego 30 m before the crossing, facing it: inside every trigger radius.
car.reset(st.cx[i] - 30.0 * a[0], st.cy[i] - 30.0 * a[1], float(np.arctan2(a[1], a[0])), speed=0.0)
dt = env.cfg.dt
st.step(dt, car)
assert np.isfinite(st.start_at[group]).all(), "approaching within trigger range must schedule the group"
for _ in range(int(3.0 / dt)):                     # delays are <= 2.2 s
    st.step(dt, car)
assert (st.state[group] == 1).all(), "everyone should be on the road by now"
assert st.occupied()[i] and st.on_road().any()
# The stuck exemption is deliberately local: a car 30 m back is not "held" by
# anything, one stopped 12 m short of people on the zebra is.
assert not st.yielding(car)
car.reset(st.cx[i] - 12.0 * a[0], st.cy[i] - 12.0 * a[1], car.heading)
assert st.yielding(car), "people on the zebra just ahead hold the ego (stuck exemption)"
car.reset(st.cx[i] - 30.0 * a[0], st.cy[i] - 30.0 * a[1], car.heading)
lat0 = st.lat[group].copy(); side0 = st.side[group].copy()
# Half the road is 6 m; at >= 1.1 m/s the slowest person needs < 11 s to cross 11.2 m.
for _ in range(int(12.0 / dt)):
    st.step(dt, car)
assert (st.state[group] == 0).all(), "everyone has reached the far kerb"
assert (st.side[group] == -side0).all(), "they now wait on the opposite kerb"
assert np.allclose(np.abs(st.lat[group]), st.half - KERB_INSET)
assert not st.armed[group].any() and not np.isfinite(st.start_at[group]).any(), "not re-triggered while the ego is still close"
assert not st.yielding(car)
# Drive the ego far away and back: the group re-arms and can cross again.
car.reset(st.cx[i] - 400.0 * a[0], st.cy[i] - 400.0 * a[1], car.heading)
st.step(dt, car)
assert st.armed[group].all()
car.reset(st.cx[i] - 30.0 * a[0], st.cy[i] - 30.0 * a[1], car.heading)
st.step(dt, car)
assert np.isfinite(st.start_at[group]).all(), "re-armed group triggers again on the next approach"
# Observation slot for the nearest person is a real unit bearing; LIDAR sees them too.
feats = st.obs_features(car, 4, env.cfg.ped_obs_range, car.p.max_speed)
assert feats.shape == (20,) and abs(np.hypot(feats[1], feats[2]) - 1.0) < 1e-5
assert len(st.circles(car.x, car.y, 60.0)) >= group.size
# Put the car right on top of someone: that is a hit.
j = group[0]
car.reset(st.px[j], st.py[j], car.heading)
assert st.hits(car)
car.reset(st.px[j] + 10.0 * p[0], st.py[j] + 10.0 * p[1], car.heading)
assert not st.hits(car)
print("scheduled -> crossing -> far kerb -> re-armed; yielding, obs slot, LIDAR circles, hit test : OK")

print()
print("=" * 62)
print("4. TRAFFIC YIELDS TO AN OCCUPIED CROSSING")
print("=" * 62)
class FakeStreet:
    def __init__(self, cx, cy, occ): self.cx, self.cy, self.half, self._occ = np.array(cx), np.array(cy), 6.0, np.array(occ)
    @property
    def n_crossings(self): return len(self.cx)
    def occupied(self): return self._occ
t = env.traffic
t.n_moving = 1
t.x, t.y, t.heading = np.array([0.0]), np.array([0.0]), np.array([0.0])
t.length = np.array([4.4])
ahead = t._crossing_target(FakeStreet([20.0], [0.0], [True]))[0]
behind = t._crossing_target(FakeStreet([-20.0], [0.0], [True]))[0]
beside = t._crossing_target(FakeStreet([20.0], [8.0], [True]))[0]
empty = t._crossing_target(FakeStreet([20.0], [0.0], [False]))[0]
expect = np.sqrt(2 * t.A_DECEL * (20.0 - 1.5 - t.STOP_MARGIN - 2.2))
assert abs(ahead - expect) < 1e-9, (ahead, expect)
assert behind == np.inf and beside == np.inf and empty == np.inf
print(f"cap {ahead:.1f} m/s for a busy zebra 20 m ahead; none behind / off-corridor / empty : OK")

print()
print("=" * 62)
print("5. PENALTIES AND THE BASELINE")
print("=" * 62)
env = mk(pedestrians=True, speed_signs=True); o, info = env.reset(seed=5)
assert info["speed_limit"] in (30 / 3.6, 50 / 3.6) and info["pedestrians_on_road"] == 0
# Speeding: hold full throttle inside a 30 zone -> a negative 'speeding' component appears.
st = env.street
env.car.reset(st.cx[0], st.cy[0], 0.0, speed=15.0)
_, r, _, _, info = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
assert info["reward_components"]["speeding"] < 0.0, info["reward_components"]
assert set(info["reward_components"]) >= {"pedestrian", "speeding"}
# Hitting a person: its own penalty, its own component, its own crash_with.
env.reset(seed=5)
j = 0
env.car.reset(env.street.px[j], env.street.py[j], 0.0, speed=0.0)
_, r, te, tr, info = env.step(np.array([0.0, -1.0, 0.0], dtype=np.float32))
assert te and info["crash_with"] == "pedestrian"
assert abs(info["reward_components"]["pedestrian"] + env.cfg.pedestrian_penalty) < 1e-9
assert info["reward_components"]["crash"] == 0.0
# The baseline unpacks the wider vector and drives.
env.reset(seed=5)
drv = GapFollower.for_env(env)
o, _ = env.reset(seed=5); drv.reset()
for _ in range(200):
    o, r, te, tr, info = env.step(drv.act(o))
    if te or tr: break
d = drv.diagnostics()
assert "pedestrian" in d["caps"] and "limit" in d["caps"]
print(f"speeding component negative in a 30 zone; pedestrian hit -> -{env.cfg.pedestrian_penalty:.0f} as its own component; baseline drives {96}-D : OK")

print()
print("=" * 62)
print("ALL STREET CHECKS PASSED")
print("=" * 62)
