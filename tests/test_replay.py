"""Replay contract: ring-buffer behaviour, episode-window sampling, save/load
round-trips, RecordingWrapper's transparent passthrough, and optional frame
capture -- all kept outside `env/`, so none of this touches CarNavEnv itself.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import carnav
from replay import EpisodeBuffer, ReplayBuffer, RecordingWrapper

print("=" * 62)
print("1. REPLAYBUFFER: ring behaviour, sample shapes")
print("=" * 62)
buf = ReplayBuffer(capacity=5)
for i in range(8):
    buf.add(np.full(3, i, np.float32), np.zeros(2, np.float32), float(i),
            np.full(3, i + 1, np.float32), False, False)
assert len(buf) == 5, "capacity must cap the buffer, oldest entries dropped"
batch = buf.sample(4)
assert batch["obs"].shape == (4, 3) and batch["action"].shape == (4, 2)
assert batch["reward"].dtype == np.float32 and batch["terminated"].dtype == bool
assert batch["frame"] is None, "no frame was ever added -- must come back as None, not a stacked array"
assert len(batch["info"]) == 4
print(f"len(buf)={len(buf)} after 8 adds at capacity 5, sample() shapes OK, frame=None : OK")

print()
print("=" * 62)
print("2. REPLAYBUFFER: obs_type='both' (dict obs) stacks per key")
print("=" * 62)
for _ in range(6):
    obs = {"vector": np.zeros(4, np.float32), "image": np.zeros((8, 8, 3), np.uint8)}
    nxt = {"vector": np.ones(4, np.float32), "image": np.ones((8, 8, 3), np.uint8)}
    buf.add(obs, np.zeros(2, np.float32), 0.0, nxt, False, False)
b2 = buf.sample(3)
assert set(b2["obs"]) == {"vector", "image"}
assert b2["obs"]["vector"].shape == (3, 4) and b2["obs"]["image"].shape == (3, 8, 8, 3)
print("dict-obs sampling stacks 'vector' and 'image' independently : OK")

print()
print("=" * 62)
print("3. EPISODEBUFFER: sample_sequences never crosses an episode boundary")
print("=" * 62)
eb = EpisodeBuffer(capacity_episodes=10)
for ep_len in (3, 20, 20, 1):
    for t in range(ep_len):
        eb.add(np.full(2, t, np.float32), np.zeros(1, np.float32), 1.0,
                np.full(2, t + 1, np.float32), t == ep_len - 1, False)
assert len(eb) == 4
seq = eb.sample_sequences(batch_size=16, length=10)
assert seq["obs"].shape == (16, 10, 2)
# Every window's obs[t][0] must increase by exactly 1 each step -- the signature
# of a real contiguous run, which would break the instant a window spanned two
# episodes (that boundary resets the counter to 0).
diffs = np.diff(seq["obs"][:, :, 0], axis=1)
assert np.all(diffs == 1.0), "a sampled window is not contiguous within one episode"
try:
    eb.sample_sequences(batch_size=4, length=25)
    raise AssertionError("must reject a length longer than every stored episode")
except ValueError:
    pass
print(f"{len(eb)} episodes stored (lengths 3,20,20,1 -- one too short for length=25) : OK")

print()
print("=" * 62)
print("4. SAVE / LOAD round-trip")
print("=" * 62)
with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "rb.pkl")
    buf.save(p1)
    loaded = ReplayBuffer.load(p1)
    assert len(loaded) == len(buf)
    p2 = os.path.join(d, "eb.pkl")
    eb.save(p2)
    loaded_eb = EpisodeBuffer.load(p2)
    assert len(loaded_eb) == len(eb)
print("ReplayBuffer and EpisodeBuffer both round-trip through save()/load() : OK")

print()
print("=" * 62)
print("5. RECORDINGWRAPPER: transparent passthrough, fills the buffer")
print("=" * 62)
env = carnav.make(width=32, height=32, traffic=False, traffic_lights=False, seed=0)
rec_buf = ReplayBuffer(capacity=1000)
wrapped = RecordingWrapper(env, rec_buf)
obs, info = wrapped.reset(seed=3)
assert wrapped.cfg is env.cfg and wrapped.action_space is not None, \
    "unrelated attributes must pass through to the wrapped env"
for _ in range(20):
    obs, reward, te, tr, info = wrapped.step(wrapped.action_space.sample())
    if te or tr:
        obs, info = wrapped.reset()
assert len(rec_buf) == 20
b3 = rec_buf.sample(8)
assert b3["obs"].shape == (8, env.vector_dim) and b3["frame"] is None
print(f"buffer filled itself over 20 steps, obs_dim={env.vector_dim}, frame=None (never asked for) : OK")

print()
print("=" * 62)
print("6. FRAME CAPTURE: costs nothing unless asked for, real pixels when it is")
print("=" * 62)
# capture_frames=True must refuse an env with no render_mode -- silently storing
# None for every "frame" because nothing was configured to render would be a
# worse failure than an upfront error.
plain_env = carnav.make(width=32, height=32, traffic=False, seed=0)
try:
    RecordingWrapper(plain_env, ReplayBuffer(10), capture_frames=True)
    raise AssertionError("must reject capture_frames=True without render_mode='rgb_array'")
except ValueError:
    pass
print("capture_frames=True without render_mode='rgb_array' raises : OK")

img_env = carnav.make(width=32, height=32, traffic=False, traffic_lights=False,
                      seed=0, render_mode="rgb_array", image_size=32)
frame_buf = ReplayBuffer(capacity=200)
frame_wrapped = RecordingWrapper(img_env, frame_buf, capture_frames=True)
obs, info = frame_wrapped.reset(seed=4)
assert obs.shape == (img_env.vector_dim,), \
    "capture_frames must not change what the policy trains on -- obs stays the vector"
for _ in range(15):
    obs, reward, te, tr, info = frame_wrapped.step(frame_wrapped.action_space.sample())
    if te or tr:
        obs, info = frame_wrapped.reset()
b4 = frame_buf.sample(6)
assert b4["obs"].shape == (6, img_env.vector_dim), "obs is still the vector, not the frame"
assert b4["frame"] is not None and b4["frame"].shape == (6, 32, 32, 3), b4["frame"].shape
assert b4["frame"].dtype == np.uint8
img_env.close()
print(f"obs stayed vector ({img_env.vector_dim}-D) while frame batch came back {b4['frame'].shape} uint8 : OK")

print()
print("=" * 62)
print("ALL REPLAY CHECKS PASSED")
print("=" * 62)
