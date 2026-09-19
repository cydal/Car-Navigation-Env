"""
Replay storage for whoever is training on this env -- kept entirely outside
`env/`, so nothing here is imported by, or known to, `CarNavEnv`. A training
codebase uses these directly against ordinary `env.reset()`/`env.step()`
calls; see `replay.wrapper.RecordingWrapper` for a zero-boilerplate way to
fill one automatically.

Two shapes, because they serve different training styles:

`ReplayBuffer` -- a ring buffer of single-step transitions, uniform random
sampling. What an off-policy method (DQN, SAC, ...) wants: independent steps,
oldest overwritten once full.

`EpisodeBuffer` -- whole episodes, sampling contiguous multi-step windows.
What a sequence model wants (an RSSM/Dreamer-style world model, or any
RNN/transformer trained with BPTT): `ReplayBuffer`'s i.i.d. sampling can't
give you a temporally contiguous run, because it never promises step t+1 in
the buffer actually followed step t in the environment.

Both store raw `(obs, action, reward, next_obs, terminated, truncated, info)`
tuples rather than preallocated typed arrays, deliberately: that makes no
assumption about `obs`/`action` shape or dtype, so a vector observation, an
image, or an `obs_type="both"` dict all work unchanged, and sampling only
allocates the batch it returns instead of the whole buffer's worth of
memory up front.
"""

import pickle
from collections import deque

import numpy as np

_FIELDS = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "info")


def _stack(items):
    """np.stack, except a list of dicts (obs_type='both') stacks per key."""
    if isinstance(items[0], dict):
        return {k: _stack([it[k] for it in items]) for k in items[0]}
    return np.stack(items)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self._buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        self._buf.append((obs, action, reward, next_obs, terminated, truncated, info))

    def __len__(self):
        return len(self._buf)

    def sample(self, batch_size, rng=None):
        """Uniform random sample, with replacement, as a dict of stacked arrays."""
        if not self._buf:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self._buf), size=batch_size)
        rows = [self._buf[i] for i in idx]
        obs, action, reward, next_obs, terminated, truncated, info = zip(*rows)
        return {
            "obs": _stack(obs), "action": np.stack(action),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_obs": _stack(next_obs),
            "terminated": np.asarray(terminated, dtype=bool),
            "truncated": np.asarray(truncated, dtype=bool),
            "info": info,
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(list(self._buf), f)

    @classmethod
    def load(cls, path, capacity=None):
        with open(path, "rb") as f:
            rows = pickle.load(f)
        buf = cls(capacity or len(rows))
        buf._buf.extend(rows)
        return buf


class EpisodeBuffer:
    def __init__(self, capacity_episodes):
        self.capacity = capacity_episodes
        self._episodes = deque(maxlen=capacity_episodes)
        self._current = []

    def add(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        self._current.append((obs, action, reward, next_obs, terminated, truncated, info))
        if terminated or truncated:
            self.end_episode()

    def end_episode(self):
        """Close out the in-progress episode even without a done step -- for
        cutting collection short mid-episode without losing what's been
        gathered so far."""
        if self._current:
            self._episodes.append(self._current)
            self._current = []

    def __len__(self):
        return len(self._episodes)

    def sample_sequences(self, batch_size, length, rng=None):
        """`batch_size` random contiguous `length`-step windows, each drawn
        from a single stored episode -- never spanning two. Episodes shorter
        than `length` are excluded, not padded, so every returned window is a
        real contiguous run.

        Returns a dict of arrays shaped (batch_size, length, ...), except
        `info`, which stays a plain (batch_size, length) list of whatever
        `env.step()` put there.
        """
        eligible = [e for e in self._episodes if len(e) >= length]
        if not eligible:
            raise ValueError(
                f"no stored episode has >= {length} steps "
                f"(longest is {max((len(e) for e in self._episodes), default=0)})")
        rng = rng or np.random.default_rng()
        windows = []
        for _ in range(batch_size):
            ep = eligible[int(rng.integers(0, len(eligible)))]
            start = int(rng.integers(0, len(ep) - length + 1))
            windows.append(ep[start:start + length])

        out = {}
        for k, name in enumerate(_FIELDS):
            per_window = [[step[k] for step in w] for w in windows]   # (batch, length)
            if name == "info":
                out[name] = per_window
            elif name in ("obs", "next_obs"):
                out[name] = _stack([_stack(steps) for steps in per_window])
            elif name in ("terminated", "truncated"):
                out[name] = np.asarray(per_window, dtype=bool)
            elif name == "reward":
                out[name] = np.asarray(per_window, dtype=np.float32)
            else:                                       # action: keep its native dtype
                out[name] = np.asarray(per_window)
        return out

    def save(self, path):
        self.end_episode()
        with open(path, "wb") as f:
            pickle.dump(list(self._episodes), f)

    @classmethod
    def load(cls, path, capacity_episodes=None):
        with open(path, "rb") as f:
            episodes = pickle.load(f)
        buf = cls(capacity_episodes or len(episodes))
        buf._episodes.extend(episodes)
        return buf
