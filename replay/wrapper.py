"""RecordingWrapper: zero-boilerplate replay logging, applied by the caller.

Not a `gymnasium.Wrapper` subclass -- gymnasium is an optional dependency in
this repo, and this needs to work whether or not it's installed. Plain
composition instead: forward everything to the wrapped env unchanged, and log
each step. `CarNavEnv` never imports this, never knows it exists -- wrap it
from the outside, in whatever codebase is doing the training:

    import carnav
    from replay import ReplayBuffer, RecordingWrapper

    env = RecordingWrapper(carnav.make(), ReplayBuffer(capacity=200_000))
    obs, info = env.reset()
    ...                                    # buffer fills itself as you step
    batch = env.buffer.sample(256)

The explicit `buffer.add(...)` call is always available too, for a loop that
wants to log by hand instead (e.g. to also stash something `env.step()`
doesn't return, like an agent's own recurrent state).
"""


class RecordingWrapper:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.add(self._last_obs, action, reward, obs, terminated, truncated, info)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        # Passthrough for anything not defined above: .cfg, .action_space,
        # .observation_space, .car, .city, ... Only reached when the wrapper
        # itself has no such attribute, so reset/step above always win.
        return getattr(self.env, name)
