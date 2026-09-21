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

`capture_frames=True` additionally renders and stores a frame every step,
*without* changing what the policy trains on -- `obs`/`next_obs` stay exactly
whatever `obs_type` already returns (typically the vector), and the frame
goes into the buffer's separate `frame` field instead. That is the point of
it: train on the cheap vector observation today, bank rendered frames in the
same run in case a later image-based experiment wants them, and pay for
rendering only on the runs that ask for it --

    env = carnav.make(render_mode="rgb_array")     # obs_type stays "vector"
    env = RecordingWrapper(env, ReplayBuffer(200_000), capture_frames=True)

`render_mode="rgb_array"` has to be set at construction (the same Gymnasium
convention `env.render()` already follows) or this raises immediately, rather
than silently storing `None` for every frame because the env had nothing to
render.
"""


class RecordingWrapper:
    def __init__(self, env, buffer, capture_frames=False):
        if capture_frames and getattr(env, "render_mode", None) is None:
            raise ValueError(
                "capture_frames=True needs the wrapped env built with "
                "render_mode='rgb_array' (e.g. carnav.make(render_mode='rgb_array')) "
                "-- otherwise env.render() has nothing to return every step")
        self.env = env
        self.buffer = buffer
        self.capture_frames = capture_frames
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render() if self.capture_frames else None
        self.buffer.add(self._last_obs, action, reward, obs, terminated, truncated, info, frame=frame)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        # Passthrough for anything not defined above: .cfg, .action_space,
        # .observation_space, .car, .city, ... Only reached when the wrapper
        # itself has no such attribute, so reset/step above always win.
        return getattr(self.env, name)
