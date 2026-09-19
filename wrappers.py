"""RewardOverrideWrapper: let a training codebase compute its own reward.

Same discipline as `replay.wrapper.RecordingWrapper` and `agents/`: `env/` never
knows this exists. Not a `gymnasium.Wrapper` subclass, for the same reason
`RecordingWrapper` isn't one -- gymnasium is an optional dependency here, and
this needs to work whether or not it's installed. Plain composition instead:
forward `reset()` unchanged, and on every `step()` replace the reward with
whatever `reward_fn` returns before handing the tuple back.

    import carnav
    from wrappers import RewardOverrideWrapper

    def sparse_only(obs, action, reward, terminated, truncated, info):
        return info["reward_components"]["target_bonus"]   # ignore everything else

    env = RewardOverrideWrapper(carnav.make(), sparse_only)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)   # reward is sparse_only's

The env still computes its own reward internally first -- unchanged, still
exactly the formula INTEGRATION.md documents, and still what `info[
"reward_components"]` breaks down -- `reward_fn` just gets the last word on
what the *caller* sees. That makes additive shaping a one-liner instead of a
reimplementation:

    def plus_lane_bonus(obs, action, reward, terminated, truncated, info):
        return reward + my_lane_center_bonus(info)

One gotcha worth knowing, and it is the reason this is a wrapper instead of an
`EnvConfig` flag: `info["episode_reward"]` is the *env's own* running sum of
its own formula, computed with no idea this wrapper exists, so it will not
match a running sum of what `reward_fn` returns. Track your own episode return
if you override the reward and want to log it -- do not read `episode_reward`
expecting it to reflect the override.
"""


class RewardOverrideWrapper:
    def __init__(self, env, reward_fn):
        self.env = env
        self.reward_fn = reward_fn

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn(obs, action, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        # Passthrough for anything not defined above: .cfg, .action_space,
        # .observation_space, .car, .city, ... Only reached when the wrapper
        # itself has no such attribute, so reset/step above always win.
        return getattr(self.env, name)
