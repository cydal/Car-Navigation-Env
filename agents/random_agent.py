"""The minimal Agent: samples the action space and nothing else.

Useful as a smoke test (does this env/viewer/wrapper run end to end without a
real controller?) and as the shortest possible example of the interface --
copy this file's shape when wiring up a real policy.
"""

from .base import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        self.action_space = env.action_space

    def reset(self):
        pass

    def act(self, obs, info=None):
        return self.action_space.sample()
