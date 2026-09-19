"""
The decision-making interface: observation in, action out.

This is deliberately the smallest contract that covers scripted control, RL
policies, world-model planners, and a human at the keyboard alike:

    reset()               -- clear any per-episode state (e.g. a filter)
    act(obs, info=None)   -- the action for this observation
    diagnostics()         -- optional: a free-form dict for a HUD to show

`act` never receives a reward, on purpose. `CarNavEnv.step()` still returns
one every call -- it's cheap, and useful as a comparable score across
different agents -- but nothing here requires an agent to consume it. A
rule-based controller, a planner rolling out a learned world model, or an
imitation-learned policy at inference time all have no use for a reward
signal, and forcing one into this interface would make it look RL-specific
when it isn't. An RL *training* loop gets reward from `env.step()`'s own
return value directly, never through an Agent.

Subclassing `Agent` is optional -- duck typing is enough, exactly like
`GapFollower` (baselines/scripted.py) already satisfies this contract without
inheriting from anything. This class exists to document the shape once and
give `isinstance` checks somewhere to point at, not to gatekeep.
"""


class Agent:
    def reset(self):
        """Clear any per-episode state. Called once per env.reset()."""

    def act(self, obs, info=None):
        """Return the action for this observation. `info` is optional context
        (env.step()'s info dict, or None right after a reset) -- most agents
        can ignore it and act from `obs` alone."""
        raise NotImplementedError

    def diagnostics(self):
        """Optional free-form dict describing the current decision, for a
        HUD to display. Returning {} (the default) means "nothing to show" --
        callers must treat every key as optional, never assume one agent's
        diagnostics shape matches another's."""
        return {}
