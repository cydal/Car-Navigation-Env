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

One `diagnostics()` key is a documented convention rather than a free-for-all
field, because the live viewer (`serve/`, `web/`) renders it specially:
`imagined_trajectories`, for agents that plan by imagining -- a world model
like DreamerV3 rolling out its latent dynamics is the motivating case, but
anything that can produce candidate future poses qualifies. Shape:

    [{"x": [x1, x2, ...], "y": [y1, y2, ...]}, ...]   # one dict per rollout

Absolute world-frame metres, in step order starting from the *next* predicted
step (not the current pose) -- the same frame `info["x"]`/`info["y"]` already
hand an agent every step, so anchoring an imagined rollout to the current pose
needs no extra bookkeeping. Everything else about it is deliberately
unconstrained: any number of rollouts (a single mean trajectory, or several
stochastic samples to show spread), any horizon length, and it is fine to
change length between calls. Omit the key entirely (the `{}` default) if the
agent has nothing to imagine -- most agents never will, and the viewer treats
its absence as "nothing to draw", not an error. See
INTEGRATION.md#visualising-a-world-models-imagination for the full contract
and a worked example.
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
