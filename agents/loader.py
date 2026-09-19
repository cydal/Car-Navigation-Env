"""
Turns a plain string into an Agent, so both `main.py demo` and `main.py serve`
pick a decision-maker the same way, without either of them importing a
specific controller by name.

    load_agent("scripted", env)          # baselines.scripted.GapFollower
    load_agent("manual", env)            # keyboard control
    load_agent("random", env)            # smoke-test agent
    load_agent("configs/my_agent.json", env)   # anything else

A custom agent is described by a small JSON file rather than a bare
`module:factory` string on the command line -- one new dependency (a config
file, not a package) instead of zero, but it scales better once a factory
needs several constructor kwargs (checkpoint path, device, temperature, ...)
that would otherwise pile up as ad-hoc CLI flags:

    {
      "module": "my_world_model.agent",
      "factory": "PlannerAgent",
      "kwargs": {"checkpoint": "runs/ckpt_500.pt", "device": "cpu"}
    }

`factory` is looked up as an attribute of `module` and always called as
`factory(env, **kwargs)` -- a class (`__init__(self, env, **kwargs)`) or a
plain function both work, as long as `env` is the first positional argument,
even if the implementation ignores it. That one rule means this loader never
has to guess a constructor signature.
"""

import importlib
import json
from pathlib import Path

from .manual import ManualAgent
from .random_agent import RandomAgent


def _scripted(env, **kwargs):
    from baselines.scripted import GapFollower
    return GapFollower.for_env(env, **kwargs)


def _manual(env, **kwargs):
    return ManualAgent()


def _random(env, **kwargs):
    return RandomAgent(env)


BUILTIN = {"scripted": _scripted, "manual": _manual, "random": _random}


def load_agent(spec, env):
    """`spec` is a built-in name (see `BUILTIN`) or a path to a JSON config."""
    if spec in BUILTIN:
        return BUILTIN[spec](env)

    path = Path(spec)
    if not path.is_file():
        raise ValueError(
            f"unknown agent {spec!r}: not one of {sorted(BUILTIN)} and not an "
            f"existing config file")
    cfg = json.loads(path.read_text())
    try:
        module_name, factory_name = cfg["module"], cfg["factory"]
    except KeyError as e:
        raise ValueError(f"{path}: agent config needs a {e.args[0]!r} key") from e
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    return factory(env, **cfg.get("kwargs", {}))
