"""Integration surface for outside code: one import, one factory, no internals.

    import carnav
    env = carnav.make()                      # 73-D vector obs, traffic + lights
    env = carnav.make(traffic=False)         # 53-D, lights only
    env = carnav.make(obs_type="image")      # 64x64x3 uint8, needs panda3d

    import gymnasium as gym                  # if gymnasium is installed
    env = gym.make("CarNav-v0")

Why this file exists rather than telling callers to build `EnvConfig`,
`CityConfig` and `CarNavEnv` themselves: the settings that have to move together
live in three different config objects. `traffic=False` has to reach `EnvConfig`,
`road_width` has to reach `CityConfig`, and `max_speed` has to reach `CarParams`.
A training script that constructs those by hand gets one of them wrong eventually
-- and the failure is silent, because a mismatched config still produces a valid
episode. `make` routes every keyword to the object that owns it and raises on
anything it does not recognise.

See INTEGRATION.md for the full observation / reward / termination contract.
"""

import inspect

from env.nav_env import CarNavEnv, EnvConfig
from env.world import CityConfig
from env.car import CarParams

__all__ = ["make", "make_factory", "register", "CarNavEnv", "EnvConfig",
           "CityConfig", "CarParams"]

def _params(cls):
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


# `width` means map tiles on CityConfig and metres of bodywork on CarParams, so
# the flat namespace has to disambiguate: bare `width`/`height` are the map, and
# the car's body is `car_width`/`car_length`. Everything else is unique.
_CAR_ALIAS = {"car_width": "width", "car_length": "length"}
_ENV_KEYS = _params(EnvConfig)
_CITY_KEYS = _params(CityConfig)
_CAR_KEYS = (_params(CarParams) - set(_CAR_ALIAS.values())) | set(_CAR_ALIAS)
_OVERLAP = (_ENV_KEYS & _CITY_KEYS) | (_ENV_KEYS & _CAR_KEYS) | (_CITY_KEYS & _CAR_KEYS)
assert not _OVERLAP, f"ambiguous config keyword(s): {sorted(_OVERLAP)}"


def make(obs_type="vector", seed=None, renderer=None, image_size=64,
         render_mode=None, **kwargs):
    """Build a CarNavEnv, routing keywords to whichever config owns them.

    Any keyword accepted by `EnvConfig`, `CityConfig` or `CarParams` can be
    passed flat, with the car's body as `car_width` / `car_length`. Unknown
    keywords raise rather than being silently dropped, because a typo'd
    `traffic_light=False` that quietly leaves traffic lights on invalidates an
    experiment without failing.
    """
    unknown = set(kwargs) - _ENV_KEYS - _CITY_KEYS - _CAR_KEYS
    if unknown:
        raise TypeError(
            f"unknown setting(s) {sorted(unknown)}.\n"
            f"env:  {sorted(_ENV_KEYS)}\ncity: {sorted(_CITY_KEYS)}\n"
            f"car:  {sorted(_CAR_KEYS)}")
    car = {_CAR_ALIAS.get(k, k): v for k, v in kwargs.items() if k in _CAR_KEYS}
    return CarNavEnv(
        config=EnvConfig(**{k: v for k, v in kwargs.items() if k in _ENV_KEYS}),
        city_config=CityConfig(**{k: v for k, v in kwargs.items() if k in _CITY_KEYS}),
        car_params=CarParams(**car),
        obs_type=obs_type, seed=seed, renderer=renderer,
        image_size=image_size, render_mode=render_mode)


class _EnvFactory:
    """Picklable zero-argument env builder.

    A closure would be enough for SB3 and gymnasium, which both pickle worker
    payloads with cloudpickle, but not for plain `multiprocessing`. Being
    picklable by the standard library costs one class and removes a footgun.
    """

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def __call__(self):
        return make(**self.kwargs)

    def __repr__(self):
        return f"carnav.make_factory({', '.join(f'{k}={v!r}' for k, v in self.kwargs.items())})"


def make_factory(**kwargs):
    """A zero-argument callable that builds one env -- what vector envs want.

    `SubprocVecEnv`/`AsyncVectorEnv` pickle the factory and call it in the child
    process, so this carries plain keywords rather than a built env. Do not pass
    `renderer=`: a Panda3D renderer cannot cross a process boundary, and each
    worker has to build its own (see INTEGRATION.md).
    """
    if kwargs.get("renderer") is not None:
        raise ValueError("a renderer cannot be pickled into a worker process; "
                         "let each worker construct its own")
    return _EnvFactory(kwargs)


def register():
    """Register gymnasium ids. Idempotent; a no-op if gymnasium is not installed.

    No `max_episode_steps` is declared, deliberately. The env applies its own
    limit from `EnvConfig.max_episode_steps` and reports it as `truncated`, so
    letting gymnasium add a `TimeLimit` on top would give two limits -- and the
    outer one would fire without the inner one's bookkeeping.
    """
    try:
        from gymnasium.envs.registration import register as _reg, registry
    except ImportError:
        return False
    for env_id, obs_type in (("CarNav-v0", "vector"),
                             ("CarNavImage-v0", "image"),
                             ("CarNavBoth-v0", "both")):
        if env_id not in registry:
            _reg(id=env_id, entry_point="carnav:make",
                 kwargs={"obs_type": obs_type}, order_enforce=True,
                 disable_env_checker=False)
    return True


register()
