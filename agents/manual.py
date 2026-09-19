"""Human-at-the-keyboard control, expressed as an ordinary Agent.

Manual driving isn't special-cased by whatever is stepping the env (a demo
window, the live viewer's Session) -- it's just another implementation of
`act(obs) -> action` that happens to ignore `obs` and read externally-set key
state instead. That keeps every caller's step loop identical regardless of
who or what is driving.
"""

import numpy as np

from .base import Agent


class ManualAgent(Agent):
    def __init__(self):
        self.keys = {"up": False, "down": False, "left": False, "right": False}

    def set_keys(self, **keys):
        """Update whichever keys are given; unmentioned keys keep their state."""
        for k, v in keys.items():
            if k in self.keys:
                self.keys[k] = bool(v)

    def reset(self):
        self.keys = {k: False for k in self.keys}

    def act(self, obs, info=None):
        # Signed throttle: up drives forward, down reverses -- no separate
        # brake key, same convention as the rest of this project's manual
        # controls (main.py's --keys demo).
        k = self.keys
        thr = (1.0 if k["up"] else 0.0) - (1.0 if k["down"] else 0.0)
        steer = (1.0 if k["right"] else 0.0) - (1.0 if k["left"] else 0.0)
        return np.array([thr, -1.0, steer], dtype=np.float32)

    def diagnostics(self):
        return {"intent": "manual"}
