# -*- coding: utf-8 -*-
"""
Backwards-compatibility shim.

All classes previously defined here have been split into:
  neurodemo.sim_core   — Sim, SimState, SimObject, Mechanism, Channel,
                         Section, PatchClamp, alpha-synapse helpers
  neurodemo.channels   — Leak, HHNa, HHK, IH, KA, CaL, CaT,
                         LGNa, LGKfast, LGKslow
"""

from neurodemo.sim_core import *      # noqa: F401, F403
from neurodemo.channels import *      # noqa: F401, F403
