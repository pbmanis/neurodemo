# -*- coding: utf-8 -*-
"""Leak channel — used by every model."""

from neurodemo.sim_core import Channel
import neurodemo.units as NU
import numpy as np


class Leak(Channel):
    type = "Ileak"

    def __init__(self, gbar=0.1 * NU.mS / NU.cm ** 2, **kwds):
        Channel.__init__(self, gbar=gbar, init_state={}, **kwds)

    @property
    def erev(self):
        return self.section.eleak

    def set_erev(self, erev):
        self.section.eleak = erev

    def open_probability(self, state):
        if state.state.ndim == 2:
            return np.ones(state.state.shape[1])
        return 1

    def derivatives(self, state):
        return []
