# -*- coding: utf-8 -*-
"""Hodgkin-Huxley channels: HHK and HHNa."""

from collections import OrderedDict
import numpy as np
import warnings

from neurodemo.sim_core import Channel
import neurodemo.units as NU


class HHK(Channel):
    """Hodgkin-Huxley K channel."""

    type = "IK"
    max_op = 0.55

    @classmethod
    def compute_rates(cls):
        cls.rates_vmin = -100
        cls.rates_vstep = 0.1
        vm = np.arange(cls.rates_vmin, cls.rates_vmin + 400, cls.rates_vstep)
        cls.rates = np.empty((len(vm), 2))
        cls.rates[:, 0] = (0.1 - 0.01 * vm) / (np.exp(1.0 - 0.1 * vm) - 1.0)
        cls.rates[:, 1] = 0.125 * np.exp(-vm / 80.0)

    def __init__(self, gbar=12 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("n", 0.3)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0
        self.lastn = init_state["n"]

    @property
    def erev(self):
        return self.section.ek

    def set_erev(self, erev):
        self.section.ek = erev

    def open_probability(self, state):
        return state[self, "n"] ** 4

    def derivatives(self, state):
        q10 = 3 ** ((self.sim.temp - 6.3) / 10.0)
        vm = state[self.section, "V"] - self.shift
        vm = vm + 65e-3
        vm *= 1000.0
        n = state[self, "n"]
        an = (0.1 - 0.01 * vm) / (np.exp(1.0 - 0.1 * vm) - 1.0)
        bn = 0.125 * np.exp(-vm / 80.0)
        dn = q10 * (an * (1.0 - n) - bn * n)
        return [dn * 1e3]


class HHNa(Channel):
    """Hodgkin-Huxley Na channel."""

    type = "INa"
    max_op = 0.2

    @classmethod
    def compute_rates(cls):
        cls.rates_vmin = -100
        cls.rates_vstep = 0.1
        vm = np.arange(cls.rates_vmin, cls.rates_vmin + 400, cls.rates_vstep)
        cls.rates = np.empty((len(vm), 4))
        cls.rates[:, 0] = (2.5 - 0.1 * vm) / (np.exp(2.5 - 0.1 * vm) - 1.0)
        cls.rates[:, 1] = 4.0 * np.exp(-vm / 18.0)
        cls.rates[:, 2] = 0.07 * np.exp(-vm / 20.0)
        cls.rates[:, 3] = 1.0 / (np.exp(3.0 - 0.1 * vm) + 1.0)

    def __init__(self, gbar=40 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0.05), ("h", 0.6)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0
        self.lastm = init_state["m"]
        self.lasth = init_state["h"]

    @property
    def erev(self):
        return self.section.ena

    def set_erev(self, erev):
        self.section.ena = erev

    def open_probability(self, state):
        return state[self, "m"] ** 3 * state[self, "h"]

    def derivatives(self, state):
        q10 = 3 ** ((self.sim.temp - 6.3) / 10.0)
        vm = state[self.section, "V"] - self.shift
        m = state[self, "m"]
        h = state[self, "h"]
        vm = vm + 65e-3
        vm *= 1000.0
        with warnings.catch_warnings(record=True):
            am = (2.5 - 0.1 * vm) / (np.exp(2.5 - 0.1 * vm) - 1.0)
            bm = 4.0 * np.exp(-vm / 18.0)
            dm = q10 * (am * (1.0 - m) - bm * m)
            ah = 0.07 * np.exp(-vm / 20.0)
            bh = 1.0 / (np.exp(3.0 - 0.1 * vm) + 1.0)
            dh = q10 * (ah * (1.0 - h) - bh * h)
        return [dm * 1e3, dh * 1e3]
