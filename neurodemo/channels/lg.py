# -*- coding: utf-8 -*-
"""Lewis-Gerstner cortical channels: LGNa, LGKfast, LGKslow."""

from collections import OrderedDict
import numpy as np

from neurodemo.sim_core import Channel
import neurodemo.units as NU


class LGNa(Channel):
    """Cortical sodium channel (Lewis & Gerstner 2002, p.124)."""

    type = "INa"

    def __init__(self, gbar=112.5 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0.019), ("h", 0.876)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.ena1

    def set_erev(self, erev):
        self.section.ena1 = erev

    def open_probability(self, state):
        return state[self, "m"] ** 3 * state[self, "h"]

    def derivatives(self, state):
        q10 = 3 ** ((self.sim.temp - 37.0) / 10.0)
        vm = state[self.section, "V"]
        m = state[self, "m"]
        h = state[self, "h"]
        vm *= 1000.0
        am = (-3020 + 40 * vm) / (1.0 - np.exp(-(vm - 75.5) / 13.5))
        bm = 1.2262 / np.exp(vm / 42.248)
        mtau = 1 / (am + bm)
        minf = am * mtau
        dm = q10 * (minf - m) / mtau
        ah = 0.0035 / np.exp(vm / 24.186)
        bh = 0.017 * (51.25 + vm) / (1.0 - np.exp(-(51.25 + vm) / 5.2))
        htau = 1.0 / (ah + bh)
        hinf = ah * htau
        dh = q10 * (hinf - h) / htau
        return [dm * 1e3, dh * 1e3]


class LGKfast(Channel):
    """Cortical fast potassium channel (Lewis & Gerstner 2002, p.124)."""

    type = "IKf"

    def __init__(self, gbar=225 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("n", 0.00024)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.ekf

    def set_erev(self, erev):
        self.section.ekf = erev

    def open_probability(self, state):
        return state[self, "n"] ** 2

    def derivatives(self, state):
        q10 = 3 ** ((self.sim.temp - 37.0) / 10.0)
        vm = state[self.section, "V"]
        n = state[self, "n"]
        vm *= 1000.0
        an = (vm - 95) / (1.0 - np.exp(-(vm - 95) / 11.8))
        bn = 0.025 / np.exp(vm / 22.22)
        ntau = 1 / (an + bn)
        ninf = an * ntau
        dn = q10 * (ninf - n) / ntau
        return [dn * 1e3]


class LGKslow(Channel):
    """Cortical slow potassium channel (Lewis & Gerstner 2002, p.124)."""

    type = "IKs"

    def __init__(self, gbar=0.225 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("n", 0.0005)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.eks

    def set_erev(self, erev):
        self.section.eks = erev

    def open_probability(self, state):
        return state[self, "n"] ** 4

    def derivatives(self, state):
        q10 = 3 ** ((self.sim.temp - 37.0) / 10.0)
        vm = state[self.section, "V"]
        n = state[self, "n"]
        vm *= 1000.0
        an = 0.014 * (vm + 44) / (1.0 - np.exp(-(44 + vm) / 2.3))
        bn = 0.0043 / np.exp((vm + 44) / 34)
        ntau = 1 / (an + bn)
        ninf = an * ntau
        dn = q10 * (ninf - n) / ntau
        return [dn * 1e3]
