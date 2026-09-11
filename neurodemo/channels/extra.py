# -*- coding: utf-8 -*-
"""Extra channels: IH, KA, CaL, CaT."""

from collections import OrderedDict
import numpy as np

from neurodemo.sim_core import Channel
import neurodemo.units as NU


class IH(Channel):
    """Ih from Destexhe 1993."""

    type = "IH"
    max_op = 0.3

    def __init__(self, gbar=30 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("f", 0), ("s", 0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0.0
        self.lastf = init_state["f"]
        self.lasts = init_state["s"]

    @property
    def erev(self):
        return self.section.eh

    def set_erev(self, erev):
        self.section.eh = erev

    def open_probability(self, state):
        return state[self, "f"] * state[self, "s"]

    def check_state(self, state, gv, lastgv):
        n = state[self, gv]
        if np.isnan(n):
            n = lastgv
        if n > 1.0:
            n = 1.0
        elif n < 0:
            n = 0.0
        return n, n

    def derivatives(self, state):
        vm = state[self.section, "V"] - self.shift
        f = state[self, "f"]
        s = state[self, "s"]
        vm *= 1000.0
        Hinf = 1.0 / (1.0 + np.exp((vm + 68.9) / 6.5))
        tauF = np.exp((vm + 158.6) / 11.2) / (1.0 + np.exp((vm + 75.0) / 5.5))
        tauS = np.exp((vm + 183.6) / 15.24)
        df = (Hinf - f) / tauF
        ds = (Hinf - s) / tauS
        return [df * 1e3, ds * 1e3]


class KA(Channel):
    """KA from Rothman and Manis, 2003."""

    type = "IKA"
    max_op = 1.0

    def __init__(self, gbar=30 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("a", 0), ("b", 0), ("c", 0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0.0
        self.lasta = init_state["a"]
        self.lastb = init_state["b"]
        self.lastc = init_state["c"]

    @property
    def erev(self):
        return self.section.ek

    def set_erev(self, erev):
        self.section.ek = erev

    def open_probability(self, state):
        return state[self, "a"] ** 4 * state[self, "b"] * state[self, "c"]

    def derivatives(self, state):
        self.q10 = 3 ** ((self.sim.temp - 22.0) / 10.0)
        vm = state[self.section, "V"] - self.shift
        a = state[self, "a"]
        b = state[self, "b"]
        c = state[self, "c"]
        vm *= 1000.0
        Ainf = np.power(1.0 + np.exp(-(vm + 31.0) / 6.0), -0.25)
        Binf = np.power(1.0 + np.exp((vm + 66.0) / 7.0), -0.5)
        Cinf = Binf
        tauA = (7.0 * np.exp((vm + 60.0) / 14.0) + 29 * np.exp(-(vm + 60.0) / 24))
        tauA = 100.0 * (1.0 / tauA) + 0.1
        tauA = tauA / self.q10
        tauB = (14.0 * np.exp((vm + 60.0) / 27.0) + 29 * np.exp(-(vm + 60.0) / 24))
        tauB = 1000.0 * (1.0 / tauB) + 1.0
        tauB = tauB / self.q10
        tauC = 10.0 + 90.0 / (1.0 + np.exp((-66.0 - vm) / 17.0))
        tauC = tauC / self.q10
        da = (Ainf - a) / tauA
        db = (Binf - b) / tauB
        dc = (Cinf - c) / tauC
        return [da * 1e3, db * 1e3, dc * 1e3]


class CaL(Channel):
    """L-type calcium channel (Kampa & Stuart 2006, ModelDB:108458)."""

    type = "ICaL"
    max_op = 1.0

    def __init__(self, gbar=0.12 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0), ("h", 0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0
        self.lasta = init_state["m"]
        self.lastb = init_state["h"]

    @property
    def erev(self):
        return self.section.eca

    def set_erev(self, erev):
        self.section.eca = erev

    def open_probability(self, state):
        return state[self, "m"] ** 2 * state[self, "h"]

    def derivatives(self, state):
        vm = state[self.section, "V"] - self.shift
        m = state[self, "m"]
        h = state[self, "h"]
        vm *= 1000.0
        am = 0.055 * (-27.0 - vm) / (np.exp((-27.0 - vm) / 3.8) - 1)
        bm = 0.94 * np.exp((-75.0 - vm) / 17.0)
        mtau = 1.0 / (am + bm)
        minf = am * mtau
        ah = 0.000457 * np.exp((-13.0 - vm) / 50.0)
        bh = 0.0065 / (np.exp((-vm - 15.0) / 28.0) + 1.0)
        htau = 1.0 / (ah + bh)
        hinf = ah * htau
        dm = (minf - m) / mtau
        dh = (hinf - h) / htau
        return [dm * 1e3, dh * 1e3]


class CaT(Channel):
    """T-type calcium channel (Huguenard & McCormick, ModelDB:279)."""

    type = "ICaT"
    max_op = 1.0

    def __init__(self, gbar=0.008 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0), ("h", 0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 2.0
        self.lasta = init_state["m"]
        self.lastb = init_state["h"]

    @property
    def erev(self):
        return self.section.eca

    def set_erev(self, erev):
        self.section.eca = erev

    def open_probability(self, state):
        return state[self, "m"] ** 2 * state[self, "h"]

    def derivatives(self, state):
        self.q10m = 5 ** ((self.sim.temp - 24.0) / 10.0)
        self.q10h = 3 ** ((self.sim.temp - 24.0) / 10.0)
        vm = state[self.section, "V"] - self.shift / 1000.0
        m = state[self, "m"]
        h = state[self, "h"]
        vm *= 1000.0
        minf = 1.0 / (1.0 + np.exp(-(vm + 57.0) / 6.2))
        hinf = 1.0 / (1.0 + np.exp((vm + 81.0) / 4.0))
        mtau = 0.612 + 1.0 / (np.exp((vm + 16.8) / 18.2) + np.exp(-(vm + 132.0) / 16.7))
        mtau = mtau / self.q10m
        htau = 85.0 + 1.0 / (np.exp((vm + 46.0) / 4.0) + np.exp(-(vm + 405.0) / 50.0))
        dm = (minf - m) / mtau
        dh = (hinf - h) / htau
        return [dm * 1e3, dh * 1e3]
