# -*- coding: utf-8 -*-
"""
McCormick & Huguenard (1992) thalamocortical relay neuron channels.

Equations taken directly from the Appendix of:
  McCormick DA, Huguenard JR (1992) A model of the electrophysiological
  properties of thalamocortical relay neurons. J Neurophysiol 68:1384-1400.

All voltage variables in the derivative equations are absolute membrane
potential in mV (no resting-potential offset), matching the paper's
convention.  Rates returned from derivatives() are in SI (1/s), matching
the convention used by every other channel in this codebase.
"""

from collections import OrderedDict
import numpy as np

from neurodemo.sim_core import Channel
import neurodemo.units as NU


def _vtrap(x, y):
    """Compute x / (1 - exp(-x/y)), the standard HH rate form with singularity at x=0.
    The removable singularity limit is y (L'Hôpital).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        r = x / (1.0 - np.exp(-x / y))
    return float(np.where(np.abs(x) < 1e-6 * max(abs(y), 1e-12), y, r))


class MHNa(Channel):
    """Fast transient Na⁺ channel (McCormick & Huguenard 1992, Appendix).

    Kinetics are those used for thalamocortical relay neurons; the rate
    equations use absolute membrane potential and are calibrated at 36 °C.
    gating: m³h
    """

    type = "INa"
    max_op = 0.2

    def __init__(self, gbar=90 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0.02), ("h", 0.99)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.ena

    def set_erev(self, erev):
        self.section.ena = erev

    def open_probability(self, state):
        return state[self, "m"] ** 3 * state[self, "h"]

    def derivatives(self, state):
        q10 = 3.0 ** ((self.sim.temp - 36.0) / 10.0)
        vm = float(state[self.section, "V"]) * 1000.0   # SI → mV, absolute
        m = state[self, "m"]
        h = state[self, "h"]

        # Singularity-safe rates: L'Hôpital limit at vm=-54 gives am=1.28,
        # at vm=-27 gives bm=1.40 (limit = 0.32×4 and 0.28×5 respectively).
        am = 0.32 * _vtrap(vm + 54.0, 4.0)
        bm = 0.28 * _vtrap(-(vm + 27.0), 5.0)   # β uses -(vm+27) form
        dm = q10 * (am * (1.0 - m) - bm * m)

        ah = 0.128 * np.exp(-(vm + 50.0) / 18.0)
        bh = 4.0 / (1.0 + np.exp(-(vm + 27.0) / 5.0))
        dh = q10 * (ah * (1.0 - h) - bh * h)

        return [float(dm) * 1e3, float(dh) * 1e3]   # ms⁻¹ → s⁻¹


class MHK(Channel):
    """Delayed-rectifier K⁺ channel (McCormick & Huguenard 1992, Appendix).

    gating: n⁴
    """

    type = "IK"
    max_op = 0.55

    def __init__(self, gbar=10 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("n", 0.08)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.ek

    def set_erev(self, erev):
        self.section.ek = erev

    def open_probability(self, state):
        return state[self, "n"] ** 4

    def derivatives(self, state):
        q10 = 3.0 ** ((self.sim.temp - 36.0) / 10.0)
        vm = float(state[self.section, "V"]) * 1000.0
        n = state[self, "n"]

        # Singularity-safe: L'Hôpital limit at vm=-52 gives an=0.16 (=0.032×5).
        an = 0.032 * _vtrap(vm + 52.0, 5.0)
        bn = 0.5 * np.exp(-(vm + 57.0) / 40.0)
        dn = q10 * (an * (1.0 - n) - bn * n)

        return [float(dn) * 1e3]


class MHCaT(Channel):
    """Low-threshold T-type Ca²⁺ channel (McCormick & Huguenard 1992, Appendix).

    Uses the exact piecewise τ_h formulation from the Appendix, distinct from
    the smoothed version in channels/extra.py CaT.
    gating: m²h
    q10_m = 5^((T-24)/10),  q10_h = 3^((T-24)/10)  (calibrated at 24 °C)
    """

    type = "ICaT"
    max_op = 1.0

    def __init__(self, gbar=2.0 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("m", 0.0), ("h", 0.0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.eca

    def set_erev(self, erev):
        self.section.eca = erev

    def open_probability(self, state):
        return state[self, "m"] ** 2 * state[self, "h"]

    def derivatives(self, state):
        q10m = 5.0 ** ((self.sim.temp - 24.0) / 10.0)
        q10h = 3.0 ** ((self.sim.temp - 24.0) / 10.0)
        vm = float(state[self.section, "V"]) * 1000.0
        m = state[self, "m"]
        h = state[self, "h"]

        minf = 1.0 / (1.0 + np.exp(-(vm + 57.0) / 6.2))
        hinf = 1.0 / (1.0 + np.exp((vm + 81.0) / 4.0))

        taum = (0.612 + 1.0 / (np.exp(-(vm + 132.0) / 16.7)
                                + np.exp((vm + 16.8) / 18.2))) / q10m

        # Piecewise τ_h from Appendix Eq. (11); use if/else on scalar vm to
        # avoid 0-D array artefacts from np.where with scalar condition.
        if vm < -80.0:
            tauh = np.exp((vm + 467.0) / 66.6) / q10h
        else:
            tauh = (28.0 + np.exp(-(vm + 22.0) / 10.5)) / q10h

        dm = (minf - m) / taum
        dh = (hinf - h) / tauh
        return [float(dm) * 1e3, float(dh) * 1e3]


class MHIh(Channel):
    """Hyperpolarization-activated cation current Ih (McCormick & Huguenard 1992, Appendix).

    Single-gate model (one h variable), in contrast to the two-gate (f, s)
    Destexhe 1993 Ih in channels/extra.py.
    erev → section.eh  (default -43 mV, mixed Na⁺/K⁺)
    Rates calibrated at 35 °C; q10 = 3.
    """

    type = "IH"
    max_op = 0.3

    def __init__(self, gbar=0.05 * NU.mS / NU.cm ** 2, **kwds):
        init_state = OrderedDict([("h", 0.0)])
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)

    @property
    def erev(self):
        return self.section.eh

    def set_erev(self, erev):
        self.section.eh = erev

    def open_probability(self, state):
        return state[self, "h"]

    def derivatives(self, state):
        q10 = 3.0 ** ((self.sim.temp - 35.0) / 10.0)
        vm = float(state[self.section, "V"]) * 1000.0
        h = state[self, "h"]

        hinf = 1.0 / (1.0 + np.exp((vm + 75.0) / 5.5))
        # α and β in units of ms⁻¹ (Appendix); calibrated at 35 °C
        alpha = np.exp(-0.086 * vm - 14.59)
        beta  = np.exp(0.0701 * vm - 1.87)
        tauh  = 1.0 / ((alpha + beta) * q10)   # ms, temperature-corrected

        dh = (hinf - h) / tauh
        return [float(dh) * 1e3]
