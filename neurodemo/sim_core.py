# -*- coding: utf-8 -*-
"""
Core simulation classes: Sim, SimState, SimObject, Mechanism, Channel,
Section, PatchClamp, and the alpha-synapse helper.

Channel subclasses live in neurodemo/channels/.
"""

from collections import OrderedDict
import numpy as np
import scipy.integrate
import neurodemo.units as NU
import warnings


class Sim(object):
    """Simulator for a collection of objects that derive from SimObject."""

    def __init__(self, objects=None, temp=37.0, dt=10., integrator: str = 'solve_ivp'):
        if objects is None:
            objects = []
        self._objects = objects
        self._all_objs = None
        self._time = 0.0
        self.temp = temp
        self.dt = dt
        self.integrator = integrator

    def set_integrator(self, integrator: str):
        if integrator in ["odeint", "solve_ivp"]:
            self.integrator = integrator

    def change_dt(self, newdt: float = 100.e-6):
        if newdt < 5e-6:
            newdt = 5e-6
        elif newdt > 1001e-6:
            newdt = 1000e-6
        self.dt = newdt * NU.s

    def add(self, obj):
        assert obj._sim is None
        obj._sim = self
        self._objects.append(obj)
        return obj

    def all_objects(self):
        """Ordered dictionary of all objects to be simulated, keyed by name."""
        if self._all_objs is None:
            objs = OrderedDict()
            for o in self._objects:
                if not o.enabled:
                    continue
                for k, v in o.all_objects().items():
                    if k in objs:
                        raise NameError(
                            'Multiple objects with same name "%s": %s, %s'
                            % (k, objs[k], v)
                        )
                    objs[k] = v
            self._all_objs = objs
        return self._all_objs

    @property
    def time(self):
        return self._time

    def run(self, blocksize: int = 1000, **kwds):
        """Run the simulation for *blocksize* time steps."""
        self._all_objs = None
        all_objs = self.all_objects().values()

        if len(all_objs) == 0:
            raise RuntimeError("No objects added to simulation.")

        init_state = []
        difeq_vars = []
        dep_vars = {}
        for o in all_objs:
            pfx = o.name + "."
            for k, v in o.difeq_state().items():
                difeq_vars.append(pfx + k)
                init_state.append(v)
            for k, v in o.dep_state_vars.items():
                dep_vars[pfx + k] = v
        self._simstate = SimState(difeq_vars, dep_vars)
        t = np.arange(0, blocksize) * self.dt + self._time
        opts = {"rtol": 1e-6, "atol": 1e-8, "hmax": 5e-4, "full_output": 1}
        opts.update(kwds)

        if self.integrator == 'odeint':
            result, info = scipy.integrate.odeint(
                self.derivatives, init_state, t, tfirst=True, **opts
            )
            p = 0
            for o in all_objs:
                nvar = len(o.difeq_state())
                o.update_state(result[-1, p:p + nvar])
                p += nvar
            self._time = t[-1]
            return SimState(difeq_vars, dep_vars, result.T, integrator=self.integrator, t=t)

        elif self.integrator == 'solve_ivp':
            result = scipy.integrate.solve_ivp(
                self.derivatives,
                t_span=(t[0], t[-1]),
                t_eval=t,
                y0=init_state,
                method="LSODA",
                dense_output=False,
                rtol=opts['rtol'],
                atol=opts['atol'],
                max_step=opts['hmax'],
            )
            if not isinstance(result.y, np.ndarray) or result.y.ndim != 2 or result.y.shape[1] == 0:
                raise RuntimeError(
                    f"Integrator failed ({getattr(result, 'message', type(result.y).__name__)}). "
                    "Check for NaN in channel derivatives or invalid parameters."
                )
            p = 0
            for i, o in enumerate(all_objs):
                nvar = len(o.difeq_state())
                o.update_state(result.y[p:p + nvar, -1])
                p += nvar
            self._time = t[-1]
            return SimState(difeq_vars, dep_vars, result.y, integrator=self.integrator, t=t)

    def derivatives(self, t, state):
        objs = self.all_objects().values()
        self._simstate.state = state
        self._simstate.extra["t"] = t
        d = []
        for o in objs:
            d.extend(o.derivatives(self._simstate))
        return d

    def state(self):
        """Return dictionary of all dependent and independent state variables."""
        state = {}
        for o in self.all_objects():
            for k, v in o.state(self._simstate).items():
                state[k] = v
        return state


class SimState(object):
    """Carries all diff-eq and dependent state variables during and after a run.

    Parameters
    ----------
    difeq_vars : list
        Names of all diff-eq state variables.
    dep_vars : dict
        name → function pairs for dependent variables.
    difeq_state : array-like, optional
        Initial (or full recorded) values for the diff-eq variables.
    extra :
        Additional name=value pairs accessible from this object (e.g. ``t``).
    """

    def __init__(self, difeq_vars, dep_vars=None, difeq_state=None,
                 integrator='odeint', **extra):
        self.difeq_vars = difeq_vars
        self.indexes = {k: i for i, k in enumerate(difeq_vars)}
        self.dep_vars = dep_vars
        self.state = difeq_state
        self.extra = extra
        self.integrator = integrator

    def set_state(self, difeq_state):
        self.state = difeq_state

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.get_slice(key)
        if isinstance(key, tuple):
            key = key[0].name + "." + key[1]
        try:
            return self.state[self.indexes[key]]
        except KeyError:
            if key in self.dep_vars:
                return self.dep_vars[key](self)
            else:
                return self.extra[key]

    def keys(self):
        return (list(self.indexes.keys())
                + list(self.dep_vars.keys())
                + list(self.extra.keys()))

    def __contains__(self, key):
        if isinstance(key, tuple):
            key = key[0].name + "." + key[1]
        return key in self.indexes or key in self.dep_vars or key in self.extra

    def __str__(self):
        rep = "SimState:\n"
        for i, k in enumerate(self.difeq_vars):
            rep += f"  {k} = {self.state[i][-1]}\n"
        return rep

    def get_final_state(self):
        """Return a dict of all state variables at the last time point."""
        return self.get_state_at_index(-1)

    def get_state_at_time(self, t):
        index = np.searchsorted(self['t'], t)
        return self.get_state_at_index(index)

    def get_state_at_index(self, index):
        state = {}
        s = self.copy()
        clip = not np.isscalar(self["t"])
        if clip:
            s.set_state(self.state[:, index])
        for k in self.difeq_vars:
            state[k] = s[k]
        for k in self.dep_vars:
            state[k] = s[k]
        for k, v in self.extra.items():
            state[k] = v[index] if clip else v
        return state

    def get_slice(self, sl):
        kwds = {'difeq_state': self.state[:, sl]}
        for k, v in self.extra.items():
            kwds[k] = v[sl]
        return self.copy(**kwds)

    def copy(self, **kwds):
        default_kwds = {
            'difeq_vars': self.difeq_vars,
            'dep_vars': self.dep_vars,
            'difeq_state': self.state,
            'integrator': self.integrator,
        }
        default_kwds.update(self.extra)
        default_kwds.update(kwds)
        return SimState(**default_kwds)


class SimObject(object):
    """Base class for objects that provide diff-eq state variables."""

    instance_count = 0

    def __init__(self, init_state, name=None):
        self._sim = None
        if name is None:
            i = self.instance_count
            type(self).instance_count = i + 1
            name = self.type if i == 0 else self.type + "%d" % i
        self._name = name
        self.enabled = True
        self._init_state = init_state.copy()
        self._current_state = init_state.copy()
        self._sub_objs = []
        self.records = []
        self._rec_dtype = [(sv, float) for sv in init_state.keys()]
        self.dep_state_vars = {}

    @property
    def name(self):
        return self._name

    def all_objects(self):
        """Ordered dict of all enabled SimObjects in this branch of the hierarchy."""
        objs = OrderedDict()
        objs[self.name] = self
        for o in self._sub_objs:
            if not o.enabled:
                continue
            objs.update(o.all_objects())
        return objs

    def difeq_state(self):
        """Ordered dict of variables needed to solve the diff-eq."""
        return self._current_state

    def update_state(self, result):
        """Update diff-eq state variables with their last simulated values."""
        for i, k in enumerate(self._current_state.keys()):
            self._current_state[k] = result[i]

    def derivatives(self, state):
        """Return derivatives of all state variables. Must be reimplemented."""
        raise NotImplementedError()

    @property
    def sim(self):
        return self._sim


class Mechanism(SimObject):
    """Base class for objects that interact with a section's membrane."""

    def __init__(self, init_state, section=None, **kwds):
        SimObject.__init__(self, init_state, **kwds)
        self._name = kwds.pop("name", None)   # override auto-name
        self._section = section
        self.dep_state_vars["I"] = self.current

    def current(self, state):
        """Return membrane current. Must be implemented in subclasses."""
        raise NotImplementedError()

    @property
    def name(self):
        if self._name is None:
            names = []
            if self._section is None:
                return None
            for o in self._section.mechanisms:
                if isinstance(o, Mechanism) and o._name is None:
                    continue
                names.append(o.name)
            pfx = self._section.name + "."
            name = pfx + self.type
            i = 1
            while name in names:
                name = pfx + self.type + str(i)
                i += 1
            self._name = name
        return self._name

    @property
    def section(self):
        return self._section

    @property
    def sim(self):
        return self.section.sim


class Channel(Mechanism):
    """Base class for simple ion channels."""

    rates = None
    max_op = 1.0

    @classmethod
    def compute_rates(cls):
        return

    def __init__(self, gmax=None, gbar=None, init_state=None, **kwds):
        Mechanism.__init__(self, init_state, **kwds)
        self._gmax = gmax
        self._gbar = gbar
        if self.rates is None:
            type(self).compute_rates()
        self.dep_state_vars["G"] = self.conductance
        self.dep_state_vars["OP"] = self.open_probability

    @property
    def gmax(self):
        if self._gmax is not None:
            return self._gmax
        return self._gbar * self.section.area

    @gmax.setter
    def gmax(self, v):
        self._gmax = v
        self._gbar = None

    @property
    def gbar(self):
        if self._gbar is not None:
            return self._gbar
        return self._gmax / self.section.area

    @gbar.setter
    def gbar(self, v):
        self._gbar = v
        self._gmax = None

    def conductance(self, state):
        return self.gmax * self.open_probability(state)

    def current(self, state):
        vm = state[self.section, "V"]
        return -self.conductance(state) * (vm - self.erev)

    @staticmethod
    def interpolate_rates(rates, val, minval, step):
        """Interpolate kinetic rates from a precomputed table."""
        i = (val - minval) / step
        i1 = int(i)
        i2 = i1 + 1
        s = i2 - i
        if i1 < 0:
            return rates[0]
        elif i2 >= len(rates):
            return rates[-1]
        return rates[i1] * s + rates[i2] * (1 - s)


class Section(SimObject):
    type = "section"

    def __init__(self, radius=None, cap=10e-12 * NU.F, vm=-65 * NU.mV, **kwds):
        self.cap_bar = 1 * NU.uF / NU.cm ** 2
        if radius is None:
            self.cap = cap
            self.area = cap / self.cap_bar
        else:
            self.cap = self.area * self.cap_bar
            self.area = 4 * 3.1415926 * radius ** 2
        self.set_default_erev()
        init_state = OrderedDict([("V", vm)])
        SimObject.__init__(self, init_state, **kwds)
        self.dep_state_vars["I"] = self.current
        self.mechanisms = []

    def set_default_erev(self):
        self.ek    = -77  * NU.mV
        self.ena   =  50  * NU.mV
        self.ena1  =  74  * NU.mV
        self.eca   =  140 * NU.mV
        self.ekf   = -90  * NU.mV
        self.eks   = -90  * NU.mV
        self.ecl   = -70  * NU.mV
        self.eh    = -43  * NU.mV
        self.eleak = -55  * NU.mV

    def add(self, mech):
        assert mech._section is None
        mech._section = self
        self.mechanisms.append(mech)
        self._sub_objs.append(mech)
        return mech

    def derivatives(self, state):
        Im = 0
        for mech in self.mechanisms:
            if not mech.enabled:
                continue
            Im += mech.current(state)
        return [Im / self.cap]

    def current(self, state):
        """Current flowing across the membrane capacitance."""
        return -self.cap * self.derivatives(state)[0]


class PatchClamp(Mechanism):
    type = "PatchClamp"

    def __init__(self, mode="ic", ra=0.1 * NU.MOhm, cpip=0.5e-12 * NU.F, **kwds):
        self.ra = ra
        self.cpip = cpip
        self._mode = mode
        self.cmd_queue = []
        self.cmd = []
        self.last_time = 0.0
        self.holding = {"ic": 0.0 * NU.pA, "vc": -65 * NU.mV}
        self.gain = 50e-6
        init_state = OrderedDict([("V", -65 * NU.mV)])
        Mechanism.__init__(self, init_state, **kwds)
        self.dep_state_vars['cmd'] = self.get_cmd_from_state

    def queue_command(self, cmd, dt, start=None):
        """Queue a command array. Returns the start time."""
        assert cmd.ndim == 1 and cmd.shape[0] > 0
        if len(self.cmd_queue) == 0:
            next_start = self.last_time + dt
        else:
            last_start, last_dt, last_cmd = self.cmd_queue[-1]
            next_start = last_start + len(last_cmd) * last_dt
        if start is None:
            start = next_start
        elif start < next_start:
            raise ValueError(
                "Cannot start next command before %f; asked for %f."
                % (next_start, start)
            )
        self.cmd_queue.append((start, dt, cmd))
        return start

    def queue_commands(self, cmds, dt):
        return [self.queue_command(c, dt) for c in cmds]

    @property
    def mode(self):
        return self._mode

    def clear_queue(self):
        self.cmd_queue = []

    def set_mode(self, mode):
        self._mode = mode
        self.clear_queue()

    def set_holding(self, mode, val):
        if mode not in self.holding:
            raise ValueError("Mode must be 'ic' or 'vc'")
        self.holding[mode] = val

    def current(self, state):
        vm = state[self.section, "V"]
        ve = state[self, "V"]
        return (ve - vm) / self.ra

    def derivatives(self, state):
        t = state["t"]
        self.last_time = t
        cmd = self.get_cmd(t)
        if self.mode == "vc":
            ve = state[self, "V"]
            cmd = (cmd - ve) * self.gain
        dve = (cmd - self.current(state)) / self.cpip
        return [dve]

    def get_cmd_from_state(self, state):
        if isinstance(state['t'], np.ndarray):
            return [self.get_cmd(t) for t in state['t']]
        return self.get_cmd(state['t'])

    def get_cmd(self, t: float):
        """Return the interpolated command value at time *t*."""
        hold = self.holding[self.mode]
        while len(self.cmd_queue) > 0:
            (start, dt, data) = self.cmd_queue[0]
            i1 = int(np.floor((t - start) / dt))
            if i1 < -1:
                return hold
            elif i1 == -1:
                v1, vt1 = hold, start - dt
                v2, vt2 = data[0], start
                break
            elif i1 >= len(data):
                self.cmd_queue.pop(0)
                continue
            else:
                v1 = data[i1]
                vt1 = start + i1 * dt
                if i1 + 1 < len(data):
                    v2, vt2 = data[i1 + 1], vt1 + dt
                    break
                else:
                    if (len(self.cmd_queue) > 1
                            and vt1 + dt >= self.cmd_queue[1][0]):
                        v2 = self.cmd_queue[1][2][0]
                        vt2 = self.cmd_queue[1][0]
                    else:
                        v2, vt2 = hold, vt1 + dt
                    break
        if len(self.cmd_queue) == 0:
            return hold
        s = (t - vt1) / (vt2 - vt1)
        return v1 * (1 - s) + v2 * s


# ── Alpha-synapse constants (kept here for backward compat) ────────────────────

Area      = 1e-12 * NU.cm ** 2
Alpha_t0  = 500.0 * NU.ms
Alpha_tau = 2.0   * NU.ms
gAlpha    = 1e-3  * Area / NU.cm ** 2
EAlpha    = -7    * NU.mV


def IAlpha(Vm, t):
    if t < Alpha_t0:
        return 0.0
    tn = t - Alpha_t0
    if tn > 10.0 * Alpha_tau:
        return 0.0
    return (gAlpha * (Vm - EAlpha)
            * (tn / Alpha_tau)
            * np.exp(-(tn - Alpha_tau) / Alpha_tau))
