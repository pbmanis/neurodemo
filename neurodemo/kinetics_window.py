# -*- coding: utf-8 -*-
"""
Channel kinetics analysis window.

Plots steady-state (inf) and rate (alpha/beta) curves for HH-style ion channels
over -120 to +60 mV.  A single-channel view shows two stacked plots;  the
all-channels view shows one overlay plot.

Usage (from DemoWindow):
    win = KineticsWindow([channel], section, title="MHNa kinetics")
    # or
    win = KineticsWindow(active_channels, section, title="All channels")
Keep a reference to *win* to prevent it being garbage-collected.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

_V_MIN  = -120.0   # mV
_V_MAX  =   60.0   # mV
_V_STEP =    1.0   # mV

# Colorblind-friendly palette (10 distinct hues)
_PALETTE = [
    '#4c78a8',  # blue
    '#f58518',  # orange
    '#e45756',  # red
    '#72b7b2',  # teal
    '#54a24b',  # green
    '#eeca3b',  # yellow
    '#b279a2',  # purple
    '#ff9da6',  # pink
    '#9d755d',  # brown
    '#bab0ac',  # gray
]


# ---------------------------------------------------------------------------
# Mock state
# ---------------------------------------------------------------------------

class _MockState:
    """Minimal State substitute for evaluating channel.derivatives() at fixed V."""

    def __init__(self, section, voltage_V, var_vals):
        self._section = section
        self._v = float(voltage_V)
        self._vars = var_vals          # dict: (channel_obj, varname) → float

    def __getitem__(self, key):
        obj, var = key
        if obj is self._section:
            return self._v
        return self._vars.get((obj, var), 0.0)


# ---------------------------------------------------------------------------
# Kinetics computation
# ---------------------------------------------------------------------------

def compute_kinetics(channel, section):
    """
    Compute steady-state and rates for every gating variable over -120…+60 mV.

    For any linear dx/dt = f(V, x):
        alpha(V) = f(V, 0)       [s⁻¹, effective gate-opening rate]
        beta(V)  = -f(V, 1)      [s⁻¹, effective gate-closing rate]
        tau(V)   = 1/(alpha+beta) [s]
        inf(V)   = alpha/(alpha+beta)

    Returns
    -------
    v_mV : ndarray  — voltage axis in mV
    inf   : dict varname → ndarray (0..1)
    alpha : dict varname → ndarray (ms⁻¹)
    beta  : dict varname → ndarray (ms⁻¹)
    """
    v_mV  = np.arange(_V_MIN, _V_MAX + _V_STEP, _V_STEP)
    names = list(channel.difeq_state().keys())
    nv    = len(v_mV)

    inf_d   = {n: np.full(nv, np.nan) for n in names}
    alpha_d = {n: np.full(nv, np.nan) for n in names}
    beta_d  = {n: np.full(nv, np.nan) for n in names}

    base = {(channel, nm): 0.0 for nm in names}

    for iv, vm in enumerate(v_mV):
        vV = vm * 1e-3
        for i, var in enumerate(names):
            vd0 = base.copy(); vd0[(channel, var)] = 0.0
            vd1 = base.copy(); vd1[(channel, var)] = 1.0
            try:
                with np.errstate(all='ignore'):
                    a = float(channel.derivatives(_MockState(section, vV, vd0))[i])
                    b = float(channel.derivatives(_MockState(section, vV, vd1))[i])
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                denom = a - b          # = 1/tau_s (positive for a stable gate)
                if abs(denom) < 1e-20:
                    continue
                inf_val = a / denom
                if not (0.0 <= inf_val <= 1.0):
                    continue
                inf_d[var][iv]   = inf_val
                # convert s⁻¹ → ms⁻¹; clamp to ≥ 0 (numerical noise)
                alpha_d[var][iv] = max(a  * 1e-3, 0.0)
                beta_d[var][iv]  = max(-b * 1e-3, 0.0)
            except Exception:
                pass

    return v_mV, inf_d, alpha_d, beta_d


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _styled_plot(title, xlabel, ylabel, yrange=None):
    plt = pg.PlotWidget(title=title)
    plt.setLabel('bottom', xlabel)
    plt.setLabel('left', ylabel)
    plt.showGrid(x=True, y=True, alpha=0.3)
    if yrange is not None:
        plt.setYRange(*yrange)
    return plt


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class KineticsWindow(QtWidgets.QWidget):
    """
    Standalone popup that plots channel gating kinetics.

    Parameters
    ----------
    channels : list of Channel
        One channel → single-channel view (inf + rates).
        Multiple channels → overlay view (inf only).
    section  : Section
        The Section the channels belong to (needed for temperature and
        voltage offset conventions used inside derivatives()).
    title    : str
    """

    def __init__(self, channels, section, title='Channel Kinetics', parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.Window
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle(title)
        self.resize(900, 660)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        if len(channels) == 1:
            self._build_single(layout, channels[0], section)
        else:
            self._build_multi(layout, channels, section)

        self.show()

    # ------------------------------------------------------------------
    def _build_single(self, layout, channel, section):
        """Two stacked plots: steady-state (top) and alpha/beta rates (bottom)."""
        cls_name = type(channel).__name__
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        v_mV, inf_d, alpha_d, beta_d = compute_kinetics(channel, section)
        names = list(inf_d.keys())

        # ---- Plot 1: steady-state ----------------------------------------
        plt_inf = _styled_plot(
            f'<b>{cls_name}</b>  —  steady-state activation / inactivation',
            'Membrane potential (mV)', 'Steady-state', (0, 1),
        )
        plt_inf.addLegend(offset=(10, 10))
        splitter.addWidget(plt_inf)

        # ---- Plot 2: rates -----------------------------------------------
        plt_rate = _styled_plot(
            f'<b>{cls_name}</b>  —  rates',
            'Membrane potential (mV)', 'Rate (ms⁻¹)',
        )
        plt_rate.addLegend(offset=(10, 10))
        plt_rate.setXLink(plt_inf)
        splitter.addWidget(plt_rate)

        # ---- Curves ------------------------------------------------------
        dash = QtCore.Qt.PenStyle.DashLine
        for i, name in enumerate(names):
            color = _PALETTE[i % len(_PALETTE)]
            pen_s = pg.mkPen(color, width=2)
            pen_d = pg.mkPen(color, width=2, style=dash)

            v_ok = np.isfinite(inf_d[name])
            plt_inf.plot(v_mV[v_ok], inf_d[name][v_ok],
                         pen=pen_s, name=f'{name}∞')       # name∞

            v_a = np.isfinite(alpha_d[name])
            v_b = np.isfinite(beta_d[name])
            plt_rate.plot(v_mV[v_a], alpha_d[name][v_a],
                          pen=pen_s, name=f'α₍{name}₎')   # α₍name₎
            plt_rate.plot(v_mV[v_b], beta_d[name][v_b],
                          pen=pen_d, name=f'β₍{name}₎')   # β₍name₎

        splitter.setSizes([330, 330])

    # ------------------------------------------------------------------
    def _build_multi(self, layout, channels, section):
        """Two stacked plots (steady-state + rates) for all channels, one color per channel."""
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        plt_inf = _styled_plot(
            '<b>All active channels</b>  —  steady-state gating',
            'Membrane potential (mV)', 'Steady-state', (0, 1),
        )
        plt_inf.addLegend(offset=(10, 10))
        splitter.addWidget(plt_inf)

        plt_rate = _styled_plot(
            '<b>All active channels</b>  —  rates',
            'Membrane potential (mV)', 'Rate (ms⁻¹)',
        )
        plt_rate.addLegend(offset=(10, 10))
        plt_rate.setXLink(plt_inf)
        splitter.addWidget(plt_rate)

        line_styles = [
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenStyle.DashLine,
            QtCore.Qt.PenStyle.DotLine,
        ]
        dash = QtCore.Qt.PenStyle.DashLine

        for ci, channel in enumerate(channels):
            color    = _PALETTE[ci % len(_PALETTE)]
            cls_name = type(channel).__name__
            v_mV, inf_d, alpha_d, beta_d = compute_kinetics(channel, section)
            for i, name in enumerate(inf_d):
                pen_s = pg.mkPen(color, width=2,
                                 style=line_styles[i % len(line_styles)])
                pen_d = pg.mkPen(color, width=2, style=dash)

                v_ok = np.isfinite(inf_d[name])
                plt_inf.plot(v_mV[v_ok], inf_d[name][v_ok],
                             pen=pen_s, name=f'{cls_name}.{name}∞')

                v_a = np.isfinite(alpha_d[name])
                v_b = np.isfinite(beta_d[name])
                plt_rate.plot(v_mV[v_a], alpha_d[name][v_a],
                              pen=pen_s, name=f'α {cls_name}.{name}')
                plt_rate.plot(v_mV[v_b], beta_d[name][v_b],
                              pen=pen_d, name=f'β {cls_name}.{name}')

        splitter.setSizes([330, 330])
