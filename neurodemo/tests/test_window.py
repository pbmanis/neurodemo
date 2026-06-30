# -*- coding: utf-8 -*-
"""
Standalone PyQt6 application for model regression testing.

Run via:
    python run_model_tests.py

Workflow:
  1. Select a model in the dropdown.
  2. Click "Run Tests" — IC and VC protocols run in the background.
  3. Click "Save Reference" — stores results as a timestamped .npz file.
  4. Later: click "Load & Compare" — runs fresh tests and compares against
     the loaded reference, reporting point-by-point statistics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from neurodemo.tests.sim_protocols import (
    MODEL_PARAMS,
    IC_AMPLITUDES,
    VC_STEP_VOLTAGES,
    compare_results,
    find_latest_reference,
    load_results,
    run_model_tests,
    save_results,
)

REFERENCE_DIR = Path(__file__).parent.parent.parent / 'model_results'

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _ramp_pen(i: int, n: int, alpha: int = 220) -> pg.mkPen:
    """Blue→red colour ramp for *n* traces; returns pen for index *i*."""
    t = i / max(n - 1, 1)
    r = int(255 * t)
    b = int(255 * (1 - t))
    return pg.mkPen(color=(r, 0, b, alpha), width=1.2)


# ── Background worker ──────────────────────────────────────────────────────────

class _TestWorker(QtCore.QThread):
    results_ready = QtCore.Signal(dict)
    error_occurred = QtCore.Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            results = run_model_tests(self.model_name)
            self.results_ready.emit(results)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ── Main window ────────────────────────────────────────────────────────────────

class ModelTestWindow(QtWidgets.QWidget):
    """Main test runner window."""

    def __init__(self):
        super().__init__()
        self._results: dict | None = None
        self._reference: dict | None = None
        self._ref_path: Path | None = None
        self._worker: _TestWorker | None = None

        self._build_ui()
        self.setWindowTitle('Neurodemo Model Tests')
        self.resize(1200, 700)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── toolbar ───────────────────────────────────────────────────────────
        toolbar = QtWidgets.QHBoxLayout()
        root.addLayout(toolbar)

        toolbar.addWidget(QtWidgets.QLabel('Model:'))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems(list(MODEL_PARAMS))
        toolbar.addWidget(self._model_combo)

        self._run_btn = QtWidgets.QPushButton('Run Tests')
        self._run_btn.clicked.connect(self._run_tests)
        toolbar.addWidget(self._run_btn)

        self._save_btn = QtWidgets.QPushButton('Save Reference')
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_reference)
        toolbar.addWidget(self._save_btn)

        self._load_btn = QtWidgets.QPushButton('Load & Compare…')
        self._load_btn.clicked.connect(self._load_and_compare)
        toolbar.addWidget(self._load_btn)

        self._ref_label = QtWidgets.QLabel('Reference: none')
        self._ref_label.setStyleSheet('color: gray;')
        toolbar.addWidget(self._ref_label)

        toolbar.addStretch()

        # ── plots ─────────────────────────────────────────────────────────────
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(plot_splitter, stretch=1)

        self._ic_plot = pg.PlotWidget(title='IC — Membrane Voltage')
        self._ic_plot.setLabel('bottom', 'Time', units='s')
        self._ic_plot.setLabel('left', 'Vm', units='V')
        self._ic_plot.showGrid(True, True, alpha=0.3)
        plot_splitter.addWidget(self._ic_plot)

        self._vc_plot = pg.PlotWidget(title='VC — Membrane Current')
        self._vc_plot.setLabel('bottom', 'Time', units='s')
        self._vc_plot.setLabel('left', 'Im', units='A')
        self._vc_plot.showGrid(True, True, alpha=0.3)
        plot_splitter.addWidget(self._vc_plot)

        plot_splitter.setSizes([600, 600])

        # ── status bar ────────────────────────────────────────────────────────
        status_frame = QtWidgets.QGroupBox('Status / Comparison')
        root.addWidget(status_frame)
        status_layout = QtWidgets.QGridLayout(status_frame)
        status_layout.setContentsMargins(6, 4, 6, 4)

        self._status_label = QtWidgets.QLabel('Idle.')
        status_layout.addWidget(self._status_label, 0, 0, 1, 4)

        # IC comparison row
        status_layout.addWidget(QtWidgets.QLabel('<b>IC:</b>'), 1, 0)
        self._ic_max_lbl  = QtWidgets.QLabel('Max ΔV: —')
        self._ic_rms_lbl  = QtWidgets.QLabel('RMS ΔV: —')
        self._ic_disc_lbl = QtWidgets.QLabel('Discrepant pts: —')
        status_layout.addWidget(self._ic_max_lbl,  1, 1)
        status_layout.addWidget(self._ic_rms_lbl,  1, 2)
        status_layout.addWidget(self._ic_disc_lbl, 1, 3)

        # VC comparison row
        status_layout.addWidget(QtWidgets.QLabel('<b>VC:</b>'), 2, 0)
        self._vc_max_lbl  = QtWidgets.QLabel('Max ΔI: —')
        self._vc_rms_lbl  = QtWidgets.QLabel('RMS ΔI: —')
        self._vc_disc_lbl = QtWidgets.QLabel('Discrepant pts: —')
        status_layout.addWidget(self._vc_max_lbl,  2, 1)
        status_layout.addWidget(self._vc_rms_lbl,  2, 2)
        status_layout.addWidget(self._vc_disc_lbl, 2, 3)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _run_tests(self):
        model = self._model_combo.currentText()
        self._status_label.setText(f'Running {model} …')
        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._ic_plot.clear()
        self._vc_plot.clear()
        self._clear_comparison_labels()

        self._worker = _TestWorker(model)
        self._worker.results_ready.connect(self._on_results_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_results_ready(self, results: dict):
        self._results = results
        self._plot_ic(results)
        self._plot_vc(results)
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        model = results['model']
        self._status_label.setText(f'{model} done.')
        if self._reference is not None:
            self._do_compare(results)

    def _on_error(self, msg: str):
        self._status_label.setText(f'Error: {msg}')
        self._run_btn.setEnabled(True)

    def _save_reference(self):
        if self._results is None:
            return
        path = save_results(self._results, outdir=REFERENCE_DIR)
        self._status_label.setText(f'Saved: {os.path.basename(path)}')

    def _load_and_compare(self):
        start_dir = str(REFERENCE_DIR) if REFERENCE_DIR.exists() else '.'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Reference File', start_dir, 'NumPy archives (*.npz)'
        )
        if not path:
            return
        self._reference = load_results(path)
        self._ref_path  = Path(path)
        self._ref_label.setText(f'Reference: {os.path.basename(path)}')
        self._ref_label.setStyleSheet('color: cyan;')
        if self._results is not None:
            self._do_compare(self._results)

    def _do_compare(self, results: dict):
        try:
            comp = compare_results(self._reference, results)
        except ValueError as exc:
            self._status_label.setText(f'Comparison error: {exc}')
            return

        self._ic_max_lbl.setText(
            f"Max ΔV: {comp['max_V_deviation']*1e6:.3f} µV"
        )
        self._ic_rms_lbl.setText(
            f"RMS ΔV: {comp['rms_V_error']*1e6:.3f} µV"
        )
        n = comp['discrepant_V_points']
        thr = comp['threshold_v'] * 1e6
        self._ic_disc_lbl.setText(f"Discrepant pts: {n}  (>{thr:.1f} µV)")
        self._ic_disc_lbl.setStyleSheet('color: red;' if n > 0 else 'color: green;')

        self._vc_max_lbl.setText(
            f"Max ΔI: {comp['max_I_deviation']*1e12:.3f} pA"
        )
        self._vc_rms_lbl.setText(
            f"RMS ΔI: {comp['rms_I_error']*1e12:.3f} pA"
        )
        n = comp['discrepant_I_points']
        thr = comp['threshold_i'] * 1e12
        self._vc_disc_lbl.setText(f"Discrepant pts: {n}  (>{thr:.1f} pA)")
        self._vc_disc_lbl.setStyleSheet('color: red;' if n > 0 else 'color: green;')

        # overlay difference traces (thin, semi-transparent white)
        t_ic = results['t_ic']
        for i, dv in enumerate(comp['V_diff']):
            self._ic_plot.plot(t_ic, dv, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.PenStyle.DashLine))

        t_vc = results['t_vc']
        for i, di in enumerate(comp['I_diff']):
            self._vc_plot.plot(t_vc, di, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.PenStyle.DashLine))

        self._status_label.setText('Comparison complete.')

    def _clear_comparison_labels(self):
        for lbl in (self._ic_max_lbl, self._ic_rms_lbl, self._ic_disc_lbl,
                    self._vc_max_lbl, self._vc_rms_lbl, self._vc_disc_lbl):
            lbl.setText(lbl.text().split(':')[0] + ': —')
            lbl.setStyleSheet('')

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _plot_ic(self, results: dict):
        self._ic_plot.clear()
        t   = results['t_ic']
        V   = results['V_ic']
        n   = len(IC_AMPLITUDES)
        for i, (amp, v) in enumerate(zip(IC_AMPLITUDES, V)):
            self._ic_plot.plot(
                t, v,
                pen=_ramp_pen(i, n),
                name=f'{amp*1e12:.0f} pA',
            )

    def _plot_vc(self, results: dict):
        self._vc_plot.clear()
        t   = results['t_vc']
        I   = results['I_vc']
        n   = len(VC_STEP_VOLTAGES)
        for i, (v_step, cur) in enumerate(zip(VC_STEP_VOLTAGES, I)):
            self._vc_plot.plot(
                t, cur,
                pen=_ramp_pen(i, n),
                name=f'{v_step*1e3:.0f} mV',
            )
