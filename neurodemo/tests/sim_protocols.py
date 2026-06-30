# -*- coding: utf-8 -*-
"""
Headless simulation protocols for model regression testing.

All protocols use the same DT and initial conditions as the interactive
DemoWindow, so reference files saved here can track code changes.
"""
from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import numpy as np

import neurodemo.units as NU

# ── Protocol constants ─────────────────────────────────────────────────────────

DT = 20e-6 * NU.s          # simulation time step (seconds)

# Current clamp
IC_PRE_S    = 0.010         # baseline before pulse  (s)
IC_PULSE_S  = 0.100         # pulse duration          (s)
IC_POST_S   = 0.050         # tail after pulse        (s)
IC_AMPLITUDES = np.array(
    [-50, 0, 50, 100, 150, 200, 250, 300], dtype=float
) * NU.pA

# Voltage clamp
VC_HOLD_S   = 0.050         # settling at holding voltage (s)
VC_STEP_S   = 0.100         # step duration               (s)
VC_TAIL_S   = 0.050         # tail after step             (s)
VC_STEP_VOLTAGES = np.arange(-100, 65, 10, dtype=float) * NU.mV  # -100 to +60 mV

# Per-model parameters matching DemoWindow presets
MODEL_PARAMS: dict[str, dict] = {
    'Passive': dict(temp=6.3,  hold_vc=-65 * NU.mV),
    'HH AP':   dict(temp=6.3,  hold_vc=-65 * NU.mV),
    'LG AP':   dict(temp=37.0, hold_vc=-70 * NU.mV),
}


# ── Simulation factory ─────────────────────────────────────────────────────────

def make_model(name: str, mode: str = 'ic'):
    """Build a fresh simulation for the given model preset.

    Args:
        name: one of the keys in MODEL_PARAMS
        mode: 'ic' (current clamp) or 'vc' (voltage clamp)

    Returns:
        (sim, neuron, clamp)
    """
    from neurodemo.neuronsim import (
        Sim, Section, PatchClamp,
        Leak, HHNa, HHK,
        LGNa, LGKfast, LGKslow,
    )

    cfg = MODEL_PARAMS[name]
    sim = Sim(temp=cfg['temp'], dt=DT)
    neuron = Section(name='soma')
    sim.add(neuron)

    if name == 'Passive':
        leak = neuron.add(Leak())
        leak.gmax = 1 * NU.nS
        leak.set_erev(-55 * NU.mV)   # actual value after set_default_erev in preset

    elif name == 'HH AP':
        hhna = neuron.add(HHNa())
        hhna.set_erev(50 * NU.mV)
        leak = neuron.add(Leak())
        leak.gmax = 1 * NU.nS
        leak.set_erev(-55 * NU.mV)
        hhk  = neuron.add(HHK())
        hhk.set_erev(-74 * NU.mV)

    elif name == 'LG AP':
        lgna = neuron.add(LGNa())
        lgna.set_erev(74 * NU.mV)
        lgkf = neuron.add(LGKfast())
        lgkf.set_erev(-90 * NU.mV)
        lgks = neuron.add(LGKslow())
        lgks.set_erev(-90 * NU.mV)
        leak = neuron.add(Leak())
        leak.gmax = 2.5 * NU.nS
        leak.set_erev(-70 * NU.mV)

    else:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_PARAMS)}")

    clamp = neuron.add(PatchClamp(mode=mode))
    if mode == 'ic':
        clamp.set_holding('ic', 0.0)
    else:
        clamp.set_holding('vc', cfg['hold_vc'])

    return sim, neuron, clamp


# ── Protocol runners ───────────────────────────────────────────────────────────

def _run_ic(model_name: str) -> dict:
    """Run current-clamp step protocol."""
    pre_pts   = round(IC_PRE_S   / DT)
    pulse_pts = round(IC_PULSE_S / DT)
    post_pts  = round(IC_POST_S  / DT)
    total_pts = pre_pts + pulse_pts + post_pts

    n_amp = len(IC_AMPLITUDES)
    V_traces = np.zeros((n_amp, total_pts))

    for i, amp in enumerate(IC_AMPLITUDES):
        sim, neuron, clamp = make_model(model_name, mode='ic')
        cmd = np.zeros(total_pts)
        cmd[pre_pts : pre_pts + pulse_pts] = amp
        clamp.queue_command(cmd, DT)
        result = sim.run(blocksize=total_pts)
        V_traces[i] = result['soma.V']

    t = np.arange(total_pts) * DT
    return dict(
        t_ic      = t,
        V_ic      = V_traces,
        I_cmd_ic  = IC_AMPLITUDES.copy(),
    )


def _run_vc(model_name: str) -> dict:
    """Run voltage-clamp step protocol."""
    cfg = MODEL_PARAMS[model_name]
    hold_v = cfg['hold_vc']

    hold_pts = round(VC_HOLD_S / DT)
    step_pts = round(VC_STEP_S / DT)
    tail_pts = round(VC_TAIL_S / DT)
    total_pts = hold_pts + step_pts + tail_pts

    n_steps = len(VC_STEP_VOLTAGES)
    I_traces = np.zeros((n_steps, total_pts))

    for i, v_step in enumerate(VC_STEP_VOLTAGES):
        sim, neuron, clamp = make_model(model_name, mode='vc')
        cmd = np.full(total_pts, hold_v)
        cmd[hold_pts : hold_pts + step_pts] = v_step
        clamp.queue_command(cmd, DT)
        result = sim.run(blocksize=total_pts)
        I_traces[i] = result[clamp, 'I']

    t = np.arange(total_pts) * DT
    return dict(
        t_vc      = t,
        I_vc      = I_traces,
        V_cmd_vc  = VC_STEP_VOLTAGES.copy(),
        hold_vc   = hold_v,
    )


def run_model_tests(model_name: str) -> dict:
    """Run IC and VC protocols for *model_name*.  Returns combined result dict."""
    results: dict = {}
    results.update(_run_ic(model_name))
    results.update(_run_vc(model_name))
    results['model'] = model_name
    return results


# ── Persistence ────────────────────────────────────────────────────────────────

def save_results(results: dict, outdir: str | Path = 'model_results') -> str:
    """Save *results* to a timestamped .npz file.  Returns the file path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_tag = results['model'].replace(' ', '_')

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    filename = f"{model_tag}_{timestamp}.npz"
    filepath = out / filename

    save_dict = {k: np.array(v) if isinstance(v, str) else v
                 for k, v in results.items()}
    save_dict['timestamp'] = timestamp
    save_dict['git_hash']  = git_hash

    np.savez(str(filepath), **save_dict)
    return str(filepath)


def load_results(filepath: str | Path) -> dict:
    """Load an .npz reference file.  Returns a plain dict."""
    data = np.load(str(filepath), allow_pickle=True)
    result = {}
    for k in data.files:
        v = data[k]
        result[k] = v.item() if v.ndim == 0 else v
    return result


def find_latest_reference(ref_dir: str | Path, model_name: str) -> Path | None:
    """Return the newest .npz reference for *model_name*, or None."""
    ref_dir = Path(ref_dir)
    if not ref_dir.exists():
        return None
    tag = model_name.replace(' ', '_')
    matches = sorted(ref_dir.glob(f"{tag}_*.npz"))
    return matches[-1] if matches else None


def find_oldest_reference(ref_dir: str | Path, model_name: str) -> Path | None:
    """Return the oldest .npz reference for *model_name*, or None."""
    ref_dir = Path(ref_dir)
    if not ref_dir.exists():
        return None
    tag = model_name.replace(' ', '_')
    matches = sorted(ref_dir.glob(f"{tag}_*.npz"))
    return matches[0] if matches else None


def invalidate_oldest_reference(
    ref_dir: str | Path,
    model_name: str,
) -> Path | None:
    """Move the oldest reference for *model_name* to an 'archived' subdirectory.

    Returns the destination path, or None if no reference existed.
    The file is moved rather than deleted so it can be recovered if needed.
    """
    oldest = find_oldest_reference(ref_dir, model_name)
    if oldest is None:
        return None
    archive_dir = Path(ref_dir) / 'archived'
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / oldest.name
    oldest.rename(dest)
    return dest


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare_results(
    reference: dict,
    current: dict,
    threshold_v: float = 1e-6,    # 1 µV
    threshold_i: float = 1e-13,   # 0.1 pA
) -> dict:
    """Point-by-point comparison of *current* results against *reference*.

    Returns a dict with:
        max_V_deviation      – maximum |ΔV| across all IC traces
        rms_V_error          – RMS |ΔV|
        discrepant_V_points  – number of IC sample points where |ΔV| > threshold_v
        max_I_deviation      – maximum |ΔI| across all VC traces
        rms_I_error          – RMS |ΔI|
        discrepant_I_points  – number of VC sample points where |ΔI| > threshold_i
        V_diff               – (n_amp, n_pts) difference array for IC
        I_diff               – (n_v, n_pts) difference array for VC
        threshold_v, threshold_i – thresholds used
    """
    V_ref = reference['V_ic']
    V_cur = current['V_ic']
    I_ref = reference['I_vc']
    I_cur = current['I_vc']

    if V_ref.shape != V_cur.shape:
        raise ValueError(
            f"IC shape mismatch: reference {V_ref.shape} vs current {V_cur.shape}"
        )
    if I_ref.shape != I_cur.shape:
        raise ValueError(
            f"VC shape mismatch: reference {I_ref.shape} vs current {I_cur.shape}"
        )

    V_diff = V_cur - V_ref
    I_diff = I_cur - I_ref

    return {
        'max_V_deviation':     float(np.max(np.abs(V_diff))),
        'rms_V_error':         float(np.sqrt(np.mean(V_diff ** 2))),
        'discrepant_V_points': int(np.sum(np.abs(V_diff) > threshold_v)),
        'max_I_deviation':     float(np.max(np.abs(I_diff))),
        'rms_I_error':         float(np.sqrt(np.mean(I_diff ** 2))),
        'discrepant_I_points': int(np.sum(np.abs(I_diff) > threshold_i)),
        'V_diff':              V_diff,
        'I_diff':              I_diff,
        'threshold_v':         threshold_v,
        'threshold_i':         threshold_i,
    }
