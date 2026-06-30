#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neurodemo model regression tests.

Compares freshly simulated results against the **oldest** saved reference
for each model.  Silent on success; prints a report and exits 1 on failure.

Usage
-----
Via pytest (collects test_* functions)::

    pytest tests.py
    pytest tests.py -v

Direct execution (silent runner, Unix-friendly)::

    python tests.py

Invalidate (archive) the oldest reference for a specific model::

    python tests.py --invalidate "HH AP"

List all current reference files (oldest → newest per model)::

    python tests.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from neurodemo.tests.sim_protocols import (
    MODEL_PARAMS,
    compare_results,
    find_oldest_reference,
    invalidate_oldest_reference,
    load_results,
    run_model_tests,
)

# ── Configuration ──────────────────────────────────────────────────────────────

REFERENCE_DIR = Path(__file__).parent / 'model_results'
V_THRESHOLD   = 1e-6    # 1 µV
I_THRESHOLD   = 1e-13   # 0.1 pA


# ── Pytest test functions ──────────────────────────────────────────────────────

@pytest.mark.parametrize('model_name', list(MODEL_PARAMS))
def test_model_regression(model_name: str) -> None:
    """Compare current simulation against the oldest reference for model_name."""
    ref_path = find_oldest_reference(REFERENCE_DIR, model_name)
    if ref_path is None:
        pytest.skip(
            f"No reference for '{model_name}'. "
            "Run `python run_model_tests.py`, select the model, "
            "run tests, and click 'Save Reference'."
        )

    reference = load_results(ref_path)
    results   = run_model_tests(model_name)
    comp      = compare_results(reference, results, V_THRESHOLD, I_THRESHOLD)

    nv = comp['discrepant_V_points']
    ni = comp['discrepant_I_points']

    assert nv == 0, (
        f"IC voltage: {nv} points deviate > {V_THRESHOLD*1e6:.1f} µV  "
        f"(max {comp['max_V_deviation']*1e6:.4f} µV, "
        f"RMS {comp['rms_V_error']*1e6:.4f} µV)  "
        f"ref: {ref_path.name}"
    )
    assert ni == 0, (
        f"VC current: {ni} points deviate > {I_THRESHOLD*1e12:.2f} pA  "
        f"(max {comp['max_I_deviation']*1e12:.4f} pA, "
        f"RMS {comp['rms_I_error']*1e12:.4f} pA)  "
        f"ref: {ref_path.name}"
    )


# ── Silent runner (python tests.py) ───────────────────────────────────────────

def _run_all(ref_dir: Path) -> int:
    """Run all models, print per-model status, return 0 on pass / 1 on failure."""
    failures: list[tuple[str, Path, dict]] = []
    width = max(len(n) for n in MODEL_PARAMS)

    for model_name in MODEL_PARAMS:
        ref_path = find_oldest_reference(ref_dir, model_name)
        if ref_path is None:
            print(f"  {model_name:<{width}}  SKIP  (no reference — run run_model_tests.py first)",
                  flush=True)
            continue

        # Print "running" immediately so the user sees progress while the
        # simulation runs (IC + VC protocols can take several seconds).
        print(f"  {model_name:<{width}}  running…", end='\r', flush=True)

        reference = load_results(ref_path)
        results   = run_model_tests(model_name)

        try:
            comp = compare_results(reference, results, V_THRESHOLD, I_THRESHOLD)
        except ValueError as exc:
            print(f"  {model_name:<{width}}  ERROR  {exc}", flush=True)
            failures.append((model_name, ref_path, {}))
            continue

        nv = comp['discrepant_V_points']
        ni = comp['discrepant_I_points']
        if nv > 0 or ni > 0:
            print(f"  {model_name:<{width}}  FAIL          ", flush=True)
            failures.append((model_name, ref_path, comp))
        else:
            print(f"  {model_name:<{width}}  PASS  (ref: {ref_path.name})", flush=True)

    if not failures:
        return 0

    for model_name, ref_path, comp in failures:
        print(f"\n{'='*60}")
        print(f"FAIL  {model_name}  (ref: {ref_path.name})")
        if comp:
            nv, ni = comp['discrepant_V_points'], comp['discrepant_I_points']
            print(
                f"  IC  {nv:>6} discrepant pts  "
                f"max ΔV={comp['max_V_deviation']*1e6:.4f} µV  "
                f"RMS={comp['rms_V_error']*1e6:.4f} µV"
            )
            print(
                f"  VC  {ni:>6} discrepant pts  "
                f"max ΔI={comp['max_I_deviation']*1e12:.4f} pA  "
                f"RMS={comp['rms_I_error']*1e12:.4f} pA"
            )
    return 1


def _list_references(ref_dir: Path) -> None:
    """Print all reference files grouped by model, oldest first."""
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        return

    any_found = False
    for model_name in MODEL_PARAMS:
        tag = model_name.replace(' ', '_')
        matches = sorted(ref_dir.glob(f"{tag}_*.npz"))
        if matches:
            any_found = True
            print(f"\n{model_name}:")
            for i, p in enumerate(matches):
                suffix = '  ← baseline' if i == 0 else ''
                print(f"  {p.name}{suffix}")

    archive = ref_dir / 'archived'
    archived = sorted(archive.glob('*.npz')) if archive.exists() else []
    if archived:
        print(f"\nArchived  ({archive}):")
        for p in archived:
            print(f"  {p.name}")

    if not any_found and not archived:
        print("No reference files found.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--invalidate', metavar='MODEL',
        help='Archive the oldest reference for MODEL (e.g. "HH AP").',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all reference files and exit.',
    )
    parser.add_argument(
        '--ref-dir', metavar='DIR', default=str(REFERENCE_DIR),
        help=f'Reference directory (default: {REFERENCE_DIR})',
    )
    args = parser.parse_args()
    ref_dir = Path(args.ref_dir)

    if args.list:
        _list_references(ref_dir)
        return 0

    if args.invalidate:
        model = args.invalidate
        if model not in MODEL_PARAMS:
            print(
                f"Unknown model {model!r}.  Available: {list(MODEL_PARAMS)}",
                file=sys.stderr,
            )
            return 2
        dest = invalidate_oldest_reference(ref_dir, model)
        if dest is None:
            print(f"No reference found for '{model}'.")
        else:
            print(f"Archived  {dest.name}  →  {dest.parent}")
        return 0

    return _run_all(ref_dir)


if __name__ == '__main__':
    sys.exit(main())
