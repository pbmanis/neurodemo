Neuron Demonstration
====================

Luke Campagnola, 2015


This is an educational simulation of a simple neuron.

* Hodgkin & Huxley channels
* Lewis & Gerstner (2002) cortical channels
* McCormick & Huguenard (1992) thalamic model.
* Destexhe 1993 Ih channel
* Current/voltage clamp electrode with access resistance
* Diagram of cell membrane with circuit schematic
* Realtime simulation and plotting of voltages, currents, open probabilities, etc.
* Analysis tool for generating I/V curves and similar analyses.



Requirements
------------

* Python 3.13 (pinned in `pyproject.toml`)
* NumPy, SciPy
* PyQt6
* PyQtGraph
* lmfit

Dependency management uses [uv](https://docs.astral.sh/uv/).


Installation
------------

1. Git clone the repository.
2. Run `uv sync` in the repository root. This creates a `.venv` with all required packages.


Running the Demo
-----------------

From the repository root:

```
> uv run python neurodemo.py
```


Building a standalone application
----------------------------------

Both platforms are built with [PyInstaller](https://pyinstaller.org/).

* **macOS (Apple Silicon)**: run `./make_M1_dmg.sh`. Requires `create-dmg` (`brew install create-dmg`). Produces `dist_m1/neurodemo_M1.dmg`.
* **Windows (Intel x86_64)**: run `make_win_exe.bat` from an environment with the dev extras installed (`uv sync --extra dev`). Produces `dist_exe\NeuroDemo.exe`. This path has not yet been verified end-to-end (build it on a Windows machine and check `setup_win.spec` if you hit missing-DLL errors).
