@echo off
REM build the Windows exe with PyInstaller
REM run this from an activated environment that has pyinstaller installed
REM   uv sync --extra dev
REM   uv run make_win_exe.bat   (or activate .venv and run directly)

if exist dist_exe rmdir /s /q dist_exe
if exist build_exe rmdir /s /q build_exe

pyinstaller setup_win.spec --distpath dist_exe --workpath build_exe

echo Build complete: dist_exe\NeuroDemo.exe
