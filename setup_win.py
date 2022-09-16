"""
This is a setup.py script generated by py2applet

Usage:
    python setup_win.py py2exe
"""
# from setuptools import setup

from setuptools import setup
import py2exe

# base="Win32GUI" should be used only for Windows GUI app
import sys
base = None
if sys.platform == "win32":
    base = "Win32GUI"
if sys.platform == "win64":
    base = "Win64GUI"

APP = 'demo.py'
DATA_FILES = []
OPTIONS = {
        # this forces the inclusion of ctypes.util, which is required to load 
        # the opengl win32 backend. Neither py2exe nor cx_freeze seem to
        # be able to autodiscover this, which leads to application that 
        # crash without printing an error
        'packages': ['ctypes', "neurodemo"],  
        'includes': ["numpy", "scipy", "pyqtgraph", "lmfit", 
            'PyQt6.sip',
            'pkg_resources',
            'PyQt6',
            'PyQt6.QtCore',
            'PyQt6.QtOpenGL',
            'PyQt6.QtGui',
            'pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt6', # Add This one
            'pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt6', # Add This one
            'pyqtgraph.imageview.ImageViewTemplate_pyqt6', # Add This one
            'PyQt6.QtPrintSupport',
        ],
        'excludes':  ['tkinter'],
        'dist_dir': "dist"
        }
    


setup(
    name="NeuroDemo",
    version="1.1.0",
    description="Neuron Demonstration Simulator",
    windows=['demo.py'],
    data_files=DATA_FILES,
    options={'py2exe': OPTIONS},
    setup_requires=['py2exe'],
    console=['demo.py'],
    #entry_points = {"console_scripts": ["demo=demo.__main__:main"]},
    # entry_points={
    #     'console_scripts': [
    #         'demo=demo:main',
    #     ]
    #   }
 )


# pyinstaller/cx_Freeze

# 

# from cx_Freeze import setup, Executable

# # Dependencies are automatically detected, but it might need fine tuning.
# additional_modules = ["ctypes"]

# build_exe_options = {"includes": additional_modules,
#                      "packages": ["numpy", "scipy", "pyqtgraph", "PyQt6", "lmfit"],
#                      "excludes": [],
#                      "include_files": ["pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt6"], # ["res"], # ['icon.ico', 'res']
#                      }
# directory_table = [
#     ("ProgramMenuFolder", "TARGETDIR", "."),
#     ("MyProgramMenu", "ProgramMenuFolder", "NEUROD~1|NeuronDemo"),
# ]


# msi_data = {
#     "Directory": directory_table,
#     "ProgId": [
#         ("Prog.Id", None, None, "Neuron Simulator Demonstration", "IconId", None),
#     ],
#     "Icon": [
#         ("IconId", "icon.ico"),
#     ],
# }

# bdist_msi_options = {
#     "add_to_path": True,
#     "data": msi_data,
#     "target_name": "NeuroDemo.exe",
#     "environment_variables": [
#         ("NEURODEMO_VAR", "=-*NEURODEMO_VAR", "1", "TARGETDIR")
#     ],
#     "upgrade_code": "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}",
# }

# build_exe_options = {"excludes": ["tkinter"], "include_msvcr": True}
# executables = (
#     [
#         Executable(
#             script="demo.py",
#             copyright="Copyright (C) 2022",
#             base=base,
#             icon="", # "icon.ico",
#             shortcut_name="NeuroDemo",
#             shortcut_dir="NeuroDemoProg",
#             target_name="NeuroDemo1_1.exe",
#            # main_script="demo=demo.__main__:main",
#         ),
#     ],
# )


# setup(
#     name="demo",
    
#     version="1.1.0",
#     description="Neurondemo",
#     long_description_content_type="text/markdown",
#     url="https://github.com/pbmanis/neurodemo",
#     author="pbm/lc",
#     author_email="pmanis@med.unc.edu",
#     license="MIT",
#     classifiers=[
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3",
#     ],
#     packages=["neurodemo"],
#     include_package_data=True,
#     install_requires=[
#         "ctypes",*- "pyqt6", "pyqtgraph", "numpy", "scipy", "lmfit",
#     ],
#     entry_points={"console_scripts": ["demo=demo.__main__:main"]},
#     executables = executables,
#     options={"build_exe": build_exe_options,
#              "build_msi": bdist_msi_options,
#              },
# )


