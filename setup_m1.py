"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['demo.py']
DATA_FILES = []
OPTIONS = {
    'iconfile':'icon.icns',
    'plist': {'CFBundleShortVersionString':'1.1.0',},
    'arch': "arm64",
}


setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
        entry_points={
        'console_scripts': [
            'neurodemo=demo:main',
        ]
        }
)
