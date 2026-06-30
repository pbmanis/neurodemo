#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for the neurodemo model regression test app.

Usage:
    python run_model_tests.py
"""
import sys
import pyqtgraph as pg
from neurodemo.tests.test_window import ModelTestWindow


def main():
    app = pg.mkQApp('Neurodemo Model Tests')
    app.setStyle('Fusion')
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

    win = ModelTestWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
