# -*- coding: utf-8 -*-
"""
NeuroDemo - Physiological neuron sandbox for educational purposes
Luke Campagnola 2015
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree as pt
from lmfit import Model
from lmfit.models import ExponentialModel

class TraceAnalyzer(QtGui.QWidget):
    def __init__(self, seq_plotter):
        QtGui.QWidget.__init__(self)
        self.plotter = seq_plotter
        
        self.layout = QtGui.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.hsplitter = QtGui.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.layout.addWidget(self.hsplitter)
        
        self.ptree = pt.ParameterTree(showHeader=False)
        self.hsplitter.addWidget(self.ptree)
        
        self.table = pg.TableWidget()
        self.hsplitter.addWidget(self.table)
        self.table.verticalHeader().hide()
        
        self.analysis_plot = EvalPlotter()
        self.hsplitter.addWidget(self.analysis_plot)
        
        self.hsplitter.setSizes([300, 200, 400])
        
        self.clear()
        
        self.params = TraceAnalyzerGroup(name="Analyzers")
        self.params.need_update.connect(self.update_analyzers)
        self.ptree.setParameters(self.params)

    def clear(self):
        self.data = []
        self.table.clear()
        
    def add_data(self, t, data, info):
        self.params.set_inputs(data.dtype.names)
        self.data.append((t, data, info))
        self.update_analysis()

    def update_analyzers(self):
        for anal in self.params.children():
            self.plotter.plots[anal['Input']].addItem(anal.rgn, ignoreBounds=True)
        self.update_analysis()
        
    def update_analysis(self):
        fields = ['cmd'] + [anal.name() for anal in self.params.children()]
        data = np.empty(len(self.data), dtype=[(str(f), float) for f in fields])
        for i, rec in enumerate(self.data):
            # print(rec)
            t, d, info = rec
            data['cmd'][i] = info['amp']
            for analysis in self.params.children():
                data[analysis.name()][i] = analysis.process(t, d)
        self.table.setData(data)
        self.analysis_plot.update_data(data)
        

class TraceAnalyzerGroup(pt.parameterTypes.GroupParameter):
    need_update = QtCore.Signal()

    def __init__(self, **kwds):
        analyses = ['min', 'max', 'mean', 'exp_tau', 'spike_count', 'spike_latency']
        self.inputs = []
        pt.parameterTypes.GroupParameter.__init__(self, addText='Add analysis..', addList=analyses, **kwds)

    def addNew(self, typ):
        param = TraceAnalyzerParameter(name=typ, analysis_type=typ, inputs=self.inputs, autoIncrementName=True)
        self.addChild(param)
        param.need_update.connect(self.need_update)
        self.need_update.emit()
        
    def set_inputs(self, inputs):
        self.inputs = list(inputs)
        self.inputs.remove('t')
        for ch in self.children():
            ch.set_input_list(inputs)


class TraceAnalyzerParameter(pt.parameterTypes.GroupParameter):
    need_update = QtCore.Signal(object)  # self

    def __init__(self, **kwds):
        kwds.update({'removable': True, 'renamable': False})
        childs = [
            dict(name='Input', type='list', values=kwds.pop('inputs')),
            dict(name='Type', type='list', value=kwds.pop('analysis_type'), values=['mean', 'min', 'max', 'exp_tau', 'spike_count', 'spike_latency']),
            dict(name='Start', type='float', value=0, suffix='s', siPrefix=True, step=5e-3),
            dict(name='End', type='float', value=10e-3, suffix='s', siPrefix=True, step=5e-3),
            dict(name='Threshold', type='float', value=-30e-3, suffix='V', siPrefix=True, step=5e-3, visible=False),
        ]
        kwds['children'] = childs + kwds.get('children', [])
        
        pt.parameterTypes.GroupParameter.__init__(self, **kwds)
        self.sigTreeStateChanged.connect(self.tree_changed)
        
        self.rgn = pg.LinearRegionItem([self['Start'], self['End']])
        self.rgn.sigRegionChanged.connect(self.region_changed)

        self.show_threshold_param()
    
    def tree_changed(self, root, changes):
        for param, change, val in changes:
            if change != 'value':
                continue
            if param is self.child('Start') or param is self.child('End'):
                self.rgn.sigRegionChanged.disconnect(self.region_changed)
                try:
                    self.rgn.setRegion([self['Start'], self['End']])
                finally:
                    self.rgn.sigRegionChanged.connect(self.region_changed)
            elif param is self.child('Type'):
                self.show_threshold_param()
            self.need_update.emit(self)

    def show_threshold_param(self):
        needs_threshold = self['Type'] in ['spike_count', 'spike_latency']
        self.child('Threshold').setOpts(visible=needs_threshold)

    def region_changed(self):
        """If the region is changed, read the position and update the values
        """        
        start, end = self.rgn.getRegion()
        self.sigTreeStateChanged.disconnect(self.tree_changed)
        try:
            self['Start'] = start
            self['End'] = end
        finally:
            self.sigTreeStateChanged.connect(self.tree_changed)
            
        self.need_update.emit(self)
            
    def set_input_list(self, inputs):
        self.child('Input').setLimits(inputs)

    def process(self, t, data):
        dt = t[1] - t[0]
        i1 = int(self['Start'] / dt)
        i2 = int(self['End'] / dt)
        data = data[self['Input']][i1:i2]
        sign = 1.0
        if self['Input'] in [
            "soma.INa.I", "soma.IK.I", "soma.IKA.I", 
            "soma.ICaT.I", "soma.ICaL.I",
            "soma.IH.I",
            "soma.INa1.I", "soma.IKf.I", "soma.IKs.I", 
             ]:
            sign = -1.0   # flip sign of cation currents for display
        t = t[i1:i2]
 
        typ = self['Type']
        if typ == 'mean':
            return sign*data.mean()
        elif typ == 'min':
            return sign*data.min()
        elif typ == 'max':
            return sign*data.max()
        elif typ.startswith('spike'):
            spikes = np.argwhere((data[1:] > self['Threshold']) & (data[:-1] < self['Threshold']))[:,0]
            if typ == 'spike_count':
                return len(spikes)
            elif typ == 'spike_latency':
                if len(spikes) == 0:
                    return np.nan
                else:
                    return spikes[0] * dt
        elif typ == 'exp_tau':
            return self.measure_tauDecay(data, t)
        elif typ == 'expTauRise4':
            return(self.measure_tauRise4(data, t))
            
    def measure_tau_old(self, data, t):
        from scipy.optimize import curve_fit
        dt = t[1] - t[0]
        def expfn(x, yoff, amp, tau):
            return yoff + amp * np.exp(-x / tau)
        guess = (data[-1], data[0] - data[-1], t[-1] - t[0])
        fit = curve_fit(expfn, t-t[0], data, guess)
        return fit[0][2]

    def measure_tauDecay(self, data, t):
        model = ExponentialModel()
        pars = model.guess(data-data[-1], x=t-t[0])
        result = model.fit(data-data[-1], pars,  x=t-t[0], nan_policy="propagate")
        return result.params['decay'] # fit[0][2]            
        
    def measure_tauRise4(self, data, t):
        # this is not working quite right yet... 
        print('taurise4')
        def expfn4(x, amp, tau):
            return amp * ((1.0-np.exp(-x / tau))**4.0)
        emodel = Model(expfn4)
        params = emodel.make_params()
        d = data-data[0]
        tp = t-t[0]
        emodel.set_param_hint('tau', value=t[-1]-t[0], min=0.0001)
        emodel.set_param_hint('amp', value=data[-1]-data[0])
        result = emodel.fit(d[1:], params, x=tp[1:])
        print(result.params)
        return result.params['tau'] # fit[0][2]       

class EvalPlotter(QtGui.QWidget):
    def __init__(self):
        self.held_plots = []
        self.last_curve = None
        self.held_index = 0
        self.cursor_visible = False
        
        QtGui.QWidget.__init__(self)
        self.layout = QtGui.QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)
        
        self.x_label = QtGui.QLabel('X data')
        self.y_label = QtGui.QLabel('Y data')
        self.layout.addWidget(self.x_label, 0, 0)
        self.layout.addWidget(self.y_label, 1, 0)

        self.x_code = QtGui.QLineEdit('cmd')
        self.y_code = QtGui.QLineEdit()
        self.layout.addWidget(self.x_code, 0, 1, 1, 2)
        self.layout.addWidget(self.y_code, 1, 1, 1, 2)
        
        self.x_units_label = QtGui.QLabel('units')
        self.y_units_label = QtGui.QLabel('units')
        self.layout.addWidget(self.x_units_label, 0, 3)
        self.layout.addWidget(self.y_units_label, 1, 3)
        
        self.x_units_text = QtGui.QLineEdit('A')
        self.y_units_text = QtGui.QLineEdit()
        self.layout.addWidget(self.x_units_text, 0, 3)
        self.layout.addWidget(self.y_units_text, 1, 3)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 6)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)

        self.plot = pg.PlotWidget()
        self.layout.addWidget(self.plot, 2, 0, 1, -1)
        self.layout.setColumnStretch(1, 1)
        self.hold_plot_btn = QtGui.QPushButton('Hold Plot')
        self.layout.addWidget(self.hold_plot_btn, 3, 0, 1, 1)
        self.clear_plot_btn = QtGui.QPushButton('Clear Plot')
        self.layout.addWidget(self.clear_plot_btn, 3, 1, 1, 2)
        self.replot_btn = QtGui.QPushButton('Replot')
        self.layout.addWidget(self.replot_btn, 3, 3, 1, 1)
        self.show_cursor_check = QtGui.QCheckBox('Enable Cursor')
        self.show_cursor_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.layout.addWidget(self.show_cursor_check, 3, 4, 1, 1)
        
        self.x_code.editingFinished.connect(self.replot)
        self.y_code.editingFinished.connect(self.replot)
        self.x_units_text.editingFinished.connect(self.replot)
        self.y_units_text.editingFinished.connect(self.replot)
        self.hold_plot_btn.clicked.connect(self.hold_plot)
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        self.replot_btn.clicked.connect(self.replot)
        self.show_cursor_check.stateChanged.connect(self.cursor_state_changed)

        # add a crosshair to the plot, but hide until ready
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.vLine.setVisible(False)
        self.hLine.setVisible(False)
        self.mouse_label = pg.TextItem(' ', color='c', anchor=[0.5, 1.0])
        self.mouse_label.setVisible(False)
     
        self.plot.addItem(self.vLine, ignoreBounds=False)
        self.plot.addItem(self.hLine, ignoreBounds=False)
        self.plot.addItem(self.mouse_label)
        self.vLine.scene().sigMouseHover.connect(self.mouse_moved_over_plot)  # enable cursor


    def mouse_moved_over_plot(self, evt):
        # if self.last_curve is None and self.held_index == 0:
        #     return
        if not self.cursor_visible:
            return
        pos = evt[0]
        widget = pos.getViewWidget()
        globalPos = pg.QtGui.QCursor.pos()
        localPos = widget.mapFromGlobal(globalPos)
        scenePos = pos.mapFromDevice(localPos)
        viewPos = pos.vb.mapSceneToView(scenePos)
        self.vLine.setPos(viewPos.x())
        self.hLine.setPos(viewPos.y())
        self.mouse_label.setPos(viewPos.x(), viewPos.y())
        xt = f"<span style='font-size: 10pt; color:cyan'>x={viewPos.x():0.3e}</span style><br>"
        yt = f"<span style='font-size: 10pt; color:cyan'>y={viewPos.y():0.3e}</span style>"
        self.mouse_label.setHtml(xt + yt)
    
    def update_data(self, data):
        self.data = data
        self.replot()
        
    def replot(self):
        data = self.data
        ns = {}
        for k in data.dtype.names:
            ns[k.replace(' ', '_')] = data[k]
        xcode = str(self.x_code.text())
        ycode = str(self.y_code.text())
        if xcode == '' or ycode == '':
            return
        
        try:
            x = eval(xcode, ns)
        except:
            pg.debug.printExc('Error evaluating plot x values:')
            self.x_code.setStyleSheet("QLineEdit { border: 2px solid red; }")
            return
        else:
            self.x_code.setStyleSheet("")
            
        try:
            y = eval(ycode, ns)
        except:
            pg.debug.printExc('Error evaluating plot y values:')
            self.y_code.setStyleSheet("QLineEdit { border: 2px solid red; }")
            return
        else:
            self.y_code.setStyleSheet("")
            
        if self.last_curve is None:
            self.last_curve = self.plot.plot(x, y, symbol='o', symbolBrush=(self.held_index, 10))
        else:
            self.last_curve.setData(x, y)
        self.x_data = x
        self.y_data = y  # for mouse   
        self.plot.setLabels(bottom=(xcode, self.x_units_text.text()),
                            left=(ycode, self.y_units_text.text()))

        
    def hold_plot(self):
        if self.last_curve is None:
            return
        self.held_plots.append(self.last_curve)
        self.held_index += 1
        self.last_curve = None

    def clear_plot(self):
        self.held_plots = []
        self.held_index = 0
        self.vLine.scene().sigMouseHover.disconnect(self.mouse_moved_over_plot)
        self.plot.clear()
        # re-add the cursor
        self.plot.addItem(self.vLine, ignoreBounds=False)
        self.plot.addItem(self.hLine, ignoreBounds=False)
        self.plot.addItem(self.mouse_label)
        self.vLine.scene().sigMouseHover.connect(self.mouse_moved_over_plot)  # enable cursor
        self.vLine.scene().sigMouseHover.connect(self.mouse_moved_over_plot)  # reenable cursor
        self.cursor_visible = False
        self.show_cursor_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.replot()

    def cursor_state_changed(self):
        if self.show_cursor_check.checkState() == QtCore.Qt.CheckState.Checked:
            self.vLine.setVisible(True)
            self.hLine.setVisible(True)
            self.mouse_label.setVisible(True)
            self.cursor_visible = True
        else:
            self.mouse_label.setVisible(False)
            self.vLine.setVisible(False)
            self.hLine.setVisible(False)
            self.cursor_visible = False
