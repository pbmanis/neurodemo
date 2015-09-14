# -*- coding: utf-8 -*-
"""
NeuroDemo - Physiological neuron sandbox for educational purposes
Luke Campagnola 2015
"""
from __future__ import division, unicode_literals
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree as pt


class TraceAnalyzer(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        
        self.ptree = pt.ParameterTree()
        self.splitter.addWidget(self.ptree)
        
        self.table = pg.TableWidget()
        self.splitter.addWidget(self.table)
        self.splitter.setSizes([200, 600])
        
        self.clear()
        
        self.params = TraceAnalyzerGroup(name='analyzers')
        self.ptree.setParameters(self.params)
        self.params.update_inputs(['Vm', 'Ipip'])

    def clear(self):
        self.data = np.empty(0, dtype=[(str('command'), float)]) # note: no unicode allowed in py2/numpy dtypes
        self.table.clear()
        
    def add_data(self, t, v, i, info):
        mode, amp, cmd, seq_ind, seq_len = info
        rec = np.array([(amp,)], dtype=self.data.dtype)
        self.data = np.append(self.data, rec)
        self.table.setData(self.data)
        self.update_analysis()
        
    def update_analysis(self):
        pass


class TraceAnalyzerGroup(pt.parameterTypes.GroupParameter):
    inputs_changed = QtCore.Signal()
    analyzers_changed = QtCore.Signal()
    need_update = QtCore.Signal()

    def __init__(self, **kwds):
        analyses = ['min', 'max', 'mean', 'exp tau', 'spike count', 'spike latency']
        pt.parameterTypes.GroupParameter.__init__(self, addText='Add analysis..', addList=analyses, **kwds)

    def addNew(self, typ):
        param_classes = {
            'min': MinAnalyzer,
            'max': MaxAnalyzer,
            'mean': MeanAnalyzer,
            'exp tau': ExpTauAnalyzer,
            'spike count': SpikeCountAnalyzer,
            'spike latency': SpikeLatencyAnalyzer,
        }
        param = param_classes[typ](name=typ, autoIncrementName=True)
        self.addChild(param)
        param.need_update.connect(self.need_update)
        param.input_changed.connect(self.inputs_changed)
        self.analyzers_changed.emit()
        
    def update_inputs(self, inputs):
        for ch in self.children():
            ch.set_input_list(inputs)


class TraceAnalyzerParameter(pt.parameterTypes.GroupParameter):
    input_changed = QtCore.Signal(object, object)  # self, input
    need_update = QtCore.Signal(object)  # self

    def __init__(self, **kwds):
        kwds.update({'removable': True, 'renamable': True})
        childs = [
            dict(name='Input', type='list', values=[]),
            dict(name='Start', type='float', value=0, suffix='s', siPrefix=True),
            dict(name='End', type='float', value=10e-3, suffix='s', siPrefix=True),
        ]
        kwds['children'] = childs + kwds.get('children', [])
        
        pt.parameterTypes.GroupParameter.__init__(self, **kwds)
        self.sigTreeStateChanged.connect(self.tree_changed)
        
        self.rgn = pg.LinearRegionItem()
        self.rgn.sigRegionChanged.connect(self.region_changed)
    
    def tree_changed(self, root, changes):
        for param, change, val in changes:
            if change != 'value':
                continue
            if param is self.child('Input'):
                self.input_changed.emit(self, val)
            elif param is self.child('Start') or param is self.child('End'):
                self.rgn.setRegion([self['Start'], self['End']])
            else:
                self.need_update.emit(self)
                
    def region_changed(self):
        self.need_update.emit(self)
            
    def set_input_list(self, inputs):
        self.child('Input').setValues(inputs)

    def process(self, data):
        raise NotImplementedError()


class MinAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    def process(self, data):
        return data.min()


class MaxAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    def process(self, data):
        return data.max()


class MeanAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    def process(self, data):
        return data.mean()


class ExpTauAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    #def process(self, data):
        #return data.mean()


class SpikeCountAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    def process(self, data):
        return data.mean()


class SpikeLatencyAnalyzer(TraceAnalyzerParameter):
    def __init__(self, **kwds):
        TraceAnalyzerParameter.__init__(self, **kwds)

    def process(self, data):
        return data.mean()


