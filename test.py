# -*- coding: utf-8 -*-
"""
NeuroDemo - Physiological neuron sandbox for educational purposes
Luke Campagnola 2015
"""
import argparse
import pyqtgraph as pg
import neurodemo.units as NU
import neurodemo as ND
import numpy as np

pg.setConfigOption('antialias', True)
app = pg.mkQApp() # QtGui.QWidget.QApplication([])
app.setStyle("fusion")
class TestSim(pg.QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

    def run(self, simtype:str="LG"):

        if simtype == "HH":
            # HH Simulation
            sim = ND.Sim(temp=6.3)
            neuron = ND.Section()
            leak = neuron.add(ND.Leak(gbar=0.1*NU.mS/NU.cm**2))
            hhk = neuron.add(ND.HHK())
            hhna = neuron.add(ND.HHNa())

        elif simtype == 'LG':
            # Lewis & Gerstner cortical neuron
            sim = ND.Sim(temp=37)
            neuron = ND.Section(vm=-70*NU.mV)
            lgna = neuron.add(ND.LGNa())
            lgkf = neuron.add(ND.LGKfast())
            lgks = neuron.add(ND.LGKslow())
            leak = neuron.add(ND.Leak(gbar=0.25*NU.mS/NU.cm**2)) #, erev=-70*NU.mV))

        clamp = ND.PatchClamp(mode='ic')
        neuron.add(clamp)
        sim.add(neuron)


        self.win.setWindowTitle(f"Testing {simtype:s} Sim.py")
        p1 = self.win.addPlot(title='IC', labels={'left': ('Vm', 'V')})
        p2 = self.win.addPlot(labels={'left': ('Ipip', 'A')}, row=1, col=0)
        self.win.ci.layout.setRowFixedHeight(1, 150)
        p3 = self.win.addPlot(row=2, col=0)
        self.win.ci.layout.setRowFixedHeight(2, 150)
        print(sim.dt)
        sim.dt = sim.dt*NU.us
        dur = 100 * NU.ms
        npts = int(dur / sim.dt)
        print(dur, npts, sim.dt)
        x1 = int(20*NU.ms / sim.dt)
        x2 = int(80*NU.ms / sim.dt)
        x = np.linspace(-200, 200, 11) * NU.pA
        #x = [0*pA]
        cmd = np.zeros((len(x), npts)) #*-65e-3
        data = np.zeros((len(x), npts, 9))
        t = np.arange(npts) * sim.dt
        for i, v in enumerate(x):
            print("i: ", i, "v: ", v)
            cmd[i, x1:x2] = v
            clamp.queue_command(cmd[i], sim.dt)
            #data[i] = run(neuron, mode='ic', dt=dt, cmd=cmd[i])
            result = sim.run(npts)
            data = result[neuron, 'V']
            p1.plot(t, data, pen=(i, 15))
            p2.plot(t, cmd[i], pen=(i, 15))
            #p2.plot(t, clamp.current(result), pen=(i, 15))
            
            #p3.plot(t, leak.current(result), pen=(i, 15))
            #p3.setLabel('left', 'Ileak', 'A')

            #p3.plot(t, lgkf.current(result), pen=(i, 15))
            #p3.setLabel('left', 'LG Kfast Current', 'A')

            #p3.plot(t, lgks.open_probability(result), pen=(i, 15))
            #p3.setLabel('left', 'LG Kslow O.P.')

            #p3.plot(t, lgna.current(result), pen=(i, 15))
            #p3.setLabel('left', 'LG Na Current', 'A')

            if simtype == "LG":
                # p3.plot(t, lgna.open_probability(result), pen=(i, 15))
                # p3.setLabel('left', 'LG Na O.P.')
                p3.plot(t, lgna.current(result), pen=(i, 15))
                p3.setLabel('left', 'LG Na Current', 'A')

            elif simtype == "HH":
                p3.plot(t, hhna.open_probability(result), pen=(i, 15))
                p3.setLabel('left', 'HH Na O.P.')
    
    
    def set_window(self, parent=None):
        self.fullscreen_widget = None
        self.resize(1024, 768)
        self.win = pg.GraphicsLayoutWidget(title="Test Simulator")
        layout = pg.QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.win)
        self.setLayout(layout)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('simtype', default="LG", help="Test the Sim.py module",
                                   choices=["LG", "HH"]
                                   )
    args = parser.parse_args()
    Tester = TestSim()
    Tester.set_window()
    Tester.run(args.simtype)
    Tester.show()
    print("show called")
    import sys
    if sys.flags.interactive == 0:
        print("excc")
        app.exec()
if __name__ == "__main__":
    main()
 
