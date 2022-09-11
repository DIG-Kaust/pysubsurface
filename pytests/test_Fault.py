import pytest

import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.objects import Fault

faultfile = 'testdata/Fault/Depth/testfault.dat'
fault = Fault(faultfile)


def test_read_faults():
    """Read Fault from file
    """
    assert len(fault.x) == 99
    assert fault.x[0] == 483415.50
    assert fault.y[0] == 6684907.76
    assert fault.z[0] == 2685.06


def test_printfaults():
    """Check print
    """
    print(fault)


def test_grid():
    """Grid fault
    """
    gridded = fault.grid(xgrid=np.linspace(fault.x.min(), fault.x.max(), 21),
                         ygrid=np.linspace(fault.y.min(), fault.y.max(), 31),
                         quickplot=True)
    # clean up
    plt.close('all')

#def test_display():
#    """Display fault
#    """
#    fault.view()
#    fault.view(cmap='seismic')

    # clean up
#    plt.close('all')
