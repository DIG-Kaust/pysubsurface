import pytest

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.testing import assert_array_equal
from pysubsurface.objects import Interpretation

surfacefiles = ['testdata/Surface/dsg5_long.txt',
                'testdata/Surface/dsg5_long1.txt']
interpretation = Interpretation(surfacefiles)


def test_create_interpretation():
    """Check interpretation object
    """
    # check interval size
    print(len(interpretation.surfaces))
    assert len(interpretation.surfaces) == 2


def test_print():
    """Print interpretation object
    """
    print(interpretation)


def test_copy():
    """Copy interpretation object
    """
    intcopy = interpretation.copy()
    intempty = interpretation.copy(empty=True)

    for i in range(len(surfacefiles)):
        assert_array_equal(interpretation.surfaces[i].data,
                           intcopy.surfaces[i].data)
        assert np.sum(np.abs(intempty.surfaces[i].data)) == 0.


def test_printinterval():
    """Print interval info
    """
    print(interpretation)


def test_view():
    """Print well info
    """
    interpretation.view()
    interpretation.view(ilplot=0, xlplot=0)

    # clean up
    plt.close('all')
