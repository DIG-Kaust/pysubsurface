import pytest

import os
import matplotlib.pyplot as plt

from pysubsurface.objects import Polygon

polyfile = 'testdata/Polygon/TestPoly.data'
poly = Polygon(polyfile, polygonname='Test')


def test_readpoly():
    """Read Polygon from file
    """
    # check name
    assert poly.polygonname == 'Test'

    # check lenght
    assert poly.df.shape[0] == 75

    # check first and last picks
    assert poly.df.iloc[0]['X Absolute'] == 483044.06
    assert poly.df.iloc[0]['Y Absolute'] == 6687431.00
    assert poly.df.iloc[-1]['X Absolute'] == 483044.06
    assert poly.df.iloc[-1]['Y Absolute'] == 6687431.00

def test_display():
    """Check that print and view work
    """
    poly.display()
    poly.display(nrows=None)

    fig, ax = plt.subplots(1, 1)
    poly.view(ax=ax)
    poly.view(ax=ax, polyname=False, flipaxis=True, bbox=True)

    # clean up
    plt.close('all')
