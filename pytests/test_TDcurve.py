import pytest

import  numpy as np
import os
import matplotlib.pyplot as plt

from pysubsurface.objects import TDcurve

tdfile = 'testdata/Well/TDCurve/Vertical.csv'
tdcurve = TDcurve(tdfile, name='Test')


def test_readtraj():
    """Read Trajectory from file and check it is read into a pd.DataFrame
    """
    # check lenght
    assert tdcurve.df.shape[0] == 31

    # check first and last element
    assert tdcurve.df.iloc[0]['Depth (meters)'] == 0.
    assert np.isnan(tdcurve.df.iloc[0]['Velocity'])
    assert tdcurve.df.iloc[-1]['Depth (meters)'] == 3221.0925
    assert tdcurve.df.iloc[-1]['Velocity'] == 3049.6414


def test_print():
    """Print TD curve
    """
    print(tdcurve)


def test_viewtraj():
    """Display and plot trajectory
    """
    tdcurve.display()

    tdcurve.view()
    tdcurve.view(title='test', savefig='testfigs/tdcurve_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/tdcurve_test.png')

