import pytest

import os
import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.objects import Logs
from pysubsurface.objects import Trajectory


logsfile = 'testdata/Well/Logs/Vertical.las'
trajfile = 'testdata/Well/Trajectories/Vertical.csv'
logs = Logs(logsfile)
traj = Trajectory(trajfile, wellname='Vert')


def test_createlogs():
    """Create well and check basic info
    """
    # check inputs
    assert len(logs.logs.keys()) == 16


def test_dataframe():
    """Bring logs into dataframe
    """
    logs.dataframe()
    # check dataframe
    assert logs.df.shape == (49437, 15)


def test_addcurve():
    """Add new log
    """
    logscopy = logs.copy()
    curve = np.random.normal(0, 1, logs.df.shape[0])
    logscopy.add_curve(curve, 'test', 'm', 'Test curve', None)
    logscopy.dataframe()

    # check new curve
    np.testing.assert_array_equal(logscopy.logs['test'], curve)
    assert logscopy.logs.curves['test']['mnemonic'] == 'test'
    assert logscopy.logs.curves['test']['unit'] == 'm'
    assert logscopy.logs.curves['test']['descr'] == 'Test curve'
    assert logscopy.df.shape == (49437, 16)


def test_add_tvdss():
    """Add TVDSS curve
    """
    logscopy = logs.copy()
    logscopy.add_tvdss(traj)

    # check new curve
    assert logscopy.logs.curves['TVDSS']['mnemonic'] == 'TVDSS'
    assert logscopy.logs.curves['TVDSS']['unit'] == 'm'
    assert logscopy.logs.curves['TVDSS']['descr'] == 'TVDSS'
    assert logscopy.df.shape == (49437, 16)


def test_delete_curve():
    """Delete curve
    """
    logscopy = logs.copy()
    logscopy.delete_curve('LFP_RT')
    logscopy.dataframe()

    # check curve is deleted
    assert len(logscopy.logs.keys()) == 15
    assert 'LFP_RT' not in logscopy.logs.keys()
    assert logscopy.df.shape == (49437, 14)


def test_print():
    """Print log info
    """
    print(logs)


def test_display():
    """Display log info
    """
    logs.display()
    logs.describe()


def test_visualize_logcurve():
    """Visualize one log curve
    """
    logs.visualize_logcurve('LFP_VS')
    logs.visualize_logcurve('LFP_VS', color='r', thresh=2400, lw=4, grid=False,
                            xlabelpos=2)
    logs.visualize_logcurve('LFP_VS', color='r',
                            savefig='testfigs/logcurve_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/logcurve_test.png')


def test_visualize_logcurves():
    """Visualize one log curve
    """
    logs.visualize_logcurves(
        dict(VP=dict(logs=['LFP_VP'], colors=['k'], xlim=(2000, 5000)),
             VS=dict(logs=['LFP_VS'], colors=['k'], xlim=(1000, 4000))))
    logs.visualize_logcurves(dict(
        Volume=dict(logs=['LFP_VSH', 'LFP_VCARB', 'LFP_COAL'],
                    colors=['green', '#94b8b8', '#4d4d4d', 'yellow'],
                    xlim=(0, 1)),
        Sat=dict(logs=['LFP_SGT', 'LFP_SOT'],
                 colors=[ 'red', 'green', 'blue'],
                 envelope='LFP_PHIT',
                 xlim=(0, 0.4))))

    logs.visualize_logcurves(dict(
        Volume=dict(logs=['LFP_VSH', 'LFP_VCARB', 'LFP_COAL'],
                    colors=['green', '#94b8b8', '#4d4d4d', 'yellow'],
                    xlim=(0, 1)),
        Sat=dict(logs=['LFP_SGT', 'LFP_SOT'],
                 colors=['red', 'green', 'blue'],
                 envelope='LFP_PHIT',
                 xlim=(0, 0.4)),
        VP=dict(logs=['LFP_VP'], colors=['k'], xlim=(2000, 5000)),
        VS=dict(logs=['LFP_VS'], colors=['k'], xlim=(1000, 4000))),
    savefig='testfigs/logcurves_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/logcurves_test.png')


def test_visualize_histogram():
    """Visualize histogram
    """
    logs.visualize_histogram('LFP_VS')
    logs.visualize_histogram('LFP_VS', color='r', thresh=2400, grid=False)
    logs.visualize_histogram('LFP_VS', color='r',
                             savefig='testfigs/loghist_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/loghist_test.png')


def test_visualize_crossplot():
    """Visualize crossplot
    """
    logs.visualize_crossplot('LFP_VS', 'LFP_VP', 'LFP_RHOB')
    logs.visualize_crossplot('LFP_VS', 'LFP_VP', 'LFP_RHOB',
                             thresh1=2000, thresh2=3000, threshcolor=5000,
                             cmap='seismic', grid=False)
    logs.visualize_crossplot('LFP_VS','LFP_VP', 'LFP_RHOB',
                             thresh1=2000, thresh2=3000, threshcolor=5000,
                             cmap='seismic', grid=False, title='Test',
                             savefig='testfigs/logscatt_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/logscatt_test.png')
