import pytest

import os
import matplotlib.pyplot as plt

from pysubsurface.objects import Trajectory

trajfile = 'testdata/Well/Trajectories/Vertical.csv'
traj = Trajectory(trajfile, wellname='Vert')


def test_readtraj():
    """Read Trajectory from file and check it is read into a pd.DataFrame
    """
    # check lenght
    assert traj.df.shape[0] == 81

    # check first and last element
    assert traj.df.iloc[0]['Z (meters)'] == 0.
    assert traj.df.iloc[0]['TVDSS'] == -23.0
    assert traj.df.iloc[-1]['Z (meters)'] == 3234.3718
    assert traj.df.iloc[-1]['TVDSS'] == 3211.3718


def test_print():
    """Print trajectory
    """
    print(traj)


def test_viewtraj():
    """Display and plot trajectory
    """
    traj.display()

    traj.view_traj()
    traj.view_traj(color='r', shift=(1, 4), grid=True, axiskm=True)

    traj.view_mdtvdss()
    traj.view_mdtvdss(color='r', labels='False', title='test')

    traj.view()
    traj.view(savefig='testfigs/traj_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/traj_test.png')

