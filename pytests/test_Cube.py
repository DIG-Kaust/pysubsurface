import pytest
import numpy as np
import matplotlib.pyplot as plt
from pysubsurface.objects import Cube

par1 = {'nt': 7, 'nx': 201, 'ny': 101}
par2 = {'nt': 7, 'nx': 3,   'ny': 101}


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube(par):
    """Create Cube and check axis and data values
    """
    d = Cube(np.random.normal(0, 1, (par['ny'], par['nx'], par['nt'])))

    # check axis initial and end values
    assert d.nt == par['nt']
    assert d.nx == par['nx']
    assert d.ny == par['ny']

    assert d.x[0] == -d.x[-1]
    assert d.x[int((d.nx-1)/2)] == 0
    assert d.y[0] == -d.y[-1]
    assert d.y[int((d.ny-1)/2)] == 0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube_copy(par):
    """Create Cube and verify copy routine
    """
    d = Cube(np.random.normal(0, 1, (par['ny'], par['nx'], par['nt'])))

    # check data is the same in the copied Cube
    d1 = d.copy()
    np.testing.assert_array_equal(d.data, d1.data)

    # check data is all zeros
    d2 = d.copy(empty=True)

    assert d2.data.sum() == 0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube_print(par):
    """Create Cube and print it
    """
    d = Cube(np.random.normal(0, 1, (par['ny'], par['nx'], par['nt'])))
    print(d)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube_view(par):
    """Create Cube and visualize it
    """
    d = Cube(np.random.normal(0, 1, (par['ny'], par['nx'], par['nt'])))

    d.view()
    plt.close('all')
