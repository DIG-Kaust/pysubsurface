import pytest
import numpy as np
import matplotlib.pyplot as plt
from pysubsurface.objects import Slice

par1 = {'nt': 7, 'nx': 201}
par2 = {'nt': 7, 'nx': 3}


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_slice(par):
    """Create Slice and check axis and data values
    """
    d = Slice(np.random.normal(0,1,(par['nx'],par['nt'])))

    # check axis initial and end values
    assert d.nt == par['nt']
    assert d.nx == par['nx']

    assert d.x[0] == -d.x[-1]
    assert d.x[int((d.nx - 1) / 2)] == 0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_slice_copy(par):
    """Create Slice and check axis and data values
    """
    d = Slice(np.random.normal(0, 1, (par['nx'], par['nt'])))

    # check data is the same in the copied Slice
    d1 = d.copy()
    np.testing.assert_array_equal(d.data, d1.data)

    # check data is all zeros
    d2 = d.copy(empty=True)
    assert d2.data.sum() == 0


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube_print(par):
    """Create Slice and print it
    """
    d = Slice(np.random.normal(0, 1, (par['nx'], par['nt'])))
    print(d)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cube_view(par):
    """Create Slice and visualize it
    """
    d = Slice(np.random.normal(0, 1, (par['nx'], par['nt'])))

    d.view()
    plt.close('all')
