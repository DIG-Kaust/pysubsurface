import pytest
import numpy as np

import pysubsurface.utils.utils as utils


def test_findclosest():
    # find closest element in 1d array
    array = np.arange(10)

    # value inside
    value = 3
    assert utils.findclosest(array,value) == 3

    # value outside
    value = -10
    assert utils.findclosest(array,value) == 0

    value = 20
    assert utils.findclosest(array,value) == 9


def test_findclosest_2d():
    # find closest element in list of two 1d arrays
    array1 = np.arange(10)
    array2 = np.arange(20)

    # values inside
    value1 = 3
    value2 = 4
    assert utils.findclosest_2d([array1,array2],[value1,value2]) == [3,4]

    # one value inside, one value outside
    value1 = 3
    value2 = 30
    assert utils.findclosest_2d([array1,array2],[value1,value2]) == [3,19]

     # both values outside
    value1 = -10
    value2 = 30
    assert utils.findclosest_2d([array1,array2],[value1,value2]) == [0,19]


def test_findvalid():
    # find valid elements in 1d arrays
    n = 10
    istart, iend = 3, 7

    # array without nans
    array = np.arange(10, dtype=np.float)
    assert utils.findvalid(array) == (0, n - 1)

    # array with nans
    array[:istart] = np.nan
    array[iend+1:] = np.nan
    assert utils.findvalid(array) == (istart, iend)
