import pytest
import numpy as np

from pysubsurface.utils.signals import moving_average

def test_moving_average_constant():
    # check that moving average is transparent to constant signal
    input = np.ones(10)
    output = moving_average(input, 5)
    np.testing.assert_array_equal(input, output)
