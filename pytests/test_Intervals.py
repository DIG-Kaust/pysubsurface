import pytest

import os
import matplotlib.pyplot as plt
from pysubsurface.objects import Intervals

interval = Intervals()
interval.add_interval('World', 'Ottawa', 'Sidney', 0, 'black', order=0, field='Imaginary')
interval.add_interval('America', 'Ottawa', 'Bejing', 1, 'red', order=0, parent='World', field='Imaginary')
interval.add_interval('Asia', 'Bejing', 'KL', 1, 'green', order=1, parent='World', field='Imaginary')
interval.add_interval('Oceania', 'KL', 'Sidney', 1, 'blue', order=2, parent='World', field='Imaginary')
interval.add_interval('Canada', 'Ottawa', 'NY', 2, 'red', order=0, parent='America', field='Imaginary')
interval.add_interval('USA', 'NY', 'Bejing', 2, 'green', order=1, parent='America', field='Imaginary')
interval.add_interval('Cina', 'Bejing', 'HK', 2, 'blue', order=0, parent='Asia', field='Imaginary')
interval.add_interval('Malaysia', 'HK', 'KL', 2, 'blue', order=1, parent='Asia', field='Imaginary')
interval.add_interval('NewZealand', 'KL', 'Auckland', 2, 'yellow',
                      order=0, parent='Oceania', field='Imaginary')


def test_create_interval():
    """Check interval dataframe
    """
    # check interval size
    assert interval.df.shape[0] == 9

    # check values in interval dataframe
    assert interval.df.iloc[0]['Name'] == 'World'
    assert interval.df.iloc[0]['Parent'] is None
    assert interval.df.iloc[-1]['Name'] == 'NewZealand'
    assert interval.df.iloc[-1]['Parent'] == 'Oceania'


def test_add_interval():
    """Print interval info
    """
    interval.add_interval('Australia', 'Auckland', 'Sidney', 2, 'white',
                          parent='Oceania', field='Imaginary')

    assert interval.df.shape[0] == 10


def test_printinterval():
    """Print interval info
    """
    print(interval)

def test_view():
    """Print well info
    """
    interval.view()
    interval.view(alpha=0.3,
                  savefig='testfigs/intervals_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/intervals_test.png')