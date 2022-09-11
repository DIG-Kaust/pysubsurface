import pytest

import os
import matplotlib.pyplot as plt

from pysubsurface.objects import Picks

picksfile = 'testdata/Well/Picks/PICKS.md'


def test_readpick():
    """Read Picks from file
    """
    picks = Picks(picksfile)

    # check lenght
    assert picks.df.shape[0] == 7

    # check first and last picks
    assert picks.df.iloc[0]['Name'] == 'Seabed'
    assert picks.df.iloc[0]['Well UWI'] == 'Vertical'
    assert picks.df.iloc[-1]['Name'] == 'South_Base'
    assert picks.df.iloc[-1]['Well UWI'] == 'W5'


def test_addpick():
    """Add Pick manually
    """
    pick = Picks()
    pick.add_pick('Seabed', 'NO1', 25.2, interp='TEST', color='r')

    # check pick
    assert pick.df.iloc[0]['Name'] == 'Seabed'
    assert pick.df.iloc[0]['Well UWI'] == 'NO1'
    assert pick.df.iloc[0]['Depth (meters)'] == 25.2
    assert pick.df.iloc[0]['Interp'] == 'TEST'
    assert pick.df.iloc[0]['Color'] == 'r'


def test_select_interpreters():
    """Select interpreters
    """
    # select one interpreter
    picks = Picks(picksfile)
    picks.select_interpreters('STAT')

    assert picks.df.shape[0] == 5

    # select two interpreter
    picks = Picks(picksfile)
    picks.select_interpreters(('STAT', 'US1'))

    assert picks.df.shape[0] == 6


def test_discard_on_keywords():
    """Discard rows whose name contains any of the keywords
    """
    # discard one keyword
    picks = Picks(picksfile)
    picks.discard_on_keywords('Top')

    assert picks.df.shape[0] == 6

    # discard two keywords
    picks.discard_on_keywords(('Nord', 'Base'))

    assert picks.df.shape[0] == 2


def test_assign_color():
    """Assign color to special pick
    """
    picks = Picks(picksfile)
    picks.assign_color('Seabed', 'r')
    picks_color = list(picks.df[picks.df['Name']=='Seabed']['Color'])

    assert picks_color == ['r']*len(picks_color)


def test_extract_from_well():
    """Extract picks from single well
    """
    picks = Picks(picksfile)
    picks_Vertical = picks.extract_from_well('Vertical')
    picks_W5 = picks.extract_from_well('W5')

    assert picks_Vertical.df.shape[0] == 3
    assert picks_W5.df.shape[0] == 1


def test_display():
    """Check that prints work
    """
    picks = Picks(picksfile)
    picks.display()
    picks.display(nrows=None)
    picks.display(filtname='Seabe')
    picks.display(nrows=5)

    picks.count_picks()
    picks.count_picks(nrows=10)
    picks.count_picks(nrows=10, plotflag=True,
                      savefig='testfigs/pickscount_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/pickscount_test.png')
