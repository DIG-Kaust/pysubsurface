import pytest

import os
import matplotlib.pyplot as plt

from pysubsurface.objects import Well
from pysubsurface.objects import Intervals
from pysubsurface.objects import Picks
from pysubsurface.objects import Seismic

picksfile = 'testdata/Well/Picks/PICKS.md'
tdfile = 'testdata/Well/TD/Vertical.csv'
seismicfile = 'testdata/Seismic3d/poststack_aroundwell.sgy'
picks = Picks(picksfile, field='Imaginary')
well = Well('testdata/', 'Vertical', rename='Vertical')

interval = Intervals()
interval.add_interval('All', 'Seabed', 'Nord_Base', 0, 'black',
                      order=0, field='Imaginary')
interval.add_interval('All', 'Seabed', 'Nord_Base', 1,
                      'black', parent='All', order=0, field='Imaginary')
interval.add_interval('Sea', 'Seabed', 'Nord_Top', 2, 'red',
                      parent='All', order=0, field='Imaginary')
interval.add_interval('Earth', 'Nord_Top', 'Nord_Base', 2, 'green',
                      parent='All', order=1, field='Imaginary')


def test_createwell():
    """Create well and check basic info
    """
    # check inputs
    assert well.wellname == 'Vertical'
    assert well.xcoord == 483503.39999999903
    assert well.ycoord == 6684143.49999999
    assert len(well.welllogs.logs.keys()) == 16
    assert len(well.tdcurves) == 0
    assert len(well.checkshots) == 0


def test_add_picks():
    """Add picks to well
    """
    well.add_picks(picks, computetvdss=True)

    assert well.picks.df.shape[0] == 3
    assert well.picks.df.iloc[0]['Well UWI'] == 'Vertical'


def test_add_tdcurve():
    """Add tdcurve to well
    """
    well.add_tdcurve('Vertical', 'Vertical')

    assert well.checkshots['Vertical'].df.shape[0] == 31
    assert well.checkshots['Vertical'].df.iloc[-1]['Velocity'] == 3049.6414


def test_create_intervals():
    """Create intervals
    """
    well.create_intervals(interval)

    # check size and values
    assert well.intervals.shape[0] == 4
    assert well.intervals.iloc[0]['Name'] == 'All'
    assert well.intervals.iloc[0]['Thickness (meters)'] == 3103.312266161689


def test_compute_picks_tvdss_twt():
    """Add TVDSS and TWT to well picks
    """
    #well.add_picks(picks, computetvdss=True)
    well.compute_picks_tvdss()
    well.compute_picks_twt(checkshot_name='Vertical')

    # check that new columns have been made in picks dataframe
    assert well.picks.df['TVDSS (meters)'].shape[0] == 3
    assert well.picks.df['TWT - Vertical (ms)'].shape[0] == 3


def test_compute_logs_tvdss_twt():
    """Add TVDSS and TWT to well logs
    """
    well.compute_logs_tvdss()
    well.compute_logs_twt(checkshot_name='Vertical')

    # check that new columns have been made in picks dataframe
    assert well.welllogs.df['TVDSS'].shape[0] == 49437
    assert well.welllogs.df['TWT - Vertical'].shape[0] == 49437


def test_add_intervals_twt():
    """Add TWT to intervals
    """
    well.add_intervals_twt('TWT - Vertical (ms)')

    # check that new columns have been made in picks dataframe
    assert well.intervals['Top TWT - Vertical (ms)'].shape[0] == 4
    assert well.intervals['Base TWT - Vertical (ms)'].shape[0] == 4


def test_extract_logs_in_interval():
    """Extract logs in specific interval containing entire log lenght
    """
    extrprop = well.extract_logs_in_interval(well.intervals.iloc[0], 'LFP_VP')
    print(well.intervals.iloc[0], extrprop, well.welllogs.df)
    assert extrprop.shape[0] == well.welllogs.df.shape[0]


def test_create_averageprops_intervals():
    """Create average properties in interval
    """
    avprops = well.create_averageprops_intervals(level=0)

    # check values
    assert avprops['LFP_VP']['All']['mean'] == 3519.4087875526275
    assert avprops['LFP_VP']['All']['stdev'] == 490.3812042405878


def test_print():
    """Print well info
    """
    print(well)


def test_display():
    """Display well info
    """
    well.display()


def test_view_picks_and_intervals():
    """Display picks and intervals
    """
    interval = Intervals()
    interval.add_interval('All', 'Seabed', 'Nord_Base', 0, 'black')
    interval.add_interval('All', 'Seabed', 'Nord_Base', 1, 'black',
                          parent='All')
    interval.add_interval('Sea', 'Seabed', 'Nord_Top', 2, 'red', parent='All')
    interval.add_interval('Earth', 'Nord_Top', 'Nord_Base', 2, 'green',
                          parent='All')

    well.add_picks(picks, computetvdss=True)
    well.create_intervals(interval)

    fig, axs = plt.subplots(1, 2)
    well.view_picks_and_intervals(axs)

    # clean up
    plt.close('all')


def test_view_logtrack():
    """Display log track
    """
    well.view_logtrack('petro', lfp=True, vcoal='COAL', figsize=(25,17))
    well.view_logtrack('rock', lfp=True, vcoal='COAL', figsize=(25,17))

    # clean up
    plt.close('all')


def test_view_in_seismicsection():
    """Display well in seismicsection
    """
    seismic = Seismic(seismicfile)

    well.view_in_seismicsection(seismic, domain='depth', which='il')
    well.view_in_seismicsection(seismic, domain='depth', which='xl')

    # clean up
    plt.close('all')


def test_view_logprops_intervals():
    """Display log properties in intervals
    """
    # histogram
    well.view_logprops_intervals(0, prop1name='LFP_VP')
    well.view_logprops_intervals(1, prop1name='LFP_VP')
    well.view_logprops_intervals(2, prop1name='LFP_VP')

    fig, ax = plt.subplots(1, 1)
    well.view_logprops_intervals(2, ax=ax, prop1name='LFP_VP')

    # scatterplot
    well.view_logprops_intervals(0, prop1name='LFP_VP', prop2name='LFP_VS')
    well.view_logprops_intervals(1, prop1name='LFP_VP', prop2name='LFP_VS')
    well.view_logprops_intervals(2, prop1name='LFP_VP', prop2name='LFP_VS')

    fig, ax = plt.subplots(1, 1)
    well.view_logprops_intervals(2, ax=ax, prop1name='LFP_VP',
                                 prop2name='LFP_VS')

    # scatterplot (draw samples)
    well.view_logprops_intervals(2, vpname='LFP_VP', vsname='LFP_VS',
                                 rhoname='LFP_RHOB', prop1name='LFP_VP',
                                 prop2name='LFP_VS')

    fig, ax = plt.subplots(1, 1)
    well.view_logprops_intervals(2, ax=ax,vpname='LFP_VP', vsname='LFP_VS',
                                 rhoname='LFP_RHOB', prop1name='LFP_VP',
                                 prop2name='LFP_VS')

    # clean up
    plt.close('all')
