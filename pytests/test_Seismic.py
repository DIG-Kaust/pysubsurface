import pytest

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.testing import assert_equal, assert_almost_equal
from pysubsurface.objects import Well
from pysubsurface.objects import Surface
from pysubsurface.objects import Seismic


segyfile = 'testdata/Seismic3d/poststack.sgy'
segyfile_aroundwell = 'testdata/Seismic3d/poststack_aroundwell.sgy'

par = {'ot': 1000, 'dt': 0.004, 'nt': 76,
       'ox': 300,  'dx': 1,     'nx': 101,
       'oy': 300,  'dy': 1,     'ny': 101}


@pytest.mark.parametrize("par", [(par)])
def test_seismic_dims(par):
    """Load Seismic and check axis
    """
    # load seismic
    d = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # check axis lenght
    assert d.ntz == par['nt']
    assert d.nxl == par['nx']
    assert d.nil == par['ny']


def test_seismic_copy():
    """Copy object
    """
    # load seismic
    d = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # copy seismic
    d1 = d.copy()
    # copy seismic
    d1empty = d.copy(empty=True)

    assert d.data.shape == d1.data.shape
    assert d.data.shape == d1empty.data.shape

    assert np.array_equal(d.data, d1.data)
    assert np.array_equal(np.zeros_like(d.data), d1empty.data)


@pytest.mark.parametrize("par", [(par)])
def test_seismic_print(par):
    """Load Seismic and print it
    """
    # load seismic
    d = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)
    print(d)


@pytest.mark.parametrize("par", [(par)])
def test_seismic_info(par):
    """Load Seismic and print info
    """
    # load seismic
    d = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # print info
    d.info(level=0)
    d.info(level=1)
    d.info(level=2)


def test_seismic_headers():
    """Load Seismic and check header values
    """
    # load seismic
    d = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # extract headers
    head1 = d.read_headervalues(0, headerwords=21)
    head2 = d.read_headervalues(0, headerwords=25)
    head12 = d.read_headervalues(1, headerwords=(21, 25))

    # check axis lenght
    assert head1 == 300
    assert head2 == 300
    assert head12[21] == 300
    assert head12[25] == 301


@pytest.mark.parametrize("par", [(par)])
def test_seismic_subcube(par):
    """Extract subcube and verify dimensions
    """
    # load seismic
    d = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)
    subcube = d.read_subcube(ilinein=30, ilineend=40)

    # check axis lenght
    assert subcube.shape[0] == 10
    assert subcube.shape[1] == par['nx']
    assert subcube.shape[2] == par['nt']


@pytest.mark.parametrize("par", [(par)])
def test_seismic_inline(par):
    """Extract IL slice and verify dimensions
    """
    # load seismic
    d = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)
    slice = d.read_inline(iline=30)

    # check axis lenght
    assert slice.shape[0] == par['nx']
    assert slice.shape[1] == par['nt']
    np.testing.assert_array_equal(slice, d.data[30])


@pytest.mark.parametrize("par", [(par)])
def test_seismic_crossline(par):
    """Extract XL slice and verify dimensions
    """
    # load seismic
    d = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)
    slice = d.read_crossline(xline=30)

    # check axis lenght
    assert slice.shape[0] == par['ny']
    assert slice.shape[1] == par['nt']
    np.testing.assert_array_equal(slice, d.data[:, 30])


@pytest.mark.parametrize("par", [(par)])
def test_seismic_trace(par):
    """Extract trace and verify dimensions
    """
    # load seismic
    d = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)
    slice = d.read_trace(10)

    # check axis lenght
    assert slice.size == par['nt']


def test_extract_subcube():
    """Extract subcube from seismic
    """
    # load seismic
    seismic = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)

    # extract whole cube (validate that this works fine)
    subcube = seismic.extract_subcube(tzlims=(None, None),
                                      xllims=(None, None),
                                      illims=(None, None))[0]
    # check axis lenght
    assert subcube.shape[0] == seismic.data.shape[0]
    assert subcube.shape[1] == seismic.data.shape[1]
    assert subcube.shape[2] == seismic.data.shape[2]

    # extract small cube
    subcube = seismic.extract_subcube(tzlims=[1040, 1100],
                                      xllims=[300, 301],
                                      illims=[398, 400], plotflag=True)[0]
    plt.close('all')

    # check axis lenght
    assert subcube.shape[0] == 3
    assert subcube.shape[1] == 2
    assert subcube.shape[2] == 4


def test_extract_trace_at_well():
    """Extract trace at IL-XL of well and verify that matches that extracted
    """
    # load seismic
    seismic = Seismic(segyfile_aroundwell)

    # create well
    well = Well('testdata/', 'Vertical', rename='Vertical')

    # load trace from pspro csv
    df = pd.read_csv('testdata/Seismic3d/Vertical_seismictrace.csv',
                     index_col=0)
    trace = df['Unnamed: 1'].values[1:] # traces are shifted by one sample
    nt = trace.size

    # extract trace from seismic
    trace_extracted = seismic.extract_trace_verticalwell(well)[:nt]

    assert_almost_equal(trace, trace_extracted, decimal=3)


def test_extract_attribute_map_fixedtime():
    """Extract time slice using a surface
    """
    # load seismic
    seismic = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # extract time slice
    itz = seismic.ntz//2
    seismic_slice = seismic.data[:, :, itz]

    # create surface
    horizon = Surface(None)
    horizon.create_surface(y=np.arange(seismic.nil),
                           x=np.arange(seismic.nxl),
                           data=seismic.tz[itz] * np.ones((seismic.nil,
                                                           seismic.nxl)),
                           il=seismic.ilines, xl=seismic.xlines)

    # extract values from seismic at horizon
    horizon_map = seismic.extract_attribute_map(horizon, intwin=0)[0]

    assert_equal(seismic_slice, horizon_map.data)


def test_extract_attribute_map():
    """Extract amplitude at a given surface...
    and compare with commercial software
    """
    pass


def test_view():
    """Visualize seismic
    """
    # load seismic
    seismic = Seismic(segyfile, loadcube=False, iline=21, xline=25, scale=1)

    seismic.view()
    seismic.view(which='il')
    seismic.view(which='xl')
    seismic.view(which='tz')

    seismic.view(ilplot=1, xlplot=1, tzplot=1,
                 savefig='testfigs/seismic_test.png')

    seismic.view_cdpslice()
    seismic.view_cdpslice(tzplot=1, savefig='testfigs/seismicslice_test.png')

    seismic.view_geometry(polygon=False, savefig='testfigs/seismicgeom_test.png')

    # clean up
    plt.close('all')
    os.remove('testfigs/seismic_test.png')
    os.remove('testfigs/seismicslice_test.png')
    os.remove('testfigs/seismicgeom_test.png')
