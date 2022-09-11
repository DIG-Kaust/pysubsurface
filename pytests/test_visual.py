import pytest
import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.objects import Project, Surface, Seismic
from pysubsurface.visual.utils import _set_black, _discrete_cmap, _wiggletrace
from pysubsurface.visual.combinedviews import intervals_on_map, seismic_and_map


segyfile = 'testdata/Seismic3d/poststack.sgy'

def test_set_black():
    """Set black layout to figure
    """
    fig, ax = plt.subplots(1, 1)
    _set_black(fig, ax)
    plt.close('all')


def test_discrete_cmap():
    """Create discrete colormap
    """
    _ = _discrete_cmap(10)
    _ = _discrete_cmap(10, base_cmap='seismic')


def test_wiggle_trace():
    """Display wiggle traces
    """
    fig, ax = plt.subplots(1, 1)
    _ = _wiggletrace(ax, np.arange(100),
                     np.random.normal(0, 1, 100))
    _ = _wiggletrace(ax, np.arange(100),
                     np.random.normal(0, 1, 100)+10,
                     center=10,
                     cpos='y', cneg='c')
    plt.close('all')


def test_intervals_on_map():
    """Display properties of interval object on a map
    """
    project = Project(projectdir='testdata/', projectname='Test')
    # add intervals
    project.intervals.add_interval('All', 'Seabed', 'Nord_Base', 0, 'black')
    project.intervals.add_interval('All', 'Seabed', 'Nord_Base', 1, 'black',
                                   parent='All')
    project.intervals.add_interval('Sea', 'Seabed', 'Nord_Top', 2, 'red',
                                   parent='All')
    project.intervals.add_interval('Earth', 'Nord_Top', 'Nord_Base', 2, 'green',
                                   parent='All')
    # add wells
    project.add_wells(wellnames=['Vertical'], verb=True)

    # add surface
    surfacefile = 'testdata/Surface/dsg5_long.txt'
    surface = Surface(filename=surfacefile, format='dsg5_long',
                      loadsurface=True)

    level = 2
    _, _ = intervals_on_map(project.wells, surface, level=level, interval='Sea',
                            cmapproperty='magma_r',
                            **dict(which='yx', cmap='seismic',
                                   originlower=True,
                                   flipaxis=True, cbar=True,
                                   clim=(2200, 3600),
                                   chist=True, ncountour=5,
                                   nhist=101, figsize=(15, 17),
                                   figstyle='white',
                                   titlesize=12))
    # clean up
    plt.close('all')


def test_seismic_and_map():
    """Display seismic and surface
    """
    # load seismic
    seismic = Seismic(segyfile, loadcube=True, iline=21, xline=25, scale=1)

    # extract time slice
    itz = seismic.ntz // 2
    seismic_slice = seismic.data[:, :, itz]

    # create surface
    horizon = Surface(None)
    horizon.create_surface(y=np.arange(seismic.nil),
                           x=np.arange(seismic.nxl),
                           data=seismic.tz[itz] * np.ones((seismic.nil,
                                                           seismic.nxl)),
                           il=seismic.ilines, xl=seismic.xlines)

    # extract values from seismic at horizon
    seismic_and_map(seismic, horizon,
                    whichseismic='il', whichsurface='yx', ilplot=100,
                    title='test',
                    kargs_seismicplot=dict(cmap='gray', clip=0.4, cbar=True,
                                           horizons=horizon,
                                           horcolors='red'),
                    kargs_surfaceplot=dict(cmap='seismic', originlower=True,
                                           flipaxis=True, cbar=True,
                                           chist=True, ncountour=5,
                                           nhist=101, figstyle='white'))
    # clean up
    plt.close('all')