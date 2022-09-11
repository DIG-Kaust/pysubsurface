import logging

import os
import copy
from shutil import copyfile
import warnings

import numpy as np
import numpy.ma as np_ma
import matplotlib.pyplot as plt
import segyio
import pysubsurface

from mpl_toolkits.axes_grid1 import make_axes_locatable
from segyio.tracefield import TraceField as stf
from pysubsurface.utils.utils import findclosest

from pysubsurface.objects.utils import _a2e, _e2a
from pysubsurface.objects.utils import _findclosest_point, _findclosest_well_seismicsections
from pysubsurface.objects.Cube import Cube
from pysubsurface.objects.Surface import Surface
from pysubsurface.proc.seismicmod.waveletest import statistical_wavelet
from pysubsurface.visual.utils import plot_polygon


def _segyinfo(filename, iline=189, xline=193, level=2):
    """Segy info.

    Display basic information about binary headers in segy file.

    Parameters
    ----------
    filename : :obj:`str`
        File name
    iline : :obj:`int`, optional
        Byte location of inline
    xline : :obj:`int`, optional
        Byte location of crossline
    level : :obj:`bool`, optional
        Level of details (``0``:none - only geometry,
        ``1``: most important header words, ``2``: all header words)

    """
    with segyio.open(filename, 'r', iline=iline, xline=xline) as segyfile:
        segyfile.mmap()

        # define headers to display
        if level == 0:
            heads = []
        elif level == 1:
            heads = [stf.TRACE_SEQUENCE_LINE,
                     stf.TRACE_SEQUENCE_FILE,
                     stf.CDP_X, stf.CDP_Y]
        else:
            heads = segyfile.header[0].keys()

        # extract dimensions and axes
        t = segyfile.samples
        dt = t[1]-t[0]
        dil = segyfile.ilines[1]-segyfile.ilines[0]
        dxl = segyfile.xlines[1]-segyfile.xlines[0]

        # display info
        print('File: {}\n'.format(filename))
        print('Geometry:')
        print('IL:\t{} - {} - {}\t( NIL:{})'.format(segyfile.ilines[0],
                                                    dil,
                                                    segyfile.ilines[-1],
                                                    len(segyfile.ilines)))
        print('XL:\t{} - {} - {}\t( NXL:{})'.format(segyfile.xlines[0],
                                                    dxl,
                                                    segyfile.xlines[-1],
                                                    len(segyfile.xlines)))
        print('T/Z:\t{} - {} - {}\t( NTZ:{})'.format(t[0],
                                                     dt,
                                                     t[-1],
                                                     len(segyfile.samples)))

        if level > 0:
            print('\nList of headerwords with min and max values:')
            for head in heads:
                tmp = segyfile.attributes(segyio.TraceField(head))[:]
                print('{0: <45}: {1} - {2}'.format(str(segyio.TraceField(head)),
                                                   np.min(tmp), np.max(tmp)))


def _segyheaderword_from_index(segybtye):
    """Return segy headerword given its starting byte

    Parameters
    ----------
    segybtye : :obj:`int`
        Starting byte of a trace header field

    Returns
    -------
    segyheader : :obj:`str`
        Header name

    """
    segyheader = [key for key, index in segyio.tracefield.keys.items()
                  if index == segybtye][0]
    return segyheader


def _write_seismic(segyfile_in, segyfile_out, data, iline=189, xline=193):
    """Write seismic to segy file.

    Take a segy file as template and save a new data stored in ``np.ndarray``
    object into a new segy file.

    Parameters
    ----------
    segyfile_in : :obj:`str`
        Name of input file
    segyfile_out : :obj:`str`
        Name of output file
    data : :obj:`str`
        Data to write in new file (note that it must have the same size of
        content of ``segyfile_in`` file)
    iline : :obj:`int`, optional
        Byte location of inline
    xline : :obj:`int`, optional
        Byte location of crossline

    """
    copyfile(segyfile_in, segyfile_out)
    with segyio.open(segyfile_out, "r+", iline=iline, xline=xline) as dst:
        if dst.sorting == 1:
            for xl, xline in enumerate(dst.xlines):
                dst.xline[xline] = data[xl].astype('float32')
        else:
            for il, iline in enumerate(dst.ilines):
                dst.iline[iline] = data[il].astype('float32')


def _extract_lims(axis, lims, name=''):
    """Extract indeces of axis given limits.

    Find index of closest start and end values of axis given
    two arbitrary limits (``lims``)

    Parameters
    ----------
    axis : :obj:`np.ndarray`
        Axis to scan
    lims : :obj:`tuple`
        Start and end value to be used to select a subaxis, use ``None``
        if interested in the first and last values. Note that, if any of
        ``lims`` does not perfectly match a value in the axis, the closest
        value in the axis is selected
    name : :obj:`str`, optional
        Name of axis (used only to raise meaningful warnings)

    Returns
    -------
    lims : :obj:`tuple`
        Updated start and end value belonging to axis
    ilims : :obj:`tuple`
        Indeces of the updated start and end value belonging to axis

    """
    # initialize ilims
    lims = list(lims)
    ilims = [0, -1]

    # check if lims[0]=-1 and set it to first element in corresponding axis
    lims[0] = axis[0] if lims[0] is None else lims[0]
    # check if lims[1]=-1 and set it to last element in corresponding axis
    lims[1] = axis[-1] if lims[1] is None else lims[1]

    # check that lims[0] is smaller than lims[1]
    if lims[0] > lims[1]:
        raise ValueError('lims[0]={} should be smaller '
                         'than lims[1]={} for {} axis'.format(lims[0],
                                                              lims[1],
                                                              name))

    # find closest start and end points in axis
    ilims[0] = findclosest(axis, lims[0])
    start = axis[ilims[0]]
    if start != lims[0]:
        lims[0] = start
        warnings.warn('start value has been re-set to {} '
                      'for {} axis'.format(lims[0], name))

    ilims[1] = findclosest(axis, lims[1])
    end = axis[ilims[1]]
    if end != lims[1]:
        lims[1] = end
        warnings.warn('end value has been re-set to {} '
                      'for {} axis'.format(lims[1], name))

    ilims[1] = ilims[1]+1  # include also right extreme

    return tuple(lims), tuple(ilims)


def _attribute_averaging(cube, yscan, xscan, tscan, intwin_above=0,
                         intwin_below=0, inttype='mean'):
    """Attribute extraction from data over window of size intwin
    in t/z axis for line defined by xscan and tscan

    Parameters
    ----------
    cube : :obj:`pysubsurface.objects.Cube`
        Data for extraction
    yscan : :obj:`np.ndarray`
        y indices of line to be stacked
    xscan : :obj:`np.ndarray`
        x indices of line to be stacked
    tscan : :obj:`np.ndarray`
        t/z indices of line to be stacked
    intwin_above : :obj:`int`, optional
        Number of samples to be used in window above surface
    intwin_below : :obj:`int`, optional
        Number of samples to be used in window below surface
    inttype : :obj:`str`, optional
        Type of averaging to be performed in the window (``mean``, ``rms``,
        ``max``, ``min``, ``maxabs``)

    Returns
    -------
    attribute : :obj:`pysubsurface.objects.Surface`
        Attribute

    """
    if intwin_above < 0:
        warnings.warn('intwin_above={}, '
                      'setting it to 0...'.format(intwin_above))
        intwin_above = 0

    if intwin_below < 0:
        warnings.warn('intwin_below={}, '
                      'setting it to 0...'.format(intwin_below))
        intwin_below = 1

    intwin = intwin_above + intwin_below + 1

    ny, nx, nt = cube.ny, cube.nx, cube.nt
    nscan = len(yscan)

    # augument attribute by one value at the end to use when window goes outside allowed values
    if inttype=='mean':
        cubeaug = np.concatenate((np.moveaxis(cube.data, -1, 0),
                                  np.zeros((1, ny, nx))), axis=0)
    elif inttype=='rms':
        cubeaug = np.concatenate((np.moveaxis(cube.data**2, -1, 0),
                                  np.zeros((1, ny, nx))), axis=0)
    elif inttype=='max' or inttype=='maxabs':
        cubeaug = np.concatenate((np.moveaxis(cube.data, -1, 0),
                                  -1e10*np.ones((1, ny, nx))), axis=0)
    elif inttype=='min':
        cubeaug = np.concatenate((np.moveaxis(cube.data, -1, 0),
                                  1e10*np.ones((1, ny, nx))), axis=0)

    # define integration window
    tscans = np.array([tscan + it for it in range(-intwin_above, intwin_below + 1)])
    if intwin > 1:
        xscan = np.tile(xscan, intwin)
        yscan = np.tile(yscan, intwin)

    inside_data = (tscans <= nt) & (tscans >= 0)
    tscans[~inside_data] = nt
    nels_per_tz = inside_data.sum(axis=0)

    # create attributes
    if inttype == 'mean':
        attribute = np.sum(cubeaug[tscans.ravel(), yscan, xscan].reshape([intwin, nscan]), axis=0)/nels_per_tz
    elif inttype == 'rms':
        attribute = np.sqrt(np.sum(cubeaug[tscans.ravel(), yscan, xscan].reshape([intwin, nscan]), axis=0)/nels_per_tz)
    elif inttype=='max':
        attribute = np.max(cubeaug[tscans.ravel(), yscan, xscan].reshape([intwin, nscan]), axis=0)
    elif inttype=='min':
        attribute = np.min(cubeaug[tscans.ravel(), yscan, xscan].reshape([intwin, nscan]), axis=0)
    elif inttype == 'maxabs':
        attribute = np.max(np.abs(cubeaug[tscans.ravel(), yscan, xscan]).reshape([intwin, nscan]), axis=0)

    return attribute


def _extract_attribute_map(seismic, hor, intwin=None,
                           intwin_above=0, intwin_below=0,
                           inttype='mean', scale=1., verb=True):
    """Attribute map from seimic.

    Extract amplitude map from seismic cube given an horizon applying an
    operation (mean, rms, max, min, maxabs) over window of size ``intwin`` i
    n t/z axis. If seismic and horizon do not belong to the same grid (in inline
    and crossline), they are first brought to a common grid and then the
    amplitude extraction is performed in such a grid.

    Parameters
    ----------
    seismic : :obj:`Seismic` or :obj:`SeismicIrregular`
        Seismic
    hor : :obj:`Surface`
        Horizon (in time or depth) along which extraction is performed
    intwin : :obj:`int`, optional
        Number of samples to be used on either side of surface in window
        (if ``None`` use intwin_above and intwin_below)
    intwin_above : :obj:`int`, optional
        Number of samples to be used in window above surface
        (used when intwin is ``None``)
    intwin_below : :obj:`int`, optional
        Number of samples to be used in window below surface
        (used when intwin is ``None``)
    inttype : :obj:`str`, optional
        Type of averaging to be performed in the window (``mean``, ``rms``,
        ``max``, ``min``, ``maxabs``)
    scale : :obj:`float`, optional
        Scaling to be applied to time/depth values of ``hor`` prior to amplitude
        extraction

    Returns
    -------
    attrmap_in_common : :obj:`pysubsurface.objects.Surface`
        Attribute map in common grid
    hor_in_common : :obj:`pysubsurface.objects.Surface`
        Horizon in common grid

    TODO: routine works only with IL and XL so far, will return a Surface with None x and y....
    """
    # find out window
    if intwin is not None:
        intwin_above = intwin_below = intwin

    # find common grid between seismic and horizon
    il_seis, xl_seis = seismic.ilines, seismic.xlines
    il_hor, xl_hor = hor.il, hor.xl

    il_common = [il for il in il_hor if il in il_seis]
    xl_common = [xl for xl in xl_hor if xl in xl_seis]

    nil_common, nxl_common = len(il_common), len(xl_common)

    iil_seis = [np.argwhere(il == il_seis)[0][0] for il in il_common]
    ixl_seis = [np.argwhere(xl == xl_seis)[0][0] for xl in xl_common]
    iil_hor = [np.argwhere(il == il_hor)[0][0] for il in il_common]
    ixl_hor = [np.argwhere(xl == xl_hor)[0][0] for xl in xl_common]

    # extract seismic and horizon in locations of common grid
    seismic_data = seismic.data[iil_seis][:, ixl_seis]
    hor_data = hor.data[iil_hor][:, ixl_hor]

    # create indexes for horizon extraction
    IIL, IXL = np.meshgrid(np.arange(nil_common), np.arange(nxl_common),
                           indexing='ij')
    IIL, IXL = IIL.astype(np.int).ravel(), IXL.astype(np.int).ravel()

    itz = np.round(((scale*hor_data) - seismic.tz[0]) /
                   seismic.dtz).astype(np.int)
    itz[itz < 0] = 0

    # extract horizon
    attr = _attribute_averaging(Cube(seismic_data),
                                IIL, IXL, itz.ravel(),
                                intwin_above=intwin_above,
                                intwin_below=intwin_below,
                                inttype=inttype)
    if verb:
        print('Done computing attribute map, deleting seismic....')
    del seismic_data

    if isinstance(hor_data, np_ma.core.MaskedArray):
        attr = np_ma.masked_array(attr.reshape(nil_common, nxl_common),
                                  mask=hor_data.mask)
    else:
        attr = np.array(attr.reshape(nil_common, nxl_common))

    # put attribute and horizon in common grid in Surface object
    attrmap_in_common = hor.copy(empty=True)
    attrmap_in_common.data[:] = attr

    return attrmap_in_common, hor


def _extract_interval_map(seismic, hor_top, hor_base,
                          inttype='mean', scale=1., verb=True):
    """Attribute map from seimic.

    Extract amplitude map from seismic cube given an horizon applying an
    operation (mean, rms, max, min, maxabs) over window of size ``intwin`` i
    n t/z axis. If seismic and horizon do not belong to the same grid (in inline
    and crossline), they are first brought to a common grid and then the
    amplitude extraction is performed in such a grid.

    Parameters
    ----------
    seismic : :obj:`Seismic` or :obj:`SeismicIrregular`
        Seismic
    horizon_top : :obj:`pysubsurface.objects.Surface`
        Top horizon in time or depth
    horizon_base : :obj:`pysubsurface.objects.Surface`
        Base horizon in time or depth. It may be on a different grid than
        the top horizon but should be in the same domain. If the grid is
        different the base horizon will be first interpolated to the grid
        of the top horizon.
    inttype : :obj:`str`, optional
        Type of averaging to be performed in the window (``mean``, ``rms``,
        ``max``, ``min``, ``maxabs``)
    scale : :obj:`float`, optional
        Scaling to be applied to time/depth values of ``hor`` prior to amplitude
        extraction
    verb : :obj:`bool`, optional
        Verbosity

    Returns
    -------
    attrmap_in_common : :obj:`pysubsurface.objects.Surface`
        Attribute map in common grid
    hor_top_in_common : :obj:`pysubsurface.objects.Surface`
        Top horizon in common grid
    hor_base_in_common : :obj:`pysubsurface.objects.Surface`
        Base horizon in common grid

    TODO: routine works only with IL and XL so far, will return a Surface with None x and y....
    """
    # ensure horizons are in the same grid
    hor_base = hor_top.resample_surface_to_grid(hor_base, intmethod='linear')

    # find common grid between seismic and horizon
    il_seis, xl_seis = seismic.ilines, seismic.xlines
    il_hor, xl_hor = hor_top.il, hor_top.xl

    il_common = [il for il in il_hor if il in il_seis]
    xl_common = [xl for xl in xl_hor if xl in xl_seis]

    nil_common, nxl_common = len(il_common), len(xl_common)

    iil_seis = [np.argwhere(il == il_seis)[0][0] for il in il_common]
    ixl_seis = [np.argwhere(xl == xl_seis)[0][0] for xl in xl_common]
    iil_hor = [np.argwhere(il == il_hor)[0][0] for il in il_common]
    ixl_hor = [np.argwhere(xl == xl_hor)[0][0] for xl in xl_common]

    # extract seismic and horizon in locations of common grid
    seismic_data = seismic.data[iil_seis][:, ixl_seis]
    hor_top_data = hor_top.data[iil_hor][:, ixl_hor]
    hor_base_data = hor_base.data[iil_hor][:, ixl_hor]

    # create indexes for horizon extraction
    IIL, IXL = np.meshgrid(np.arange(nil_common), np.arange(nxl_common),
                           indexing='ij')
    IIL, IXL = IIL.astype(np.int).ravel(), IXL.astype(np.int).ravel()

    itz_top = np.round(((scale*hor_top_data) - seismic.tz[0]) /
                       seismic.dtz)
    itz_top[itz_top < 0] = 0
    itz_top[itz_top >= seismic.ntz] = seismic.ntz - 1
    itz_base = np.round(((scale * hor_base_data) - seismic.tz[0]) /
                        seismic.dtz)
    itz_base[itz_base < 0] = 0
    itz_base[itz_base >= seismic.ntz] = seismic.ntz - 1

    if isinstance(itz_top, np_ma.core.MaskedArray):
        itz_top_mask = itz_top.mask
        itz_top = itz_top.data
        itz_top[itz_top_mask] = np.nan
        itz_base_mask = itz_base.mask
        itz_base = itz_base.data
        itz_base[itz_base_mask] = np.nan

    # extract horizon
    attr = np.zeros(len(IIL))
    for i, (iil, ixl, itzt, itzb) in enumerate(zip(IIL, IXL, itz_top.ravel(), itz_base.ravel())):
        if not np.isnan(itzt) and not np.isnan(itzb):
            if inttype == 'mean':
                attr[i] = np.mean(seismic_data[iil, ixl, int(itzt):int(itzb)])
            elif inttype == 'rms':
                attr[i] = np.sqrt(np.sum(seismic_data[iil, ixl, int(itzt):int(itzb)] ** 2))
            elif inttype == 'max':
                attr[i] = np.max(seismic_data[iil, ixl, int(itzt):int(itzb)])
            elif inttype == 'min':
                attr[i] = np.min(seismic_data[iil, ixl, int(itzt):int(itzb)])
            elif inttype == 'maxabs':
                attr[i] = np.max(np.abs((seismic_data[iil, ixl, int(itzt):int(itzb)])))
            elif inttype == 'sum':
                attr[i] = np.sum(seismic_data[iil, ixl, int(itzt):int(itzb)])
            else:
                raise ValueError('{} not implemented...'.format(inttype))
    if verb:
        print('Done computing attribute map, deleting seismic....')
    del seismic_data

    attr = attr.reshape(nil_common, nxl_common)
    if isinstance(hor_top_data, np_ma.core.MaskedArray) and \
            isinstance(hor_base_data, np_ma.core.MaskedArray):
        attr = np_ma.masked_array(attr.reshape(nil_common, nxl_common),
                                  mask=hor_top_data.mask | hor_base_data.mask)
    elif isinstance(hor_top_data, np_ma.core.MaskedArray):
            attr = np_ma.masked_array(attr.reshape(nil_common, nxl_common),
                                      mask=hor_top_data.mask)
    elif isinstance(hor_base_data, np_ma.core.MaskedArray):
        attr = np_ma.masked_array(attr.reshape(nil_common, nxl_common),
                                  mask=hor_base_data.mask)
    else:
        attr = np.array(attr.reshape(nil_common, nxl_common))

    # put attribute and horizon in common grid in Surface object
    attrmap_in_common = hor_top.copy(empty=True)
    attrmap_in_common.data[:] = attr

    return attrmap_in_common, hor_top, hor_base


def _extract_arbitrary_path(ilvertices, xlvertices, dil, dxl, il0, xl0,
                            sampling=1):
    """Construct arbitrary path with sampling of provided inlines
    and crosslines

    Parameters
    ----------
    ilvertices : :obj:`tuple` or :obj:`list`
        Vertices of arbitrary path in inline direction
    xlvertices : :obj:`plt.axes`, optional
        Vertices of arbitrary path in crossline direction
    dil : :obj:`int` or :obj:`float`
        Sampling in inline direction
    dxl : :obj:`int` or :obj:`float`
        Sampling in crossline direction
    il0 : :obj:`int` or :obj:`float`
        First inline
    xl0 : :obj:`int` or :obj:`float`
        First crossline
    sampling : :obj:`int` or :obj:`float`
        Sampling

    Returns
    -------
    ils : :obj:`np.ndarray`
        Inlines of arbitrary path
    xls : :obj:`np.ndarray`
        Crosslines of arbitrary path
    iils : :obj:`np.ndarray`
        Indices of inlines of arbitrary path
    ixls : :obj:`np.ndarray`
        Indices of crosslines of arbitrary path
    nedges : :obj:`np.ndarray`
        Number of inlines/crosslines for each edge
    """
    nedges = []  # number of traces for each edge (between two verticies)
    ils, xls = [], []

    distance = np.sqrt(np.diff(ilvertices)**2 + np.diff(xlvertices)**2)
    nsamples = (distance/sampling).astype(int)

    for ilvertex, ilvertex1, xlvertex, xlvertex1, nsamp in \
            zip(ilvertices[:-1], ilvertices[1:],
                xlvertices[:-1], xlvertices[1:], nsamples):
        if ilvertex != ilvertex1:
            pol = np.polyfit([ilvertex, ilvertex1],
                             [xlvertex, xlvertex1], deg=1)
            illoc = np.linspace(min([ilvertex, ilvertex1]),
                                max([ilvertex, ilvertex1]), nsamp)
            if ilvertex > ilvertex1:
                illoc = illoc[::-1]
            ils.extend(list(illoc))
            xls.extend(list(np.polyval(pol, illoc)))
            nedges.append(len(list(illoc)))
        else:
            pol = np.polyfit([xlvertex, xlvertex1],
                             [ilvertex, ilvertex1], deg=1)
            xlloc = np.linspace(min([xlvertex, xlvertex1]),
                                max([xlvertex, xlvertex1]), nsamp)
            if xlvertex > xlvertex1:
                xlloc = xlloc[::-1]
            xls.extend(list(xlloc))
            ils.extend(list(np.polyval(pol, xlloc)))
            nedges.append(len(list(xlloc)))

    iils = np.array((ils - il0)/dil).astype(int)
    ixls = np.array((xls - xl0)/dxl).astype(int)
    nedges = np.cumsum(np.array(nedges))[:-1]
    return ils, xls, iils, ixls, nedges


def _seismic_view(seismic, tz, ilplot, xlplot, tzplot, axs=None, which='all',
                  il=None, xl=None, taxis=True, tzoom=None, tzoom_index=True,
                  ilzoom=None, xlzoom=None, tzfirst=False,
                  scale=1., clip=1., clim=[], cmap='seismic', cbar=False,
                  cbarhoriz=False, interp=None, figsize=(20,6),  title=''):
    r"""Quick visualization of Seismic object

    Refer to :func:`pysubsurface.object.Seismic.view` for input parameter definition.

    Returns
    -------
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
        Axes handles

    """
    # find out dimensions and axes
    nt, nil, nxl = seismic.ntz, seismic.nil, seismic.nxl

    # extract data to visualize
    if seismic._loadcube:
        if seismic._tzfirst:
            data = np.moveaxis(seismic.data, -1, 0),
        else:
            data = seismic.data
        if which == 'all':
            data = (data[ilplot, :, :].T,
                    data[:, xlplot, :].T,
                    data[:, :, tzplot])
        elif which == 'il':
            data = data[ilplot, :, :].T
        elif which == 'xl':
            data = data[:, xlplot, :].T
        else:
            data = data[:, :, tzplot]
    else:
        if which == 'all':
            data = (seismic.read_inline(ilplot).T,
                    seismic.read_crossline(xlplot).T,
                    seismic.read_slice(tzplot).T)
        elif which == 'il':
            data = seismic.read_inline(ilplot).T
        elif which == 'xl':
            data = seismic.read_crossline(xlplot).T
        else:
            data = seismic.read_slice(tzplot)

    # define axes
    if tzoom is not None:
        tzoom = sorted(tzoom, reverse=True)
        if tzoom_index:
            tzoom = tz[tzoom]
    if il is None:
        il=np.arange(nil)

    if xl is None:
        xl=np.arange(nxl)

    # define colorbar limits
    if len(clim) == 0:
        if which == 'all':
            clim=[-clip*np.abs(scale)*np.nanmax(data[0]),
                  clip*np.abs(scale)*np.nanmax(data[0])]
        else:
            clim = [-clip * np.abs(scale) * np.nanmax(data),
                    clip * np.abs(scale) * np.nanmax(data)]

    # create figure with chosen layout
    if axs is None:
        if which == 'all':
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            plt.suptitle(title, y=0.95, fontsize=18, weight='bold')
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    # display three slices
    if which == 'all':
        axs[0][0].imshow(scale * data[0], cmap=cmap,
                         vmax=clim[1], vmin=clim[0],
                         extent=(xl[0], xl[-1], tz[-1], tz[0]),
                         interpolation=interp)
        axs[0][0].axhline(tz[tzplot], c='k', ls='--')
        axs[0][0].axvline(xl[xlplot], c='k', ls='--')
        axs[0][0].set_title('IL slice at {}'.format(il[ilplot]),
                            fontsize=10, weight='bold')
        axs[0][0].set_xlabel('XL')
        axs[0][0].set_ylabel('Time' if taxis else 'Depth')
        axs[0][0].axis('tight')
        if tzoom is not None:
            axs[0][0].set_ylim(tzoom)
        if xlzoom is not None:
            axs[0][0].set_xlim(xlzoom)

        axs[0][1].imshow(scale * data[1], cmap=cmap,
                         vmax=clim[1], vmin=clim[0],
                         extent=(il[0], il[-1], tz[-1], tz[0]),
                         interpolation=interp)
        axs[0][1].axhline(tz[tzplot], c='k', ls='--')
        axs[0][1].axvline(il[ilplot], c='k', ls='--')
        axs[0][1].set_title('XL slice at {}'.format(xl[xlplot]),
                            fontsize=10, weight='bold')
        axs[0][1].set_xlabel('IL')
        axs[0][1].set_ylabel('Time' if taxis else 'Depth')
        axs[0][1].axis('tight')
        if tzoom is not None:
            axs[0][1].set_ylim(tzoom)
        if ilzoom is not None:
            axs[0][1].set_xlim(ilzoom)

        im = axs[1][0].imshow(scale * data[2], cmap=cmap,
                              vmax=clim[1], vmin=clim[0],
                              extent=(xl[0],xl[-1],il[-1],il[0]),
                              interpolation=interp)
        axs[1][0].set_title('tslice at {}'.format(tz[tzplot]),
                            fontsize=10, weight='bold')
        axs[1][0].set_xlabel('XL')
        axs[1][0].set_ylabel('IL')
        axs[1][0].axhline(il[ilplot], c='k', ls='--')
        axs[1][0].axvline(xl[xlplot], c='k', ls='--')
        if cbar:
            plt.colorbar(im, ax=axs[1][0])
        axs[1][0].axis('tight')
        if xlzoom is not None:
            axs[1][0].set_xlim(xlzoom)
        if ilzoom is not None:
            axs[1][0].set_ylim(ilzoom)
        axs[1][1].set_axis_off()

    # display il slice
    elif which == 'il':
        im = axs.imshow(scale * data, cmap=cmap,
                        vmax=clim[1], vmin=clim[0],
                        extent=(xl[0], xl[-1], tz[-1], tz[0]),
                        interpolation=interp)
        axs.set_title(title+' (IL= {})'.format(il[ilplot]),
                      fontsize=13, weight='bold')
        axs.set_xlabel('XL')
        axs.set_ylabel('Time' if taxis else 'Depth')
        axs.axis('tight')
        if cbar:
            if cbarhoriz:
                divider = make_axes_locatable(axs)
                cax = divider.append_axes('bottom', size='5%', pad=0.05)
                cax.axis('off')
                plt.colorbar(im, ax=cax, orientation='horizontal',
                             shrink=0.3, pad=0.07)
            else:
                plt.colorbar(im, ax=axs, shrink=0.3)
        if tzoom is not None:
            axs.set_ylim(tzoom)
        if xlzoom is not None:
            axs.set_xlim(xlzoom)

    # display xl slice
    elif which == 'xl':
        im = axs.imshow(scale * data, cmap=cmap,
                        vmax=clim[1], vmin=clim[0],
                         extent=(il[0], il[-1], tz[-1], tz[0]),
                        interpolation=interp)
        axs.set_title(title+' (XL={})'.format(xl[xlplot]),
                      fontsize=13, weight='bold')
        axs.set_xlabel('IL')
        axs.set_ylabel('Time' if taxis else 'Depth')
        axs.axis('tight')
        if cbar:
            if cbarhoriz:
                divider = make_axes_locatable(axs)
                cax = divider.append_axes('bottom', size='5%', pad=0.05)
                cax.axis('off')

                plt.colorbar(im, ax=cax, orientation='horizontal',
                             shrink=0.3, pad=0.1)
            else:
                plt.colorbar(im, ax=axs, shrink=0.3)
        if tzoom is not None:
            axs.set_ylim(tzoom)
        if ilzoom is not None:
            axs.set_xlim(ilzoom)

    # display time slice
    else:
        im = axs.imshow(scale * data, cmap=cmap,
                        vmax=clim[1], vmin=clim[0],
                              extent=(xl[0], xl[-1], il[-1], il[0]),
                        interpolation=interp)
        axs.set_title(title+' (tz={})'.format(tz[tzplot]))
        axs.set_xlabel('XL')
        axs.set_xlabel('IL')
        if cbar:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cax.axis('off')

            plt.colorbar(im, ax=cax, orientation='horizontal',
                         shrink=0.3, pad=0.1)
        axs.axis('tight')
        if xlzoom is not None:
            axs[1][0].set_xlim(xlzoom)
        if ilzoom is not None:
            axs[1][0].set_ylim(ilzoom)
    return fig, axs


class Seismic:
    """Seismic object.

    Create object containing a seismic cube with regular inline
    and crossline axes.

    Parameters
    ----------
    filename : :obj:`str`
        File name
    iline : :obj:`int`, optional
        Byte location of inline
    xline : :obj:`int`, optional
        Byte location of crossline
    cdpy : :obj:`int`, optional
        Byte location of CDP_Y
    cdpx : :obj:`int`, optional
        Byte location of CDP_X
    tzfirst : :obj:`int`, optional
        Bring time/depth axis to first dimension
    taxis : :obj:`int`, optional
        Define either time (``True``) or depth (``False``) axis
    scale : :obj:`int`, optional
        Apply scaling to data when reading it
    loadcube : :obj:`int`, optional
        Load data into ``self.data`` variable during initialization (``True``)
        or not (``False``)
    kind : :obj:`str`, optional
        ``local`` or ``onprem`` when data are stored locally in a folder,
    verb : :obj:`bool`, optional
        Verbosity

    """
    def __init__(self, filename, iline=189, xline=193, cdpy=185, cdpx=181,
                 tzfirst=False, taxis=True, scale=1, loadcube=True,
                 kind='local', ads=None, verb=False):
        self.filename = filename
        self._iline = iline
        self._xline = xline
        self._cdpy = cdpy
        self._cdpx = cdpx
        self._tzfirst = tzfirst
        self._taxis = taxis
        self._loadcube = loadcube
        self._scale = scale
        self._kind = kind
        self._verb = verb
        self._saveable = True
        self._interpret_seismic()
        if self._loadcube:
            self._data = self._read_cube()

    @property
    def data(self):
        if not self._loadcube:
            self._loadcube = True
            self._data = self._read_cube()
        return self._data

    @data.setter
    def data(self, data):
        if data.shape != self.dims:
            raise ValueError('Provided data does not match '
                             'size of Seismic object')
        self._data = data

    def __str__(self):
        descr = 'Seismic object:\n' + \
                '---------------\n' + \
                'Filename: {}\n'.format(self.filename) + \
                'nil={}, nxl={}, ntz={}\n'.format(self.nil, self.nxl,
                                                  self.ntz) + \
                'il = {} - {}\n'.format(self.ilines[0], self.ilines[-1]) + \
                'xl = {} - {}\n'.format(self.xlines[0], self.xlines[-1]) + \
                'tz = {} - {}\n'.format(self.tz[0], self.tz[-1])
        if self._loadcube:
            descr = descr + \
                    'min = {0:.3f},\nmax = {1:.3f}\n'.format(self.data.min(),
                                                             self.data.max()) + \
                    'mean = {0:.3f},\nstd = {1:.3f}'.format(np.mean(self.data),
                                                            np.std(self.data))
        return descr

    def _interpret_seismic(self):
        """Interpret seismic

        Open segy file and interpret its layout. Add useful information to the
        object about dimensions, time/depth-IL-XL axes, and absolute UTM
        coordinates

        Parameters
        ----------
        verb : :obj:`bool`, optional
            Verbosity

        """
        if self._verb:
            print('Interpreting {}...'.format(self.filename))

        try:
            with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
                try:
                    self.ebcdic = \
                        f.text[0].translate(bytearray(_a2e)).decode('ascii')
                except:
                    self.ebcdic = None
                self.tz = f.samples
                self.dtz = f.samples[1]-f.samples[0] # sampling in msec / m
                self.ntz = len(f.samples)

                self.ilines = f.ilines
                self.xlines = f.xlines
                self.dil = self.ilines[1]-self.ilines[0]
                self.dxl = self.xlines[1]-self.xlines[0]

                self.nil, self.nxl = len(self.ilines), len(self.xlines)
                self.dims = (self.nil, self.nxl, self.ntz)

                self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
                if self.sc < 0:
                    self.sc = 1. / abs(self.sc)
                self.cdpy = self.sc * f.attributes(self._cdpy)[:]
                self.cdpx = self.sc * f.attributes(self._cdpx)[:]
        except:
            raise ValueError('{} not available...'.format(self.filename))

    def _read_cube(self):
        """Read seismic cube

        Returns
        -------
        data : :obj:`np.ndarray`
            Data

        """
        if self._verb:
            print('Reading {}...'.format(self.filename))

        with segyio.open(self.filename, "r", iline=self._iline,
                         xline=self._xline) as f:
            data = segyio.tools.cube(f) * self._scale
        if self._tzfirst:
            data = np.moveaxis(self.data, -1, 0)
        return data

    def info(self, level=2):
        """Print summary of segy file binary headers

        Parameters
        ----------
        level : :obj:`int`, optional
            Level of details (``0``: only geometry,
            ``1``: most important header words, ``2``: all header words)

        """
        _segyinfo(self.filename,  iline = self._iline, xline = self._xline,
                  level=level)

    def copy(self, empty=False):
        """Return a copy of the object.

        Parameters
        ----------
        empty : :obj:`bool`
            Copy input data (``True``) or just create an empty data (``False``)

        Returns
        -------
        cubecopy : :obj:`pysubsurface.objects.Seismic`
            Copy of Seismic object

        """
        # read seismic data before copying
        _ = self.data.shape

        seismiccopy = copy.deepcopy(self)

        if empty:
            seismiccopy._data = np.zeros_like(self.data)
        else:
            seismiccopy._data = np.copy(self.data)
        return seismiccopy

    def copy_subcube(self, ilinein=0, ilineend=None,
                     xlinein=0, xlineend=None,
                     tzin=0, tzend=None):
        """Return a copy of the object containing a subcube of the original
        data. Note that cdpx and cdpy are those of the original data as they
        are commonly used by other routines in conjunction with the
        _segyheaderword_from_index routine.

        Parameters
        ----------
        ilinein : :obj:`int`, optional
            Index of first inline (included) to read
        ilineend : :obj:`int`, optional
            Index of last inline (excluded) to read
            (in ``None`` use last available inline)
        xlinein : :obj:`int`, optional
            Index of first crossline (included) to read
        xlineend : :obj:`int`, optional
            Index of last crossline (excluded) to read
            (in ``None`` use last available crossline)
        tzin : :obj:`int`, optional
            Index of first time/depth (included) to read
        tzend : :obj:`int`, optional
            Index of last time/depth (excluded) to read
            (in ``None`` use last available time/depth sample)
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        cubecopy : :obj:`pysubsurface.objects.Seismic`
            Copy of Seismic object

        """
        tzlims, itzlims = _extract_lims(self.tz, [tzin, tzend], name='tz')
        xllims, ixllims = _extract_lims(self.xlines, [xlinein, xlineend],
                                        name='xlines')
        illims, iillims = _extract_lims(self.ilines, [ilinein, ilineend],
                                        name='ilines')
        seismiccopy = copy.deepcopy(self)
        seismiccopy._saveable = False
        seismiccopy._loadcube = True

        seismiccopy.ilines = self.ilines[iillims[0]:iillims[1]]
        seismiccopy.xlines = self.xlines[ixllims[0]:ixllims[1]]
        seismiccopy.tz = self.tz[itzlims[0]:itzlims[1]]
        seismiccopy.dims = (iillims[1] - iillims[0],
                            ixllims[1] - ixllims[0],
                            itzlims[1] - itzlims[0])

        # Read data of interest
        seismiccopy.data = \
            self.read_subcube(ilinein=iillims[0], ilineend=iillims[1],
                              xlinein=ixllims[0], xlineend=ixllims[1],
                              tzin=itzlims[0], tzend=itzlims[1])
        return seismiccopy

    def save(self, outputfilename, verb=False):
        """Save object to segy file.

        Parameters
        ----------
        outputfilename : :obj:`str`
            Name of output file
        verb : :obj:`bool`, optional
            Verbosity

        """
        if self._savable:
            if verb:
                print('Saving seismic cube to {}...'.format(outputfilename))

            if self._tzfirst:
                _write_seismic(self.filename,outputfilename,
                               np.moveaxis(self.data, 0, -1),
                               iline=self._iline,  xline=self._xline)
            else:
                _write_seismic(self.filename,outputfilename,
                               self.data, iline=self._iline,  xline=self._xline)
        else:
            logging.warning('Seismic is not savable, it may have originated from '
                            'copy_subcube...')

    def read_headervalues(self, index, headerwords=(181, )):
        """Return header values at specific index

        Parameters
        ----------
        index : :obj:`int`
            Index at which header is returned
        headerwords : :obj:`tuple` or :obj:`int`, optional
            Headerwords (using segyio nomenclature or index integers)

        Returns
        -------
        head : :obj:`dict` or :obj:`float`
            Header value (if only one is requested) or dictionary with
            headerwords and values (if more requested)

        """
        if isinstance(headerwords, int):
            headerwords = (headerwords, )

        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            head = f.header[index]
            head = head[headerwords]
        if len(headerwords) == 1:
            head = list(head.values())[0]
        return head

    def read_subcube(self, ilinein=0, ilineend=None,
                     xlinein=0, xlineend=None,
                     tzin=0, tzend=None, verb=False):
        """Read seismic subcube

        Read subset of ilines directly from segy file (without reading the
        entire cube first)

        Parameters
        ----------
        ilinein : :obj:`int`, optional
            Index of first inline (included) to read
        ilineend : :obj:`int`, optional
            Index of last inline (excluded) to read
            (in ``None`` use last available inline)
        xlinein : :obj:`int`, optional
            Index of first crossline (included) to read
        xlineend : :obj:`int`, optional
            Index of last crossline (excluded) to read
            (in ``None`` use last available crossline)
        tzin : :obj:`int`, optional
            Index of first time/depth (included) to read
        tzend : :obj:`int`, optional
            Index of last time/depth (excluded) to read
            (in ``None`` use last available time/depth sample)
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        subcube : :obj:`np.ndarray`
            Subcube

        """
        if ilineend is None:
            ilineend = (self.nil-1)

        if ilinein < 0 or ilineend > self.nil:
            raise ValueError('Selected inlines exceed available range....')

        if verb:
            print('Reading {} for il={}-{}...'.format(self.filename,
                                                      self.ilines[ilinein],
                                                      self.ilines[ilineend]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            # Read inlines
            subcube = segyio.collect(f.iline[self.ilines[ilinein]:
                                             self.ilines[ilineend]]) * \
                      self._scale
            # Extract crosslines
            if xlinein > 0 or xlineend is not None:
                if xlineend is None:
                    xlineend = (self.nxl - 1)
                subcube = subcube[:, xlinein:xlineend]
            # Extract time/depth
            if tzin > 0 or tzend is not None:
                if tzend is None:
                    tzend = (self.ntz - 1)
                subcube = subcube[:, :, tzin:tzend]
        return subcube

    def read_inline(self, iline=0, verb=False):
        """Read seismic inline section directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        iline : :obj:`int`, optional
            Index of inline to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        section : :obj:`np.ndarray`
            Inline section

        """
        if iline < 0 or iline > self.nil:
            raise ValueError('Selected inline exceeds available range....')

        if verb:
            print('Reading {} for il={}...'.format(self.filename,
                                                   self.ilines[iline]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            section = f.iline[self.ilines[iline]] * self._scale
        return section

    def read_crossline(self, xline=0, verb=False):
        """Read seismic crossline section directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        xline : :obj:`int`, optional
            Index of crossline to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        section : :obj:`np.ndarray`
            Crossline section

        """
        if xline < 0 or xline > self.nxl:
            raise ValueError('Selected crossline exceeds available range....')

        if verb:
            print('Reading {} for xl={}...'.format(self.filename,
                                                   self.xlines[xline]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            section = f.xline[self.xlines[xline]] * self._scale
        return section

    def read_slice(self, itz=0, verb=False):
        """Read seismic depth/time slice directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        itz : :obj:`int`, optional
            Index of depth/time to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        slice : :obj:`np.ndarray`
            Time/depth slice

        """
        if itz < 0 or itz > self.ntz:
            raise ValueError('Selected time/depth exceeds available range....')

        if verb:
            print('Reading {} for tz={}...'.format(self.filename,
                                                   self.samples[itz]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            slice = f.depth_slice[itz] * self._scale
        return slice

    def read_iline_crossline_intersection(self, iline=0, xline=0, verb=False):
        """Read seismic trace at intersection between inline and crossline
        from segy file (without reading the entire cube first)

        Parameters
        ----------
        itrace : :obj:`int`, optional
            Index of trace to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        trace : :obj:`np.ndarray`
            Trace
        """
        itrace = self.nxl * iline + xline
        return self.read_trace(itrace=itrace, verb=verb)

    def read_trace(self, itrace=0, verb=False):
        """Read seismic depth/time trace directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        itrace : :obj:`int`, optional
            Index of trace to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        trace : :obj:`np.ndarray`
            Trace
        """
        if verb:
            print('Reading {} trace={}...'.format(self.filename, itrace))

        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            trace = f.trace[itrace] * self._scale
        return trace

    def extract_subcube(self, tzlims=(None, None), xllims=(None, None),
                        illims=(None, None), plotflag=False):
        """Return a subset of the cube

        Extract subset of cube given time/depth limits (``tzlims``),
        crossline limits (``xllims``), and inline limits (``illims``). For all
        limits, use ``None`` to start from first/go until last element.

        Parameters
        ----------
        tzlims : :obj:`tuple`, optional
            Start and end time/depth values to extract
            (both start and end values are included)
        xllims : :obj:`tuple`, optional
            Start and end crossline values to extract
            (both start and end values are included)
        illims : :obj:`tuple`, optional
            Start and end inline values to extract
            (both start and end values are included)
        plotflag : :obj:`bool`, optional
            Quickplot

        Returns
        -------
        subcube : :obj:`np.ndarray`
            Subcube
        itzlims : :obj:`tuple`
            Time/depth start and end indexes
        ixllims : :obj:`tuple`
            Crossline start and end indexes
        iillims : :obj:`tuple`
            Inline start and end indexes

        """
        # identify axes limits
        tzlims, itzlims = _extract_lims(self.tz, tzlims, name='tz')
        xllims, ixllims = _extract_lims(self.xlines, xllims, name='xlines')
        illims, iillims = _extract_lims(self.ilines, illims, name='ilines')
        # extract subcube
        subcube = self.data[iillims[0]:iillims[1],
                            ixllims[0]:ixllims[1],
                            itzlims[0]:itzlims[1]]
        # quickplot
        if plotflag:
            Cube(subcube).view()

        return subcube, itzlims, ixllims, iillims

    def extract_trace_verticalwell(self, well, verb=False):
        """Extract seismic trace at inline and crossline closest to a
        vertical well

        Parameters
        ----------
        well : :obj:`Well`
            Well object
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        trace : :obj:`np.ndarray`
            Seismic trace at well location

        """
        if verb:
            print('Reading trace from {} closest to '
                  'well {}...'.format(self.filename, well.wellname))

        # find IL and XL the well is passing through
        ilwell, xlwell = _findclosest_well_seismicsections(well, self,
                                                           verb=verb)
        if self._loadcube:
            trace = self.data[findclosest(self.ilines, ilwell),
                              findclosest(self.xlines, xlwell)]
        else:
            iilwell, ixlwell = findclosest(self.ilines, ilwell), \
                               findclosest(self.xlines, xlwell)
            itrace = iilwell*self.nxl + ixlwell

            with segyio.open(self.filename, "r",
                             iline=self._iline, xline=self._xline) as f:
                head = f.header[itrace]
                # check that trace has correct inline and crossline
                if head[segyio.TraceField.INLINE_3D] != self.ilines[iilwell] or \
                   head[segyio.TraceField.CROSSLINE_3D] != self.xlines[ixlwell]:
                    raise ValueError('IL or XL identified do '
                                     'not match those from trace')
                trace = self.read_trace(itrace=itrace)
        return trace

    def extract_trace_deviatedwell(self, well):
        """Extract seismic trace at inlines and crosslines closest to a
        deviated well

        Parameters
        ----------
        well : :obj:`Well`
            Well object

        Returns
        -------
        trace : :obj:`np.ndarray`
            Seismic trace at well location
        tz : :obj:`np.ndarray`
            Time/depth axis along extracted trace

        """
        ilwell, xlwell = \
            _findclosest_well_seismicsections(well, self, traj=True)
        ilwell = (np.round((ilwell - self.ilines[0]) /
                           self.dil)).astype(np.int)
        xlwell = (np.round((xlwell - self.xlines[0]) /
                           self.dxl)).astype(np.int)
        tz0 = self.tz[0]
        izwell = (np.round((well.trajectory.df['TVDSS'].values - tz0) /
                           self.dtz)).astype(np.int)

        # remove elements where well extend beyond max depth/time of seismic axis
        mask = (izwell < self.ntz) & (izwell > 0)
        ilwell = ilwell[mask]
        xlwell = xlwell[mask]
        izwell = izwell[mask]

        tz = self.tz[izwell]
        if self._tzfirst:
            trace = \
                self.data[izwell.astype(np.int), ilwell.astype(np.int), xlwell.squeeze()]
        else:
            trace = \
                self.data[ilwell.astype(np.int), xlwell.astype(np.int), izwell.squeeze()]
        return trace, tz

    def extract_attribute_map(self, horizon, intwin=None,
                              intwin_above=0, intwin_below=0,
                              inttype='mean', scale=1.):
        """Attribute map from seismic cube

        Extract values in seismic cube along a surface ``hor``. The extraction
        can be instantaneous (at closest values to surface) or an aggregation
        operation can be applied over window of
        size ``intwin`` along the time/depth axis

        Parameters
        ----------
        horizon : :obj:`pysubsurface.objects.Surface`
            Horizon in time or depth
        intwin : :obj:`int`, optional
            Number of samples to be used on either side of surface in window
        (if ``None`` use intwin_above and intwin_below)
        intwin_above : :obj:`int`, optional
            Number of samples to be used in window above surface
            (used when intwin is ``None``)
        intwin_below : :obj:`int`, optional
            Number of samples to be used in window below surface
            (used when intwin is ``None``)
        inttype : :obj:`str`, optional
            Type of averaging to be performed in the window (``mean``, ``rms``,
            ``max``, ``min``, ``maxabs``)
        scale : :obj:`float`, optional
            Scaling to be applied to horizon values prior to attribute
            extraction

        Returns
        -------
        attrmap : :obj:`pysubsurface.objects.Surface`
            Attribute map in common grid
        horizoncommon : :obj:`pysubsurface.objects.Surface`
            Horizon map in common grid

        """
        attrmap, horizoncommon = \
            _extract_attribute_map(self, horizon, intwin=intwin,
                                   intwin_above=intwin_above,
                                   intwin_below=intwin_below,
                                   inttype=inttype, scale=scale)
        return attrmap, horizoncommon

    def extract_interval_map(self, horizon_top, horizon_base,
                             inttype='mean', scale=1.):
        """Interval attribute map from seismic cube

        Extract values in seismic cube along a surface ``hor``. The extraction
        can be instantaneous (at closest values to surface) or an aggregation
        operation can be applied over window of
        size ``intwin`` along the time/depth axis

        Parameters
        ----------
        horizon_top : :obj:`pysubsurface.objects.Surface`
            Top horizon in time or depth
        horizon_base : :obj:`pysubsurface.objects.Surface`
            Base horizon in time or depth. It may be on a different grid than
            the top horizon but should be in the same domain. If the grid is
            different the base horizon will be first interpolated to the grid
            of the top horizon.
        inttype : :obj:`str`, optional
            Type of averaging to be performed in the window (``mean``, ``rms``,
            ``max``, ``min``, ``maxabs``)
        scale : :obj:`float`, optional
            Scaling to be applied to horizon values prior to attribute
            extraction

        Returns
        -------
        attrmap : :obj:`pysubsurface.objects.Surface`
            Attribute map in common grid
        horizon_top_common : :obj:`pysubsurface.objects.Surface`
            Top horizon in common grid
        horizon_base_common : :obj:`pysubsurface.objects.Surface`
            Base horizon in common grid

        """
        attrmap, horizon_top_common, horizon_base_common = \
            _extract_interval_map(self, horizon_top, horizon_base,
                                  inttype=inttype, scale=scale)
        return attrmap, horizon_top_common, horizon_base_common

    def closest_line(self, other, iline_other=None, xline_other=None):
        """Identify inline (or crossline) in ``self`` closest to an inline
        (or crossline) in another :obj:`pysubsurface.objects.Seismic` object ``other``

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Seismic`
            Other seismic cube used as reference for extraction of closest
            inline (or crossline)
        iline : :obj:`int`, optional
            Inline number in ``other`` (if ``None`` use ``xline``)
        xline : :obj:`int`, optional
            Crossline number in ``other`` (if ``None`` use ``iline``)

        Returns
        -------
        line : :obj:`int`
            Closest inline (or crossline) in ``self``
        index : :obj:`int`
            Index of closest inline (or crossline) in ``self``
        cdpline : :obj:`fload`
            CDPy(x) at chosen line
        index : :obj:`int`
            Index of closest inline (or crossline) in ``self``

        """
        if iline_other is not None:
            iilother = findclosest(iline_other, other.ilines)
            cdpy_ilother = other.cdpy.reshape(other.nil, other.nxl)
            cdpx_ilother = other.cdpx.reshape(other.nil, other.nxl)

            cdpy_ilother, cdpx_ilother = cdpy_ilother[iilother, other.nxl // 2], \
                                         np.unique(cdpx_ilother[iilother])[0]

            il, xl = _findclosest_point((cdpx_ilother, cdpy_ilother), self)
            iil, ixl = findclosest(il, self.ilines), \
                       findclosest(xl, self.xlines)
            line, index = il, iil
            cdpline = cdpx_ilother
        else:
            ixlother = findclosest(xline_other, other.xlines)
            cdpy_xlother = other.cdpy.reshape(other.nil, other.nxl)
            cdpx_xlother = other.cdpx.reshape(other.nil, other.nxl)

            cdpy_xlother, cdpx_xlother = np.unique(cdpy_xlother[:, ixlother])[0], \
                                         cdpx_xlother[other.nil // 2, ixlother]

            il, xl = _findclosest_point((cdpx_xlother, cdpy_xlother), self)
            iil, ixl = findclosest(il, self.ilines), \
                       findclosest(xl, self.xlines)
            line, index = xl, ixl
            cdpline = cdpy_xlother

        return line, index, cdpline

    def estimate_wavelet(self, method='stat', ntwest=101,
                         tzlims=(None, None), xllims=(None, None),
                         illims=(None, None), jtz=1, jil=1, jxl=1,
                         plotflag=False, savefig=None, **kwargs_estimate):
        """Estimate wavelet.

        Parameters
        ----------
        method : :obj:`str`, optional
            Estimation method (Statistical: ``stat``)
        ntwest : :obj:`int`
            Number of samples of estimated wavelet
        tzlims : :obj:`tuple`, optional
            Start and end time/depth values to extract
            (both start and end values are included)
        xllims : :obj:`tuple`, optional
            Start and end crossline values to extract
            (both start and end values are included)
        illims : :obj:`tuple`, optional
            Start and end inline values to extract
            (both start and end values are included)
        jtz : :obj:`int`, optional
            Jump in tz axis (data is subsampled prior to wavelet estimation)
        jil : :obj:`int`, optional
            Jump in inline axis (data is subsampled prior to wavelet estimation)
        jxl : :obj:`int`, optional
            Jump in crossline axis (data is subsampled prior to wavelet estimation)
        plotflag : :obj:`bool`, optional
            Quickplot
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs_estimate : :obj:`dict`, optional
            Parameters for estimation method

        Returns
        -------
        wav : :obj:`np.ndarray`
            Estimated wavelet
        wavf : :obj:`np.ndarray`
            Estimated wavelet in frequency

        """
        subcube, itzlims = self.extract_subcube(tzlims=tzlims, xllims=xllims,
                                                illims=illims)[0:2]

        subcube = subcube[::jil, ::jxl, ::jtz]

        if method == 'stat':
            wav, wavf, twav, fwav, wavc = \
                statistical_wavelet(subcube, ntwest=ntwest, dt=self.dtz,
                                    **kwargs_estimate)
        else:
            NotImplementedError('method {} not implemented'.format(method))

        if plotflag:
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle('Wavelet estimation (method={})'.format(method),
                         fontsize=14, fontweight='bold', y=1.03)
            ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            ax3 = plt.subplot2grid((2, 2), (1, 1))
            _, ax1 = self.view(axs=ax1, which='il', tzoom=itzlims,
                               tzoom_index=True, clip=0.8, cmap='gray')
            ax1.axis('tight')
            ax1.set_xlabel('XL')
            ax1.set_ylabel('Time/Depth')
            ax1.set_title('Seismic')
            ax2.plot(twav, wav, 'k', lw=2)
            ax2.set_xlabel('Time/Depth')
            ax2.set_title('Wavelet in time/depth')
            ax2.axis('tight')
            ax3.plot(fwav, np.abs(wavf), 'k')
            ax3.set_xlabel('Frequency')
            ax3.set_title('Wavelet in frequency')
            plt.tight_layout()

            # savefig
            if savefig is not None:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return wav, wavf


    #########
    # Viewers
    #########
    def view(self, axs=None, ilplot=None, xlplot=None, tzplot=None, which='all',
             tzoom=None, tzoom_index=True, tzshift=0.,
             ilzoom=None, xlzoom=None, scale=1.,
             horizons=None, horcolors=[], scalehors=1.,
             hornames=False, horlw=2, faults=None, faultsmindist=None,
             reservoir=None, clip=1., clim=[], cmap='seismic',
             cbar=False, interp=None, figsize=(20, 6),  title='', savefig=None):
        """Quick visualization of Seismic object.

        Parameters
        ----------
        axs : :obj:`plt.axes`
            Axes handles (if ``None`` create new figure)
        ilplot : :obj:`int`, optional
            Index of inline to plot (if ``None`` show inline in the middle)
        xlplot : :obj:`int`, optional
            Index of crossline to plot
            (if ``None`` show crossline in the middle)
        tzplot : :obj:`int`, optional
            Index of  time/depth slice to plot
            (if ``None`` show slice in the middle )
        which : :obj:`str`, optional
            Slices to visualize. ``all``: inline, crossline and time/depth
            slices, ``il``: inline slice, ``xl``: crossline slice,
            ``tz``: time/depth slice
        tzoom : :obj:`tuple`, optional
            Time/depth start and end values (or indeces) for visualization
            of time/depth axis
        tzoom_index : :obj:`bool`, optional
            Consider values in ``tzoom`` as indeces (``True``) or
            actual values (``False``)
        tzshift : :obj:`float`, optional
            Shift to apply to tz axis in seismic
        ilzoom : :obj:`tuple`, optional
            Inline start and end values (or indeces) for visualization
            of inline axis
        xlzoom : :obj:`tuple`, optional
            Crossline start and end values (or indeces) for visualization
            of crossline axis
        scale : :obj:`float`, optional
            Apply scaling to data when showing it
        horizons : :obj:`pysubsurface.objects.Surface` or :obj:`pysubsurface.objects.Interpretation`
         or :obj:`pysubsurface.objects.Ensemble` or :obj:`list`, optional
            Set of horizons to plot
        horcolors : :obj:`list`, optional
            Horizon colors
        scalehors : :obj:`float`, optional
            Apply scaling to horizons time/depth values when showing them
        hornames : :obj:`bool`, optional
            Add names of horizons (``True``) or not (``False``)
        horlw : :obj:`float`, optional
             Horizons linewidth
        faults : :obj:`dict`, optional
            Set of faults to plot
        faultsmindist : :obj:`int`, optional
            Minimum distance from fault point for interpolated grid (points
            whose distance is bigger than mindist will be masked out)
        reservoir : :obj:`dict`, optional
            Dictionary containing indices of ``top`` and ``base`` reservoir
            as well as the name of ``GOC`` and ``WOC`` to color-fill overlaid
            to seismic (if ``None`` do not color fill)
        clip : :obj:`float`, optional
            Clip to apply to colorbar limits (``vmin`` and ``vmax``)
        clim : :obj:`float`, optional
            Colorbar limits (if ``None`` infer from data and
            apply ``clip`` to those)
        cmap : :obj:`str`, optional
            Colormap
        cbar : :obj:`bool`, optional
            Show colorbar
        interp : :obj:`str`, optional
            Imshow interpolation
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        savefig : :obj:`str`, optional
            Figure filename, including path of location where to save plot
            (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """

        # get horizons to plot
        print(type(horizons), isinstance(horizons, pysubsurface.objects.Interpretation))
        if horizons is None:
            plothorizons = False
        elif isinstance(horizons, list):
            surfaces = horizons[0].surfaces
            plothorizons = True
            onehorizon = False
        elif isinstance(horizons, pysubsurface.objects.Interpretation):
            surfaces = horizons.surfaces
            horizons = [horizons, ]
            plothorizons = True
            onehorizon = False
        elif isinstance(horizons, pysubsurface.objects.Ensemble):
            surfaces = horizons.interpretations[horizons.firstintname].surfaces
            horizons = [horizons, ]
            plothorizons = True
            onehorizon = False
        # stop supporting single Surface?
        elif isinstance(horizons, pysubsurface.objects.Surface):
            surfaces = [horizons]
            plothorizons = True
            onehorizon = True
        else:
            raise TypeError('horizons must be of type Surface or '
                            'Interpretation...')
        if isinstance(horcolors, str):
            horcolors = [horcolors] * len(surfaces)
        if plothorizons and len(horcolors) == 0:
            horcolors = ['k'] * len(surfaces)

        # define seismic slices
        if ilplot is None:
            ilplot = int(len(self.ilines)/2)
        if xlplot is None:
            xlplot = int(len(self.xlines)/2)
        if tzplot is None:
            tzplot = int(len(self.tz)/2)

        # display seismic
        fig, axs = \
            _seismic_view(self, self.tz + tzshift, ilplot,
                          xlplot, tzplot, axs=axs, which=which,
                          il=self.ilines, xl=self.xlines, taxis=self._taxis,
                          tzoom=tzoom, tzoom_index=tzoom_index,
                          ilzoom=ilzoom, xlzoom=xlzoom,
                          tzfirst=self._tzfirst,
                          cmap=cmap, scale=scale, clim=clim, clip=clip,
                          cbar=cbar, cbarhoriz=True if hornames else False,
                          interp=interp, figsize=figsize,
                          title=title)

        # display horizons
        if plothorizons:
            if onehorizon:
                warnings.warn('Currently not available, provide an Intepretation')
            else:
                for iset, hors in enumerate(horizons):
                    hors.view(which=which, axs=axs[0] if which is 'all' else axs,
                              ilplot=self.ilines[ilplot],
                              xlplot = self.xlines[xlplot], tzshift=tzshift,
                              horcolors=horcolors, scalehors=scalehors,
                              hornames=hornames,
                              horlw=horlw if iset == 0 else horlw / 3.,
                              reservoir=reservoir if iset == 0 else None)

        # display faults
        if faults is not None:
            faultnames = list(faults.keys())
            for faultname in faultnames:
                if which == 'all' or which == 'il':
                    fault_y_in_seismic = \
                        self.cdpy.reshape(self.nil, self.nxl)[ilplot]
                    fault_x_in_seismic = \
                        self.cdpx.reshape(self.nil, self.nxl)[ilplot]
                    if len(np.unique(fault_x_in_seismic)) == 1:
                        fault_z_in_seismic = \
                            faults[faultname]['data'].grid(fault_x_in_seismic[0],
                                                           fault_y_in_seismic,
                                                           mindist=faultsmindist) + tzshift
                        if which == 'all':
                            axfault = axs[0][0]
                        else:
                            axfault = axs
                        axfault.plot(self.xlines, fault_z_in_seismic,
                                     faults[faultname]['color'],
                                     lw=1, linestyle='--')
                    else:
                        raise NotImplementedError('Cannot show fault as x axis '
                                                  'is not aligned with IL axis')

                if which == 'all' or which == 'xl':
                    fault_y_in_seismic = \
                        self.cdpy.reshape(self.nil, self.nxl)[:, xlplot]
                    fault_x_in_seismic = \
                        self.cdpx.reshape(self.nil, self.nxl)[:, xlplot]
                    if len(np.unique(fault_y_in_seismic)) == 1:
                        fault_z_in_seismic = \
                            faults[faultname]['data'].grid(fault_x_in_seismic,
                                                           fault_y_in_seismic[0],
                                                           mindist=faultsmindist) + tzshift
                        if which == 'all':
                            axfault = axs[0][1]
                        else:
                            axfault = axs

                        axfault.plot(self.ilines, fault_z_in_seismic,
                                     faults[faultname]['color'],
                                     lw=2, linestyle='--')
                    else:
                        raise NotImplementedError('Cannot show fault as x axis '
                                                  'is not aligned with IL axis')
        # savefig
        if fig is not None and savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, axs

    def view_geometry(self, ax=None, polygon=True, scatter=False, subsample=1,
                      color='k', label=None, figsize=(20, 6),
                      title=None, savefig=None):
        """Quick visualization of Seismic object geometry.

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
             Axes handle (if ``None`` draw a new figure)
        polygon : :obj:`bool`, optional
             Show polygon around points (``True``) or not (``False``)
        scatter : :obj:`bool`, optional
             Show also scatterplot of points (``True``) or not(``False``)
        subsample : :obj:`str`, optional
             Subsampling factor for scatterplot
        color : :obj:`str`, optional
             Color
        label : :obj:`str`, optional
             Label to assign to geometry
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        # display seismic
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        if polygon:
            ax = plot_polygon(ax, self.cdpx[::subsample],
                              self.cdpy[::subsample],
                              scatter=scatter, color=color, label=label)
        else:
            ax.plot(self.cdpx[::subsample], self.cdpy[::subsample],
                    '.', ms=1, color=color, alpha=0.3)
        ax.set_xlabel('World X')
        ax.set_ylabel('World Y')
        ax.axis('equal')

        if title is not None:
            ax.set_title(title, weight='bold')

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax

    def view_cdpslice(self, ax=None, tzplot=-1, scale=1.,
                      clip=1., clim=[], cmap='seismic', cbar=False,
                      subsample=1, figsize=(20, 6), title=None, savefig=None):
        """Quick visualization of depth/time slice of Seismic object.

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handle (if ``None`` draw a new figure)
        tzplot : :obj:`int`, optional
            Index of time/depth slice to plot
            (if ``None`` show slice in the middle )
        scale : :obj:`float`, optional
             Apply scaling to data when showing it
        clip : :obj:`float`, optional
             Clip to apply to colorbar limits (``vmin`` and ``vmax``)
        clim : :obj:`float`, optional
             Colorbar limits (if ``None`` infer from data and
             apply ``clip`` to those)
        cmap : :obj:`str`, optional
             Colormap
        cbar : :obj:`bool`, optional
             Show colorbar
        subsample : :obj:`str`, optional
             Subsampling factor for scatterplot
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """
        # load data if not done already
        tzplot = self.ntz//2 if tzplot==-1 else tzplot
        if not self._loadcube:
            slice = scale*self.read_slice(tzplot)
        else:
            slice = scale*self.data[..., tzplot]

        if len(clim) == 0:
            clim = [-clip * np.nanmax(np.abs(slice)),
                    clip * np.nanmax(np.abs(slice))]

        # display seismic
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        im = ax.scatter(self.cdpx[::subsample], self.cdpy[::subsample],
                        c=slice.flatten()[::subsample], edgecolor='none',
                        cmap=cmap, vmax=clim[1], vmin=clim[0])
        ax.set_xlabel('World X')
        ax.set_ylabel('World Y')
        if title is not None:
            ax.set_title('{0} (tz={1:.1f})'.format(title, self.tz[tzplot]),
                         weight='bold')
        ax.axis('equal')
        if cbar:
            plt.colorbar(im, ax=ax)

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax

    def view_arbitraryline(self, ilvertices, xlvertices, fig=None, ax=None,
                           usevertices=False, tzoom=None, tzoom_index=True,
                           tzshift=0., scale=1., clip=1., clim=[],
                           cmap='seismic', cbar=False,
                           interp=None, alpha=1.,
                           addlines=True, jumplabel=1,
                           horizons=None, horcolors=[], scalehors=1.,
                           hornames=False, horlw=2, reservoir=None,
                           figsize=(20, 6), title=None,
                           savefig=None):
        """Quick visualization of Seismic object along arbitrary line defined
        by ``ilvertices`` and ``xlvertices``.

        Parameters
        ----------
        ilvertices : :obj:`tuple` or :obj:`list`
            Vertices of arbitrary path in inline direction
        xlvertices : :obj:`plt.axes`, optional
            Vertices of arbitrary path in crossline direction
        fig : :obj:`plt.figure`, optional
            Figure handle (if ``None`` draw a new figure)
        ax : :obj:`plt.axes`, optional
            Figure handle (if ``None`` draw a new figure)
        usevertices : :obj:`bool`, optional
            Use vertices directly (``True``) or interpolate
            between them (``False``)
        tzoom : :obj:`tuple`, optional
            Time/depth start and end indeces (or values) for visualization
            of time/depth axis
        tzoom_index : :obj:`bool`, optional
            Consider values in ``tzoom`` as indeces (``True``) or
            actual values (``False``)
        tzshift : :obj:`float`, optional
            Shift to apply to tz axis in seismic
        scale : :obj:`float`, optional
            Apply scaling to data when showing it
        clip : :obj:`float`, optional
            Clip to apply to colorbar limits (``vmin`` and ``vmax``)
        clim : :obj:`float`, optional
            Colorbar limits (if ``None`` infer from data and
            apply ``clip`` to those)
        cmap : :obj:`str`, optional
            Colormap
        cbar : :obj:`bool`, optional
            Show colorbar
        interp : :obj:`str`, optional
            imshow interpolation
        alpha : :obj:`float`, optional
            Transparency
        addlines : :obj:`bool`, optional
            Add vertical lines at vertices
        jumplabel : :obj:`int`, optional
            Jump in the labels on x axis (to avoid overlap)
        horizons : :obj:`pysubsurface.objects.Surface` or :obj:`pysubsurface.objects.Interpretation`
         or :obj:`pysubsurface.objects.Ensemble` or :obj:`list`, optional
            Horizon or interpretation (set of horizons) or set of interpretations
            to plot. If a set of interpretation is provided the first one is
            considered the master interpretation and displayed
            with thicker linewidth
        scalehors : :obj:`float`, optional
            Apply scaling to horizons time/depth values when showing them
        hornames : :obj:`bool`, optional
            Add names of horizons (``True``) or not (``False``)
        horlw : :obj:`float`, optional
            Horizons linewidth
        reservoir : :obj:`dict`, optional
            Dictionary containing indices of ``top`` and ``base`` reservoir
            as well as the name of ``GOC`` and ``WOC`` to color-fill overlaid
            to seismic (if ``None`` do not color fill)
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        savefig : :obj:`str`, optional
            Figure filename, including path of location where to save plot
            (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        # get horizons to plot
        if horizons is None:
            plothorizons = False
        elif isinstance(horizons, list):
            surfaces = horizons[0].surfaces
            plothorizons = True
            onehorizon = False
        elif isinstance(horizons, pysubsurface.objects.Interpretation):
            surfaces = horizons.surfaces
            horizons = [horizons, ]
            plothorizons = True
            onehorizon = False
        elif isinstance(horizons, pysubsurface.objects.Ensemble):
            surfaces = horizons.interpretations[horizons.firstintname].surfaces
            horizons = [horizons, ]
            plothorizons = True
            onehorizon = False
        elif isinstance(horizons, pysubsurface.objects.Surface):
            surfaces = [horizons]
            plothorizons = True
            onehorizon = True
        else:
            raise TypeError('horizons must be of type Surface or '
                            'Interpretation...')

        if isinstance(horcolors, str):
            horcolors = [horcolors]
        if plothorizons and len(horcolors) == 0:
            horcolors = ['k'] * len(surfaces)

        if ax is None and fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            if fig is None:
                raise NotImplementedError('Provide also fig when providing ax')

        if len(clim) == 0:
            clim = [-clip * np.nanmax(np.abs(self.data)),
                    clip * np.nanmax(np.abs(self.data))]
        if tzoom is not None:
            tzoom = sorted(tzoom, reverse=True)
            if tzoom_index:
                tzoom = self.tz[tzoom]

        # compute arbitrary line
        if not usevertices:
            ils, xls, iils, ixls, nedges = \
                _extract_arbitrary_path(ilvertices, xlvertices,
                                        self.dil, self.dxl,
                                        self.ilines[0], self.xlines[0])
        else:
            ils, xls = ilvertices, xlvertices
            iils = np.array((ils - self.ilines[0]) / self.dil).astype(int)
            ixls = np.array((xls - self.xlines[0]) / self.dxl).astype(int)
            nedges = np.cumsum(np.ones(len(ilvertices)))[:-1]

        # display seismic
        if self._loadcube:
            seismicline = scale * self.data[iils, ixls]
        else:
            seismicline = np.zeros((len(iils), self.ntz))
            for i, (iil, ixl) in enumerate(zip(iils, ixls)):
                seismicline[i] = self.read_iline_crossline_intersection(iil, ixl)
        im = ax.imshow(seismicline.T, cmap=cmap, vmin=clim[0], vmax=clim[1],
                       interpolation=interp, alpha=alpha,
                       extent=(0, len(ils), self.tz[-1] + tzshift,
                               self.tz[0] + tzshift))
        if addlines:
            for nedge in nedges:
                ax.axvline(nedge, color='w', linestyle='--', lw=1)
        ax.axis('tight')
        if tzoom is not None:
            ax.set_ylim(tzoom)
        ax.set_xlabel('Arbitrary line')
        ax.set_ylabel('Time' if self._taxis else 'Depth')
        if title is not None:
            ax.set_title('{}'.format(title), weight='bold')

        # add xtick labels
        vertices_text = ['{}{}\n{}{}'.format('IL=' if i == 0 else '', int(iledge),
                                             'XL=' if i == 0 else '', int(xledge)) \
                         for i, (iledge, xledge) in
                         enumerate(zip(ilvertices[::jumplabel], xlvertices[::jumplabel]))]
        ax.set_xticks(np.append(np.insert(nedges, 0, 0), len(ils))[::jumplabel])
        ax.set_xticklabels(vertices_text)

        # add colorbar
        if cbar:
            plt.colorbar(im, ax=ax, shrink=0.3)

        # display horizons
        if plothorizons:
            if onehorizon:
                warnings.warn('Currently not available, provide an Intepretation')
            else:
                for iset, hors in enumerate(horizons):
                    hors.view_arbitratyline(ils, xls, tzshift=tzshift,
                                            horcolors=horcolors,
                                            scalehors=scalehors,
                                            hornames=hornames,
                                            horlw=horlw if iset == 0 else horlw/3.,
                                            reservoir=reservoir if iset == 0 else None,
                                            ax=ax)
            """
            for tmpsurface, color in zip(surfaces, horcolors):
                surface = tmpsurface.copy()
                if not isinstance(surface.data, np_ma.core.MaskedArray):
                    surface.data = np_ma.masked_array(surface.data,
                                                      mask=np.zeros_like(
                                                          surface)) + tzshift

                iils = np.array((ils - surface.il[0]) / surface.dil).astype(int)
                ixls = np.array((xls - surface.xl[0]) / surface.dxl).astype(int)
                masksurface = np.array([True]*len(iils))
                masksurface[((iils>=surface.nil) | (iils<0))] = False
                masksurface[((ixls>=surface.nxl) | (ixls<0))] = False
                surfaceline = surface.data[iils[masksurface],
                                           ixls[masksurface]] * scalehors

                ax.plot(np.arange(len(ils))[masksurface],
                        surfaceline,
                        color, lw=horlw)

                if hornames:
                    pos = np.argwhere(surfaceline.mask is False)[0]
                    if len(pos) == 0:
                        pos = -1
                    else:
                        pos = pos[-1]
                    surface_text = surfaceline[pos]
                    ax.text(1.01 * len(ils),
                            surface_text,
                            os.path.basename(surface.filename),
                            va="center", color=color,
                            bbox=dict(boxstyle="round",
                                      fc=(1., 1., 1.),
                                      ec=color, ))
                """

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax
