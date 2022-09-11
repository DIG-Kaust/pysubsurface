import logging
import os
import copy
import warnings

import numpy as np
import numpy.ma as np_ma
import scipy.interpolate as spint

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar

from azure.datalake.store import multithread
from pysubsurface.objects.utils import _findclosest_point_surface
from pysubsurface.visual.utils import _set_black

plt.rc('font', family='serif')


_FORMATS = {'dsg5_long': {'skiprows_in': 58, 'skiprows_end': 0,
                          'ilpos': 0, 'xlpos': 1,
                          'xpos': 2, 'ypos': 3, 'zpos':4,
                          'nanvals':-999, 'ilxl': True,
                          'fmt': '%-16.7f %-16.7f %-10.5f %-14.7f %-13.7f'},
            'plain_long': {'skiprows_in': 0, 'skiprows_end':0,
                           'ilpos': 0, 'xlpos': 1,
                           'xpos':2, 'ypos':3, 'zpos':4,
                           'nanvals':-999, 'ilxl':True,
                           'fmt':'%8f'}
            }


def _creategrid(ys, xs, ints=[-1, -1], ns=None, returnerror=True):
    """Create grid for regularly (or irregularly) sampled surface
    from list of y (or inline) and x (or crossline) locations

    Parameters
    ----------
    ys : :obj:`np.ndarray`
        Values in y-direction
    xs : :obj:`np.ndarray`
        Values in x-direction
    ints : :obj:`tuple` or :obj:`list`, optional
        Sampling intervals in y and x direction
        (if not provided, find it from data)
    ns : :obj:`tuple` or :obj:`list`, optional
        Number of samples for y and x axis (always prefer over ``ints`` if provided)
    returnerror : :obj:`bool`, optional
        Return error if cannot identify regular axis (``True``) or
        not (``False``). In case ``returnerror=False``, ``None`` is return in
        spite of the axis if it is deemed to be irregular.
    Returns
    -------
    y : :obj:`np.ndarray`
        y-axis
    x : :obj:`np.ndarray`
        x-axis

    """
    # define a regular IL and XL axis
    yunique = np.unique(ys)
    xunique = np.unique(xs)

    ymin, ymax = min(yunique), max(yunique)
    xmin, xmax = min(xunique), max(xunique)

    dy = np.unique(np.diff(yunique))
    dx = np.unique(np.diff(xunique))
    if len(dy) > 0:
        warnings.warn('not unique dy={}...'.format(dy))
    if len(dx) > 0:
        warnings.warn('not unique dx={}...'.format(dy))

    dy = min(dy)
    dx = min(dx)

    if ints[0] > 0:
        if ints[0] <= dy:
            dy = ints[0]
        else:
            warnings.warn('ints[0]={} has been not used as '
                          'bigger than estimated dy={}...'.format(dy, ints[0]))
    if ints[1] > 0:
        if ints[1] <= dx:
            dx = ints[1]
        else:
            warnings.warn('ints[0]={} has been not used as '
                          'bigger than estimated dy={}...'.format(dx, ints[1]))
    if ns is not None:
        y = np.linspace(ymin, ymax, ns[0])
        x = np.linspace(xmin, xmax, ns[1])
    else:
        y = np.arange(ymin, ymax + dy, dy)
        x = np.arange(xmin, xmax + dx, dx)
        #print('y[-1], ymax', y[-1], ymax)
        #print('x[-1], xmax', x[-1], xmax)

        if y[-1] != ymax:
            if returnerror:
                raise ValueError('ymax={} is not part of y axis '
                                 '(last element y[-1]={})...'.format(ymax, y[-1]))
            #else:
            #    y = None
        if x[-1] != xmax:
            if returnerror:
                raise ValueError('xmax={} is not part of x axis '
                                 '(last element x[-1]={})...'.format(xmax, x[-1]))
            #else:
            #    x = None
    return y, x


def _unifygrid(y1, x1, y2, x2, fine=False, verb=False):
    """Create a common grid from grid1 (y1-x1) and grid2 (y2-x2)

    Parameters
    ----------
    y1 : :obj:`np.ndarray`
        y-axis for grid1
    x1 : :obj:`np.ndarray`
        x-axis for grid1
    y2 : :obj:`np.ndarray`
        y-axis for grid2
    x2 : :obj:`np.ndarray`
        x-axis for grid2
    fine : :obj:`bool`, optional
        Create common grid using smallest dx and dy (``True``)
        or biggest dx and dy (``False``)
    verb : :obj:`int`, optional
        Verbosity

    Returns
    -------
    y : :obj:`np.ndarray`
        y-axis
    x : :obj:`np.ndarray`
        x-axis
    isunified : :obj:`tuple`
        Flags indicating if input grids are the same unified grid (
        e.g.,  ``isunified=(True, False)`` if grid1 is the same as unified grid
        while grid2 is not)

    """
    ymin = min([y1.min(), y2.min()])
    xmin = min([x1.min(), x2.min()])

    ymax = max([y1.max(), y2.max()])
    xmax = max([x1.max(), x2.max()])

    if not fine:
        if verb:
            print('Create coarse grid...')
        dy = max([np.abs(y1[1]-y1[0]), np.abs(y2[1]-y2[0])])
        dx = max([np.abs(x1[1]-x1[0]), np.abs(x2[1]-x2[0])])
    else:
        if verb:
            print('Create fine grid...')
        dy = min([np.abs(y1[1] - y1[0]), np.abs(y2[1] - y2[0])])
        dx = min([np.abs(x1[1] - x1[0]), np.abs(x2[1] - x2[0])])

    # check if grid1 and/or grid2 are the same as unified grid
    isunified1 = True if dy == np.abs(y1[1] - y1[0]) and \
                         dx == np.abs(x1[1] - x1[0]) and \
                         y1.min() == ymin and \
                         y1.max() == ymax else False
    isunified2 = True if dy == np.abs(y2[1] - y2[0]) and \
                         dx == np.abs(x2[1] - x2[0]) and \
                         y2.min() == ymin and \
                         y2.max() == ymax else False
    isunified = (isunified1, isunified2)
    y = np.arange(ymin, ymax + dy, dy)
    x = np.arange(xmin, xmax + dx, dx)
    return y, x, isunified


def _fillgrid(ys, xs, zs, y, x, fine=0,
              mindist=-1, intmethod='linear', verb=False):
    """Fill previously created grid with surface x, y, and z samples

    ys : :obj:`np.ndarray`
        Values in y-direction
    xs : :obj:`np.ndarray`
        Values in x-direction
    zs : :obj:`np.ndarray`
        Values in z-direction
    y : :obj:`np.ndarray`
        y-axis of grid to fill
    x : :obj:`np.ndarray`
        x-axis of grid to fill
    fine : :obj:`int`, optional
         Fill grid at (0) available points using nearest neighour
         interpolation, (1) at available points but use only integer
         locations to avoid multiple locations to fit at same location,
         (2) at available points and interpolate/extrapolate to finer
         grid
    mindist : :obj:`bool`, optional
         Minimum distance allowed for extrapolation in case ``fine=2``
    intmethod : :obj:`bool`, optional
         Interpolation method used by :func:`scipy.interpolate.griddata`
    verb : :obj:`bool`, optional
         Verbosity

    Returns
    -------
    iys : :obj:`np.ndarray`
        Indeces for values in zs to be put in grid - y direction
    ixs : :obj:`np.ndarray`
        Indeces for values in zs to be put in grid - x direction
    data : :obj:`np.ndarray`
        Gridded surface
    mask : :obj:`np.ndarray`
        Mask
    """
    # interpret grid
    dy, dx = y[1]-y[0], x[1]-x[0]

    ymin, ymax = y[0], y[-1]
    xmin, xmax = x[0], x[-1]

    ny, nx = len(y), len(x)

    # indentify look-up table between y-x and tracenumber
    traceindeces = np.full((ny, nx), np.nan)

    if fine == 0:
        if verb:
            print('Fill grid at available points using '
                  'nearest neighour interpolation...')

        iys = np.round((ys - ymin) / dy).astype(np.int)
        ixs = np.round((xs - xmin) / dx).astype(np.int)

        traceindeces[iys, ixs] = np.arange(len(ys))
        mask = np.zeros_like(traceindeces)
        mask[np.isnan(traceindeces)] = 1
        data = np.zeros_like(traceindeces)
        data[iys, ixs] = zs

    elif fine == 1:
        if verb: print('Fill grid at available points but use only integer '
                       'locations to avoid multiple locations to fit at '
                       'same location...')

        # search for elements at integer positions in grid
        iys = (ys - ymin) / dy
        ixs = (xs - xmin) / dx

        # create list of valid indeces
        valid = (iys-np.round(iys) == 0) & (ixs-np.round(ixs) == 0)
        valid_idx = np.arange(len(ys))
        valid_idx = valid_idx[valid]

        iys = iys[valid].astype(np.int)
        ixs = ixs[valid].astype(np.int)
        traceindeces[iys, ixs] = valid_idx

        mask = np.zeros_like(traceindeces)
        mask[np.isnan(traceindeces)] = 1
        data = np.zeros_like(traceindeces)
        data[iys, ixs] = zs[valid]

    elif fine==2:
        if verb:
            print('Fill grid at available points and '
                  'interpolate to finer grid...')
        iys = np.round((ys - ymin) / dy).astype(np.int)
        ixs = np.round((xs - xmin) / dx).astype(np.int)
        #traceindeces[iys, ixs] = np.arange(len(ys))

        Y, X = np.meshgrid(y, x, indexing='ij')
        Y, X = Y.flatten(), X.flatten()
        data = spint.griddata((ys, xs), zs, (Y.ravel(), X.ravel()),
                              fill_value=np.nan, method=intmethod).reshape(ny, nx)

        if mindist == -1:
            mask = np.zeros_like(data)
        else:
            if verb:
                print('Create mask using minimum '
                      'distance of {0}...'.format(mindist))
            mask  = np.array([np.min(np.sqrt((iys-iy)**2+(ixs-ix)**2)) >= mindist
                              for iy in range(ny) for ix in range(nx)]).reshape(ny, nx)
            #mask = (np.min(np.sqrt((Y-ys[:,np.newaxis])**2 + (X-xs[:,np.newaxis])**2), axis=0)>=mindist).reshape(ny,nx)

    return iys, ixs, data, mask


class Surface:
    r"""Surface object.

    This object contains a surface. The surface can be either created from
    scratch given y and x locations (and inline and crossline) and data values
    or imported from dsg formatted file with 5 columns (IL, XL, X, Y, Z) or a
    plain text file with same  5 columns. Note that internally the surface will
    be always rearranged as a 2-dimensional array of shape
    :math:`[n_x \times n_y]` (also equivalent to :math:`[n_{IL} \times n_{XL}]`)

    Parameters
    ----------
    filename : :obj:`str`
        Name of files containing surfaces (use ``None`` to create an
        empty surface)
    format : :obj:`str`, optional
        Format of file containing surface (available options: ``dsg5_long``
        and ``plain_long``)
    decimals : :obj:`int`, optional
        Number of decimals in x-y coordinates
        (if ``None`` keep all)
    nanvals : :obj:`float`, optional
         Value to use to indicate non-valid numbers
         (if not provided default one from FORMATS dict in
         :class:`pysubsurface.objects.Surface` class is used)
    fillsurface : :obj:`bool`, optional
         Fill surface using previously computed grid (``True``)
         or compute grid from x-y and/or il-xl in the input file (``False``).
         The option ``fillsurface=True`` can be used in combination with
         :func:`ptc.objects.Surface.copy` when another surface with the same
         geometry has been already read. It is user responsability to use this
         option wisely.
    mastersurface : :obj:`pysubsurface.objects.Surface`, optional
         Surface to be used as template when ``fillsurface=True``
    finegrid : :obj:`int`, optional
         Fill grid at (0) available points using nearest neighour
         interpolation, (1) at available points but use only integer
         locations to avoid multiple locations to fit at same location,
         (2) at available points and interpolate/extrapolate to finer
         grid
    gridints : :obj:`tuple` or :obj:`list`, optional
         Sampling step in IL, and XL directions (relevant only if
         ``finegrid=2``, if not provided use min value found from
         interpreting the surface)
    mindist : :obj:`bool`, optional
         Minimum distance allowed for extrapolation in case ``finegrid=2``
    loadsurface : :obj:`bool`, optional
         Read surface and load it into ``surface.data
         during initialization (``True``) or not (``False``)
    kind : :obj:`str`, optional
        must be ``local``
    plotflag : :obj:`bool`, optional
         Quickplot
    verb : :obj:`bool`, optional
         Verbosity

    """
    def __init__(self, filename, format='dsg5_long', decimals=None,
                 nanvals=None, fillsurface=False, mastersurface=None,
                 finegrid=0, gridints=[-1, -1], mindist=-1,
                 loadsurface=True, kind='local',
                 plotflag=False, verb=False):
        self.filename = filename
        self._decimals = decimals
        self._fillsurface = fillsurface
        self._gridints = gridints.copy()
        self._mindist = mindist
        self._finegrid = finegrid
        self._kind = kind
        self._verb = verb

        # create empty structure to be filled by user-provided surface
        if filename is None:
            createsurface = True
            loadsurface = False
            format = 'dsg5_long'
            self.header = []
        else:
            createsurface=False

        # evaluate format
        if format in _FORMATS.keys():
            self.format = format
            self.format_dict = _FORMATS[self.format].copy()
            if nanvals is not None:
                self.format_dict['nanvals'] = nanvals
        else:
            raise ValueError('{} not contained in list of available'
                             ' formats=[{}]'.format(format,
                                                    ' '.join(_FORMATS)))

        # load surface
        if not createsurface:
            self._read_header()

            if self._fillsurface:
                self.mastersurface = mastersurface
                self.il = mastersurface.il.copy()
                self.xl = mastersurface.xl.copy()
                self.y = mastersurface.y.copy()
                self.x = mastersurface.x.copy()
            else:
                self.mastersurface = None

            # identify grid
            self._read_surface()

            # place values in grid
            if loadsurface:
                self.interpret_surface(plotflag=plotflag)

    def __str__(self):
        descr = 'Surface object:\n' + \
                '---------------\n' + \
                'Filename: {}\n'.format(self.filename) + \
                'Format: {}\n'.format(self.format) + \
                'nil: {}, nxl:{}\n'.format(self.nil, self.nxl) + \
                'Regular y and x axis: {}\n'.format(self._regxy) + \
                'il = {} - {} : {}\n'.format(self.il[0], self.il[-1], self.dil) + \
                'xl = {} - {} : {}\n'.format(self.xl[0], self.xl[-1], self.dxl)
        if self._regxy:
            descr = descr + \
                    'x = {} - {} : {}\n'.format(self.x[0], self.x[-1],
                                                 self.dx) + \
                    'y = {} - {} : {}\n'.format(self.y[0], self.y[-1],
                                                 self.dy)
        descr = descr + \
                'min = {0:.3f},\nmax = {1:.3f}\n'.format(np.nanmin(self.data),
                                                         np.nanmax(self.data)) + \
                'mean = {0:.3f},\nstd = {1:.3f}'.format(np.nanmean(self.data),
                                                        np.nanstd(self.data))
        return descr

    def __eq__(self, other):
        """Compare if surface is same as other"""
        if np.allclose(self.data, other.data, rtol=1e-5, equal_nan=True) and \
            np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            return True
        return False

    def __add__(self, other):
        """Add surface to other, output = surface + other"""
        summed = None
        if np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            summed = self.copy()
            summed.data += other.data
        return summed

    def __sub__(self, other):
        """Subtract other from surface, output = surface - other"""
        summed = None
        if np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            summed = self.copy()
            summed.data -= other.data
        return summed

    def __mul__(self, other):
        """Multiply other from surface, output = surface * other"""
        mul = None
        if np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            mul = self.copy()
            mul.data *= other.data
        return mul

    def __truediv__(self, other):
        """Divide other from surface, output = surface / other"""
        div = None
        if np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            div = self.copy()
            div.data /= other.data
        return div

    @property
    def shape(self):
        """Returns shape of the surface (i.e., data property)

        Returns
        -------
        shape : :obj:`tuple`
            Surface shape
        """
        self._shape = self.data.shape
        return self._shape

    def _read_header(self):
        """Read surface file header
        """
        if self.format_dict['skiprows_in'] > 0:
            with open(self.filename) as f:
                self.header = f.readlines()[0:self.format_dict['skiprows_in']]
        else:
            self.header = ['']
        return

    def _read_surface(self):
        """Read data and create grid
        """
        # load surface
        if self.format_dict['skiprows_end'] == 0:
            surface = \
                np.loadtxt(self.filename,
                           skiprows=self.format_dict['skiprows_in'],
                           delimiter=',',
                           comments=['#', 'EOD'])
        else:
            surface = \
                np.genfromtxt(self.filename,
                              skip_header=self.format_dict['skiprows_in'],
                              skip_footer=self.format_dict['skiprows_end'],
                              comments=['#', 'EOD'])

        # load inline and crossline, y and x values - note il=x and xl=y in this class convention
        self._ils_orig = surface[:, self.format_dict['ilpos']]
        self._xls_orig = surface[:, self.format_dict['xlpos']]
        self._xs_orig = surface[:, self.format_dict['xpos']]
        self._ys_orig = surface[:, self.format_dict['ypos']]
        self._zs_orig = surface[:, self.format_dict['zpos']]

        # remove non-valid values
        self._valid = np.arange(len(self._zs_orig), dtype=np.int)

        self._valid = self._valid[self._zs_orig != self.format_dict['nanvals']]
        self._ils = self._ils_orig[self._zs_orig != self.format_dict['nanvals']]
        self._xls = self._xls_orig[self._zs_orig != self.format_dict['nanvals']]
        self._xs = self._xs_orig[self._zs_orig != self.format_dict['nanvals']]
        self._ys = self._ys_orig[self._zs_orig != self.format_dict['nanvals']]
        self._zs = self._zs_orig[self._zs_orig != self.format_dict['nanvals']]

        # keep only certain number of decimals in _ys and _xs
        if self._decimals is not None:
            self._ys = np.round(self._ys, decimals=self._decimals)
            self._xs = np.round(self._xs, decimals=self._decimals)

        # prepare data and create grids to be filled with z values
        if not self._fillsurface:
            self.il, self.xl = _creategrid(self._ils, self._xls,
                                           self._gridints)
            self.dil = self.il[1] - self.il[0]
            self.dxl = self.xl[1] - self.xl[0]
            self.nil, self.nxl = len(self.il), len(self.xl)

            # extract info from original grid to find out how to resample y and
            # x axis when saving the surface in the new grid
            self._regxy = True # initial assumption, put False every time we realize this is the case
            if self._gridints[0] == -1 and self._gridints[1] == -1:
                self.y, self.x = _creategrid(self._ys, self._xs,
                                             ns=[self.nxl, self.nil],
                                             returnerror=False)
                if self.y is not None and self.x is not None:
                    self.dy = self.y[1] - self.y[0]
                    self.dx = self.x[1] - self.x[0]
                    self.ny, self.nx = len(self.y), len(self.x)
                else:
                    self._regxy = False

            if self._gridints[0] > 0 or self._gridints[1] > 0:
                self.ilorig, self.xlorig = _creategrid(self._ils, self._xls)
                self.dilorig = self.ilorig[1] - self.ilorig[0]
                self.dxlorig = self.xlorig[1] - self.xlorig[0]
                self.nilorig, self.nxlorig = len(self.ilorig), len(self.xlorig)
                self.ilsub = self.dilorig/self.dil
                self.xlsub = self.dilorig/self.dxl
                self.yorig, self.xorig = _creategrid(self._ys, self._xs,
                                                     ns=[self.nxl, self.nil],
                                                     returnerror=False)
                if self.yorig is not None and self.xorig is not None:
                    self.dyorig = self.yorig[1] - self.yorig[0]
                    self.dxorig = self.xorig[1] - self.xorig[0]
                    self.nyorig, self.nxorig = len(self.yorig), len(self.xorig)
                    self.y, self.x = _creategrid(self._ys, self._xs,
                                                 #ints=[self.dyorig / self.xlsub,
                                                 #      self.dxorig / self.ilsub],
                                                 ns=[self.nxl, self.nil],
                                                 returnerror=False)
                    if self.y is not None and self.x is not None:
                        self.dy = self.y[1] - self.y[0]
                        self.dx = self.x[1] - self.x[0]
                        self.nil, self.nxl = len(self.il), len(self.xl)
                        self._regxy = True
                    else:
                        self._regxy = False
                else:
                    self._regxy = False
                print('self._regxy', self._regxy)

    def create_surface(self, il, xl, data, y=None, x=None, plotflag=False):
        """Create surface from scratch by providing inline and crossline
        (and optionally y and x) axes and values

        Parameters
        ----------
        il : :obj:`np.ndarray`
            inline axis
        xl : :obj:`np.ndarray`
            crossline axis
        data : :obj:`np.ndarray`
            data (must have size ``nil x nxl``)
        y : :obj:`np.ndarray`, optional
            y axis (must have same size as ``xl``)
        x : :obj:`np.ndarray`, optional
            x axis (must have same size as ``il``)
        plotflag : :obj:`bool`, optional
            Quickplot

        """
        # define gridints to force save method to work on a regular grid
        self._gridints[0] = il[1] - il[0]
        self._gridints[1] = xl[1] - xl[0]

        self.il = il
        self.xl = xl
        self.nil, self.nxl = len(self.il), len(self.xl)
        self.dil = self.il[1] - self.il[0]
        self.dxl = self.xl[1] - self.xl[0]

        self.y = y
        self.x = x
        if self.y is not None:
            self.ny = len(self.y)
            self.dy = self.y[1] - self.y[0]
        else:
            self.ny = self.dy = None
        if self.x is not None:
            self.nx = len(self.x)
            self.dx = self.x[1] - self.x[0]
        else:
            self.nx = self.dx = None

        if x is None or y is None:
            self._regxy = False
        else:
            self._regxy = True
        self.data = data
        if isinstance(data, np.ndarray):
            self._regsurface=True
        else:
            self._regsurface=False
        if plotflag:
            self.view(cbar=True, title=self.filename)

    def interpret_surface(self, finegrid=None, plotflag=False):
        """Interpret surface by placing values in the available grid

        Parameters
        ----------
        finegrid : :obj:`int`, optional
            Fill grid at (0) available points using nearest neighour
            interpolation, (1) at available points but use only integer
            locations to avoid multiple locations to fit at same location,
            (2) at available points and interpolate/extrapolate to finer
            grid. Use ``Ǹone`` to keep the one defined in initialization
        plotflag : :obj:`bool`, optional
            Quickplot

        """
        if finegrid is not None:
            self._finegrid = finegrid

        #  fill grid with z values
        self._iils, self._ixls, data, mask = \
            _fillgrid(self._ils, self._xls,
                      self._zs, self.il,  self.xl,
                      fine=self._finegrid,
                      mindist=self._mindist,
                      verb=self._verb)

        # obtain mask from master surface if provided
        if self._fillsurface:
            if self.mastersurface._regsurface:
                mask = np.zeros_like(data)
            else:
                mask = self.mastersurface.data.mask

        if mask.sum() == 0:
            if self._verb:
                print('Create regular surface...')
            self._regsurface = True
            self.data = data
        else:
            if self._verb:
                print('Create irregular surface...')
            self._regsurface = False
            self.data = np_ma.masked_array(data, mask=mask)

        if plotflag:
            self.view(cbar=True, title=self.filename)

    def well_intersection(self, trajectory, thresh=50., other=None):
        """Well-Surface intersection

        Identify coordinates of well interesecting surface. Note that this
        routine currently assumes a single intersection point (if more than one
        only the first will be identified)

        Parameters
        ----------
        trajectory : :obj:`pysubsurface.objects.Trajectory`
            Well trajectory
        thresh : :obj:`float`, optional
            Threshold to satisfy to consider trajectory passing through the
            surface
        other : :obj:`pysubsurface.objects.Surface`, optional
            Other surface. If provided, the value of the surface is extracted
            at the specified x-y location

        Returns
        -------
        interesection : :obj:`list`
            x-y-z coordinates of intersection (returns ``None`` if well
            trajectory does not interesect the surface)
        othervalue : :obj:`float`
            value of other surface at x-y locations of intersection
            (returns ``None`` if well trajectory does not interesect the surface)

        """
        othervalue = None
        trajectory = trajectory.df.copy()
        xwell = trajectory['X Absolute'].values
        ywell = trajectory['Y Absolute'].values
        zwell = trajectory['TVDSS'].values

        if self._regxy:
            fsurface = RegularGridInterpolator((self.x, self.y), self.data,
                                               bounds_error=False)
            if other is not None:
                fother = RegularGridInterpolator((other.x, other.y), other.data,
                                                 bounds_error=False)
            surface_atwellxy = fsurface(np.vstack([xwell, ywell]).T)
            zdiff = surface_atwellxy - zwell

            # find closest point in the trajectory (minimal vertical distance to surface)
            itraj = np.argmin(np.abs(zdiff))
            if itraj == 0:
                if zdiff[0] > thresh:
                    return None, None
                else:
                    itraj += 1
            elif itraj == len(trajectory) - 1:
                if zdiff[-1] > thresh:
                    return None, None
                else:
                    itraj -= 1

            # refine estimate ensuring zero distance with surface
            if trajectory.iloc[itraj]['TVDSS'] > surface_atwellxy[itraj]:
                traj = [trajectory.iloc[itraj - 1], trajectory.iloc[itraj]]
                zdiff_traj = zdiff[itraj - 1:itraj + 1]
            else:
                traj = [trajectory.iloc[itraj], trajectory.iloc[itraj + 1]]
                zdiff_traj = zdiff[itraj:itraj + 2]

            traj_x = [traj[0]['X Absolute'], traj[1]['X Absolute']]
            traj_y = [traj[0]['Y Absolute'], traj[1]['Y Absolute']]

            xcross = np.interp(0., zdiff_traj, traj_x)
            ycross = np.interp(0., zdiff_traj, traj_y)
            zcross = fsurface([xcross, ycross])[0]
            if other is not None:
                othervalue = fother([xcross, ycross])[0]
        else:
            raise NotImplementedError('Well-surface intersection is currently'
                                      'not available for surfaces with '
                                      'irregularly sampled x/y axes')
            # Use _findclosest_point_surface
        cross = [xcross, ycross, zcross]
        return cross, othervalue

    def extract_around_points(self, points, extent=(0, 0)):
        """Extract values in surface around points and return averages

        Parameters
        ----------
        points : :obj:`np.ndarray`
            Points around which surface extraction is performed. Points are
            passed as 2-d array of size :math:`\lbrack n_{points} x 2 \rbrack`
            with x in first column and y in second column
        extent : :obj:`tuple`, optional
            half-size of extraction window in x and y directions

        Returns
        -------
        points_ave : :obj:`np.ndarray`
            Averages around points

        """
        npoints = points.shape[0]
        points_ave = np.full(npoints, np.nan)
        for i in range(npoints):
            if not np.isnan(points[i, 0]) and not np.isnan(points[i, 1]):
                ilpoint, xlpoint = _findclosest_point_surface(points[i], self)
                if ilpoint > self.il[0] and ilpoint < self.il[-1] and \
                        xlpoint > self.xl[0] and xlpoint < self.xl[-1]:
                    ilmin = int(np.max([self.il[0], ilpoint - extent[0]]))
                    xlmin = int(np.max([self.xl[0], xlpoint - extent[1]]))
                    ilmax = int(np.max([self.il[0], ilpoint + extent[0]]))
                    xlmax = int(np.max([self.il[0], xlpoint + extent[1]]))
                    iilmin = ((ilmin - self.il[0]) / self.dil).astype(int)
                    ixlmin = ((xlmin - self.xl[0]) / self.dxl).astype(int)
                    iilmax = ((ilmax - self.il[0]) / self.dil).astype(int)
                    ixlmax = ((xlmax - self.xl[0]) / self.dxl).astype(int)
                    points_around = self.data[iilmin:iilmax+1, ixlmin:ixlmax+1]
                    if isinstance(self.data, np_ma.core.MaskedArray):
                        points_around = points_around[~points_around.mask].flatten()
                    else:
                        points_around = points_around.flatten()
                    if len(points_around) > 0:
                        points_ave[i] = np.mean(points_around)
        return points_ave

    def subsample(self, jil, jxl):
        """Subsample surface by factor in ilines and/or xlines.

        The surface is subsampled by factor in inlines and/or crossline and a
        new surface is returned. Note that if the original surface has irregular
        x and y axes, the returned surface cannot be saved as it is not
        possible to infer the values at original locations after such
        subsampling

        Parameters
        ----------
        jil : :obj:`int`
            Subsampling factor in iline axis
        jxl : :obj:`bool`, optional
            Subsampling factor in crossline axis

        Returns
        -------
        hor_subsampled : :obj:`pysubsurface.objects.Surface`
            Subsampled surface

        """
        hor_subsampled = self.copy()
        hor_subsampled.il = self.il[::jil]
        hor_subsampled.dil = self.dil * jil
        hor_subsampled.nil = len(hor_subsampled.il)
        hor_subsampled.xl = self.xl[::jxl]
        hor_subsampled.dxl = self.dxl * jxl
        hor_subsampled.nil = len(hor_subsampled.xl)

        if hor_subsampled._regxy:
            hor_subsampled.x = self.x[::jil]
            hor_subsampled.dx = self.dx * jil
            hor_subsampled.nx = len(hor_subsampled.x)
            hor_subsampled.y = self.y[::jxl]
            hor_subsampled.dy = self.dy * jxl
            hor_subsampled.ny = len(hor_subsampled.y)
        hor_subsampled.data = self.data[::jil, ::jxl]
        return hor_subsampled

    def same_grid(self, other):
        if np.array_equal(self.il, other.il) and \
            np.array_equal(self.xl, other.xl) and \
            np.array_equal(self.y, other.y) and \
            np.array_equal(self.x, other.x):
            samegrid = True
        else:
            samegrid = False
        return samegrid

    def resample_surface_to_grid(self, other, intmethod='linear'):
        """Convert other surface to current surface grid.

        This method interpolates ``other`` to the IL-XL grid of the object.

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Surface`
            Other surface to be resampled from the current surface
        intmethod : :obj:`bool`, optional
         Interpolation method used by :func:`scipy.interpolate.griddata`

        Returns
        -------
        hor_slave : :obj:`pysubsurface.objects.Surface`
            Surface converted into grid of object

        """
        # Check if the surfaces are already on the same grid and return other
        if self.same_grid(other):
            return other

        il = self.il
        xl = self.xl
        il_slave = other.il
        xl_slave = other.xl

        nil, nxl = il.size, xl.size
        IL, XL = np.meshgrid(il, xl, indexing='ij')
        IL, XL = IL.flatten(), XL.flatten()
        hor_slave = self.copy(empty=True)

        # Interpolate to self grid and save to new Surface object
        if isinstance(other.data, np_ma.MaskedArray):
            other_data = other.data.data
        else:
            other_data = other.data
        grid_slave = RegularGridInterpolator((il_slave, xl_slave), other_data,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             method=intmethod)(
            np.vstack([IL, XL]).T).reshape(nil, nxl)
        if isinstance(self.data, np_ma.MaskedArray) and \
                isinstance(other.data, np_ma.MaskedArray):
            mask_slave = \
                RegularGridInterpolator((il_slave, xl_slave),
                                        other.data.mask.astype(np.int),
                                        bounds_error=False, fill_value=np.nan,
                                        method=intmethod)(
                    np.vstack([IL, XL]).T).reshape(nil, nxl)
            hor_slave.data.data[:] = grid_slave
            hor_slave.data.mask[:] = mask_slave
        else:
            hor_slave.data[:] = grid_slave
        return hor_slave

    def sum_surface_different_grid(self, other, intmethod='linear',
                                   subtract=False):
        """Subtract surface from a different grid

        This method interpolates ``other`` to the IL-XL grid of the object and
        returns the sum between the two surfaces

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Surface`
            Other surface to be subtracted from the current surface
        intmethod : :obj:`bool`, optional
            Interpolation method used by :func:`scipy.interpolate.griddata`
        subtract : :obj:`bool`, optional
            Subtract instead of sum

        Returns
        -------
        diff : :obj:`pysubsurface.objects.Surface`
            Difference Surface

        """
        other_regriddded = self.resample_surface_to_grid(other, intmethod=intmethod)
        if subtract:
            sum = self - other_regriddded
        else:
            sum = self + other_regriddded
        return sum

    def subtract_surface_different_grid(self, other, intmethod='linear'):
        """Subtract surface from a different grid

        This method interpolates ``other`` to the IL-XL grid of the object and
        returns the difference between the two surfaces

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Surface`
            Other surface to be subtracted from the current surface
        intmethod : :obj:`bool`, optional
         Interpolation method used by :func:`scipy.interpolate.griddata`

        Returns
        -------
        diff : :obj:`pysubsurface.objects.Surface`
            Difference Surface

        """
        return self.sum_surface_different_grid(other, intmethod=intmethod, subtract=True)

    def multiply_surface_different_grid(self, other, intmethod='linear',
                                        divide=False):
        """Multiply surface from a different grid

        This method interpolates ``other`` to the IL-XL grid of the object and
        returns the product between the two surfaces

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Surface`
            Other surface to be subtracted from the current surface
        intmethod : :obj:`bool`, optional
            Interpolation method used by :func:`scipy.interpolate.griddata`
        divide : :obj:`bool`, optional
            Divide instead of multiplying

        Returns
        -------
        mul : :obj:`pysubsurface.objects.Surface`
            Product Surface

        """
        other_regriddded = self.resample_surface_to_grid(other, intmethod=intmethod)
        if divide:
            mul = self / other_regriddded
        else:
            mul = self * other_regriddded
        return mul

    def divide_surface_different_grid(self, other, intmethod='linear'):
        """Divide surface from a different grid

        This method interpolates ``other`` to the IL-XL grid of the object and
        returns the ratio between the two surfaces

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Surface`
            Other surface to be subtracted from the current surface
        intmethod : :obj:`bool`, optional
         Interpolation method used by :func:`scipy.interpolate.griddata`

        Returns
        -------
        div : :obj:`pysubsurface.objects.Surface`
            Difference Surface

        """
        return self.multiply_surface_different_grid(other, intmethod=intmethod,
                                                    divide=True)

    def copy(self, empty=False):
        """Return a copy of the object.

        Parameters
        ----------
        empty : :obj:`bool`
            Copy input data (``True``) or just create an empty data (``False``)

        Returns
        -------
        surfacecopy : :obj:`pysubsurface.objects.Surface`
            Copy of Surface object

        """
        surfacecopy = copy.deepcopy(self)
        if empty:
            if isinstance(self.data, np_ma.core.MaskedArray):
                surfacecopy.data.data[:] = 0.
            else:
                surfacecopy.data = np.zeros((self.nil, self.nxl))
        return surfacecopy

    def save(self, outputfilename, newsurfacename='', format=None, fmt=None,
             regulargrid=False, nanvals=None, verb=False):
        """Save object to file

        Parameters
        ----------
        outputfilename : :obj:`str`
            Name of output file
        newsurfacename : :obj:`str`
            Name of surface to use in file header
        format : :obj:`str`, optional
            Format of file to save (if ``None`` replicate that of input file)
        fmt : :obj:`str` , optional
            Format of data to save (if ``None`` use default fmt of format)
        regulargrid : :obj:`bool`, optional
            Save in regular grid (``True``) or original grid (``False``)
        nanvals : :obj:`bool`, optional
            Value to write to indicate non-valid values
            (if not provided use default one from _FORMATS dict)
        verb : :obj:`bool`, optional
            Verbosity

        """
        if format is None:
            format = self.format
        if fmt is None:
            fmt = _FORMATS[self.format]['fmt']

        # catch scenarios where IL and XL axes are changed and y and x
        # axes cannot be inferred
        if not self._regxy:
            raise ValueError('Cannot export surface as y and/or x are not in'
                             'regular axis and IL and XL were previously set'
                             'to a finer scale than in input file...')

        # check that original grid exists (if not revert to regular grid)
        if not regulargrid:
            zs_orig = self._zs_orig.copy()
            ys_orig = self._ys_orig.copy()
            xs_orig = self._xs_orig.copy()
            ils_orig = self._ils_orig.copy()
            xls_orig = self._xls_orig.copy()
        data = self.data.copy()

        if not regulargrid:
            if self._gridints[0] > 0 or self._gridints[1] > 0:
                warnings.warn('Cannot save surface in irregular '
                              'grid because gridints were forced by user, '
                              'revert to regulargrid=True...')
                regulargrid = True

        # define nanvals if not provided
        if nanvals is None:
            nanvals = self.format_dict['nanvals']
        if verb:
            print('Use nanvalue = {}'.format(nanvals))

        # initialize _zs_orig with non-valid values
        if not regulargrid:
            zs_orig = zs_orig*0 + nanvals

        # when working with masked arrays, put nanvalues where mask is True
        if isinstance(data, np_ma.core.MaskedArray):
            data.data[self.data.mask] = nanvals

        # recreate ils_orig, xls_orig, ys_orig, xs_orig and zs_orig to save
        if not regulargrid:
            if verb:
                print('Save in original grid...')
            if self._finegrid == 0 or self._finegrid == 1:
                zs_orig[self._valid] = data[self._iils, self._ixls]
            #elif self._finegrid == 1:
            #    IL, XL = np.meshgrid(self.il, self.xl, indexing='ij')
            #    if self._regsurface:
            #        ils_orig, xls_orig = IL.ravel(), XL.ravel()
            #        zs_orig = data.flatten()
            #    else:
            #        ils_orig = IL[~data.mask]
            #        xls_orig = XL[~data.mask]
            #        zs_orig = data[~self.data.mask]
        else:
            if verb:
                print('Save in regular grid...')
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            IL, XL = np.meshgrid(self.il, self.xl, indexing='ij')

            print('Save in regular grid...')

            if not self._regsurface:
                mask = data.mask
                data = data.data
                data[mask] = nanvals
            ils_orig, xls_orig = IL.ravel(), XL.ravel()
            ys_orig, xs_orig = Y.flatten(), X.flatten()
            zs_orig = data.ravel()

        # modify header
        if format == 'dsg5_long':
            if len(self.header) > 0:
                header_new = self.header.copy()
                if header_new[-1][-1:] == '\n':
                    header_new[-1] = header_new[-1][:-1] # remove \n from last element

            header_new[66] = \
                ' '.join(['#Horizon_name____________->{}'
                          '\n'.format(newsurfacename)])
        elif format == 'plain_long':
            header_new = ''
        else:
            raise ValueError('Format provided not valid....')
        # save
        np.savetxt(outputfilename,
                   np.vstack([ils_orig, xls_orig,
                              xs_orig, ys_orig, zs_orig]).T,
                   header=''.join(header_new),
                   fmt=fmt, comments='')

    def view(self, ax=None, which='ilxl', alpha=1., transparency=[],
             interp=None, clip=1., clim=[], cmap='seismic', ncountour=0,
             lwcountour=1., countourlabels=False, cbar=False, cbartitle='',
             chist=False, nhist=11, flipaxis=False, originlower=False,
             flipy=False, flipx=False, axiskm=False, scalebar=False,
             figstyle='white', figsize=(8,6), title='',
             titlesize=12, savefig=None):
        """Quick visualization of Surface object.

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handle (if ``None`` draw a new figure)
        which : :obj:`str`, optional
            Visualize surface in IL-XL coordinates (``ilxl``) or
            ``yx`` y-x coordinates
        alpha : :obj:`float`, optional
            Surface transparency
        transparency : :obj:`pysubsurface.objects.Surface`, optional
            Space variant transparency based on provided surface values
        interp : :obj:`str`, optional
            Interpolation method for imshow display
        clip : :obj:`float`, optional
             Clip to apply to colorbar limits (``vmin`` and ``vmax``)
        clim : :obj:`float`, optional
             Colorbar limits (if ``None`` infer from data and
             apply ``clip`` to those)
        cmap : :obj:`str`, optional
             Colormap
        ncontour : :obj:`int`, optional
             Number of contour levels to add to plot
        lwcontour : :obj:`float`, optional
             Linewidth of contour lines to add to plot
        countourlabels : :obj:`int`, optional
             Add labels to contours
        cbar : :obj:`bool`, optional
             Show colorbar
        cbartitle : :obj:`bool`, optional
             Colorbar title
        chist : :obj:`bool`, optional
             Show histogram
        nhist : :obj:`int`, optional
             Number of bins for histogram of values in data
        flipaxis : :obj:`bool`, optional
             Flip x and y axis (``True``) or not (``False``)
        originlower : :obj:`bool`, optional
             Origin at bottom-left (``True``) or top-left (``False``)
        flipy : :obj:`bool`, optional
             flip y axis (``True``) or not (``False``)
        flipx : :obj:`bool`, optional
             flip x axis (``True``) or not (``False``)
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)
        scalebar : :obj:`bool`, optional
            Show scalebar
        figstyle : :obj:`str`, optional
            Style of figure (``white`` or ``black``)
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
        axisscale = 1000. if axiskm else 1.
        labelcol = 'black' if figstyle == 'white' else 'white'

        if nhist % 2 == 0:
            nhist += 1

        if len(clim) == 0:
            clim = [clip * np.nanmin(self.data), clip * np.nanmax(self.data)]

        if ax is None:
            if chist:
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(1, 4)
                ax = plt.subplot(gs[0, 0:3])
            else:
                fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        # choose coordinates to visualize and reorganize data axes if flipped
        if which == 'ilxl':
            if not flipaxis:
                x, y = self.il.copy()/axisscale, self.xl.copy()/axisscale
                data = self.data.T.copy()
            else:
                x, y = self.xl.copy()/axisscale, self.il.copy()/axisscale
                data = self.data.copy()
        else:
            if self._regxy:
                if not flipaxis:
                    x, y = self.x.copy()/axisscale, self.y.copy()/axisscale
                    data = self.data.T.copy()
                else:
                    x, y = self.y.copy()/axisscale, self.x.copy()/axisscale
                    data = self.data.copy()
            else:
                if not flipaxis:
                    x, y = self._xs / axisscale, self._ys / axisscale
                else:
                    x, y = self._ys / axisscale, self._xs / axisscale
                data = self._zs

        # plotting
        if which == 'yx' and self._regxy is False:
            # consider case when x-y are not in a grid, scatter plot
            im = ax.scatter(x, y, c=data, edgecolor='none',
                            cmap=cmap, vmax=clim[1], vmin=clim[0])
        else:
            # add transparency if provided
            if type(transparency)!=list:
                # create data as RGB
                data = Normalize(clim[0], clim[1], clip=True)(data)
                cmap = cm.get_cmap(cmap) if type(cmap)==str else cmap
                data = cmap(data)
                # create transparency
                if type(transparency.data) == np_ma.core.MaskedArray:
                    transp = transparency.data.data
                    transp[transparency.data.mask]=0.
                else:
                    transp = transparency.data
                data[..., -1] = transp

            if originlower:
                im = ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1],
                               origin='lower', interpolation=interp, alpha=alpha,
                               extent=(x[0], x[-1], y[0], y[-1]))
                if ncountour>0:
                    ax_im = ax.axis()
                    cs = ax.contour(data, ncountour, colors='k', origin='lower',
                                    linewidths=lwcountour,
                                    extent=(x[0], x[-1], y[0], y[-1]))
                    if countourlabels:
                        ax.clabel(cs, inline=1, fontsize=10)
                    ax.axis(ax_im)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1],
                               interpolation=interp,
                               extent = (x[0], x[-1], y[-1], y[0]))
                if ncountour>0:
                    ax_im = ax.axis()
                    cs = ax.contour(data, ncountour,
                                    linewidths=lwcountour,
                                    colors='k' if figstyle == 'white' else 'w',
                                    alpha=0.7, extent=(x[0], x[-1], y[0], y[-1]))
                    if countourlabels:
                        ax.clabel(cs, inline=1, fontsize=10)
                    ax.axis(ax_im)
            if flipy:
                ax.invert_yaxis()
            if flipx:
                ax.invert_xaxis()
        if which == 'ilxl':
            if flipaxis:
                ax.set_xlabel('XL',color=labelcol)
                ax.set_ylabel('IL',color=labelcol)
                ax.set_title(title, fontsize=titlesize,
                             color=labelcol, weight='bold')
            else:
                ax.set_xlabel('IL',color=labelcol)
                ax.set_ylabel('XL',color=labelcol)
                ax.set_title(title, fontsize=titlesize,
                             color=labelcol, weight='bold')
        else:
            if flipaxis:
                print('here')
                ax.set_xlabel('Y (%sm)' % ('k' if axiskm else ''), color=labelcol)
                ax.set_ylabel('X (%sm)' % ('k' if axiskm else ''), color=labelcol)
                ax.set_title(title, fontsize=titlesize,
                             color=labelcol, weight='bold')
            else:
                ax.set_xlabel('X (%sm)' % ('k' if axiskm else ''), color=labelcol)
                ax.set_ylabel('Y (%sm)' % ('k' if axiskm else ''), color=labelcol)
                ax.set_title(title, fontsize=titlesize,
                             color=labelcol, weight='bold')

        if chist and fig is not None:
            data_hist = data[~np.isnan(data)].flatten() if type(data) == np.ndarray \
                else data.data[~data.mask].flatten()
            data_hist[data_hist<clim[0]] = clim[0]
            data_hist[data_hist>clim[1]] = clim[1]

            d_min = clim[0]
            d_max = clim[1]

            # Add the colorbar
            cbaxes = fig.add_axes([0.79, 0.55, 0.03, 0.3])
            cb = plt.colorbar(im, cax=cbaxes)
            cb.set_label(cbartitle, rotation=270,
                         color=labelcol, labelpad=20, fontsize=14)

            # Add the histogram
            histax = fig.add_axes([0.75, 0.55, 0.04, 0.3])

            bins = np.linspace(d_min, d_max, nhist + 1)
            hist, bins = np.histogram(data_hist.ravel(), bins=bins)
            plt.barh((bins[:-1] + bins[1:]) / 2, hist, align='center',
                     height=bins[1] - bins[0],
                     color=plt.cm.get_cmap(cmap)(np.linspace(0, 1, nhist)),
                     edgecolor=[labelcol] * nhist, lw=[0.5] * nhist)
            histax.invert_xaxis()
            histax.set_ylim([d_min, d_max])
            histax.axis('off')
        else:
            if cbar:
                cb = plt.colorbar(im, ax=ax, shrink=0.7)
                cb.set_label(cbartitle, rotation=270,
                             color=labelcol, labelpad=20, fontsize=14)
            else:
                cb = None

        if scalebar:
            xlen = np.diff(np.array(ax.get_xlim()))[0]
            scalebar = ScaleBar(axisscale,
                                fixed_value=np.round(xlen/5),
                                fixed_units='km' if axiskm else 'm',
                                color='w' if figstyle=='black' else 'k',
                                box_color='k' if figstyle=='black' else 'w',
                                frameon=False)
            ax.add_artist(scalebar)

        # styling
        if fig is not None and figstyle == 'black':
            _set_black(fig, ax, cb=cb)

        ax.grid(True, linestyle='dashed', alpha=0.5)
        ax.axis('tight')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if savefig is not None:
            plt.savefig(savefig, dpi=300,
                        facecolor=fig.get_facecolor(), transparent=True)
        return fig, ax


class SurfacePair:
    """SurfacePair object.

    This object contains a pair of surfaces in unified grid layout. Note that
    surfaces do not need to share the same grid but if this is the case
    y and x locations of each surface will be set to ``None``.

    Parameters
    ----------
    filename1 : :obj:`str`
        Name of files containing surface1
    filename2 : :obj:`str`
        Name of files containing surface2
    format : :obj:`str`, optional
        Format of file containing surface (available options: ``dsg5_long``)
    decimals : :obj:`int`, optional
        Number of decimals in x-y coordinates
        (if ``None`` keep all)
    nanvals : :obj:`float`, optional
         Value to use to indicate non-valid numbers
         (if not provided default one from FORMATS dict in
         :class:`pysubsurface.objects.Surface` class is used)
    finegrid : :obj:`int`, optional
         Create common grid using smallest dx and dy and interpolate surfaces
         (``True``) or use largest dx and dy and decimate surfaces (``False``)
    gridints : :obj:`tuple` or :obj:`list`, optional
         Sampling step in IL, and XL directions (relevant only if
         ``finegrid=True``, if not provided use min value found from
         interpreting the surfaces)
    mindist : :obj:`bool`, optional
         Minimum distance allowed for extrapolation in case ``finegrid=2``
    loadsurfaces : :obj:`bool`, optional
         Read surfaces and load it into ``surface1.data and ``surface2.data``
         variables during initialization (``True``) or not (``False``)
    plotflag : :obj:`bool`, optional
         Quickplot
    verb : :obj:`bool`, optional
         Verbosity

    """
    def __init__(self, filename1, filename2, format='dsg5_long',
                 decimals=None, nanvals=False, finegrids=False,
                 gridints=[-1, -1], mindist=-1,
                 loadsurfaces=True, plotflag=False, verb=False):

        self.filename1 = filename1
        self.filename1 = filename2
        self._decimals = decimals
        self._nanvals = nanvals
        self._finegrid = finegrids
        self._gridints = gridints.copy()
        self._mindist = mindist
        self._verb = verb

        if format in _FORMATS.keys():
            self.format = format
            self.format_dict = _FORMATS[self.format].copy()
        else:
            raise ValueError('{} not contained in list of available'
                             ' formats=[{}]'.format(format,
                                                    ' '.join(_FORMATS)))
        # create surface objects
        if not self._finegrid:
            self._finegrid_load=1
        else:
            self._finegrid_load=2
        self.surface1 = Surface(filename=filename1, format=self.format,
                                decimals=self._decimals,
                                nanvals=self._nanvals,
                                finegrid=self._finegrid_load,
                                gridints=self._gridints,
                                mindist=self._mindist, loadsurface=False,
                                plotflag=plotflag, verb=verb)
        self.surface2 = Surface(filename=filename2, format=self.format,
                                decimals=self._decimals,
                                nanvals=self._nanvals,
                                finegrid=self._finegrid_load,
                                gridints=self._gridints,
                                mindist=self._mindist, loadsurface=False,
                                plotflag=plotflag, verb=verb)

        # check if surface1 and surface2 have same grid
        if np.array_equal(self.surface1.il, self.surface2.il) and \
                np.array_equal(self.surface1.xl, self.surface2.xl):
            il, xl = self.surface1.il, self.surface1.xl
            self.isunified = [True, True]
        else:
            # unify grid
            il, xl, self.isunified = \
                _unifygrid(self.surface1.il, self.surface1.xl,
                           self.surface2.il, self.surface2.xl,
                           fine=self._finegrid, verb=self._verb)

        # force both surfaces to be in new grid if user has specified gridints
        if self._gridints[0] != -1 or self._gridints[1] != -1:
            self.isunified = [False, False]

        # identify which grid has been refined
        if self.isunified[0]:
            self.surface1._finegrid = 0
        if self.isunified[1]:
            self.surface2._finegrid = 0

        # assign new axis to surfaces
        self.surface1.il, self.surface1.xl = il, xl
        self.surface2.il, self.surface2.xl = il, xl

        self.surface1.dil = self.surface1.il[1] - self.surface1.il[0]
        self.surface1.dxl = self.surface1.xl[1] - self.surface1.xl[0]

        self.surface1.nil, self.surface1.nxl = len(self.surface1.il), len(self.surface1.xl)
        self.surface2.nil, self.surface2.nxl = self.surface1.nil, self.surface1.nxl

        # use y and x axis from other surface if isunified=True or set to None
        # whenor neither surface has a grid equivalent to the unified one
        if not self.isunified[0]:
            if self.isunified[1]:
                self.surface1.y, self.surface1.x = self.surface2.y, self.surface2.x
                self.surface1._regxy = True
            else:
                self.surface1.y, self.surface1.x = None, None
                self.surface1._regxy = False
        if not self.isunified[1]:
            if self.isunified[0]:
                self.surface2.y, self.surface2.x = self.surface1.y, self.surface1.x
                self.surface2._regxy = True
            else:
                self.surface2.y, self.surface2.x = None, None
                self.surface2._regxy = False

        # load surfaces by placing z values in grid
        if loadsurfaces:
            self.interpret_surfaces(plotflag=plotflag)
        return

    def interpret_surfaces(self, plotflag=False):
        """Interpret surfaces

        Parameters
        ----------
        plotflag : :obj:`bool`, optional
         Quickplot

        """
        if self._verb:
            print('Load surface1...')
        self.surface1.interpret_surface(plotflag=plotflag)
        if self._verb:
            print('Load surface2...')
        self.surface2.interpret_surface(plotflag=plotflag)

        # find out which surface is already on unified grid and copy the mask
        # or create mask using elements that are nan in either surface when
        # none of the surfaces is on unified grid
        if not self.isunified[0] and not self.isunified[1]:
            if np.isnan(self.surface1.data).sum() > 0 or \
                    np.isnan(self.surface2.data).sum() > 0:
                self.surface1._regsurface = False
                self.surface2._regsurface = False
                # create mask and masked surfaces
                if isinstance(self.surface1.data, np.ndarray):
                    mask = np.isnan(self.surface1.data) & np.isnan(self.surface2.data)
                    self.surface1.data = np_ma.masked_array(self.surface1.data,
                                                            mask=mask)
                if isinstance(self.surface2.data, np.ndarray):
                    mask = np.isnan(self.surface1.data) & np.isnan(self.surface2.data)
                    self.surface2.data = np_ma.masked_array(self.surface2.data,
                                                            mask=mask)
                mask = self.surface1.data.mask & self.surface2.data.mask & \
                       np.isnan(self.surface1.data) & np.isnan(self.surface2.data)
                self.surface1.data = np_ma.masked_array(self.surface1.data.data,
                                                        mask=mask)
                self.surface2.data = np_ma.masked_array(self.surface2.data.data,
                                                        mask=mask)
        elif not self.isunified[0] or not self.isunified[1]:
            if self.isunified[0]:
                if isinstance(self.surface1.data, np.ndarray):
                    self.surface1.data = \
                        np_ma.masked_array(self.surface1.data,
                                           mask = np.zeros(self.surface1.data.shape))
                self.surface2._regsurface = False
                self.surface1.data = \
                    np_ma.masked_array(self.surface1.data.data,
                                       mask=self.surface2.data.mask)
                # add mask elements to where nan values are located in interpolated grid
                self.surface1.data.mask[np.isnan(self.surface2.data.data)] = True
                self.surface2.data.mask[np.isnan(self.surface2.data.data)] = True
            elif self.isunified[1]:
                if isinstance(self.surface2.data, np.ndarray):
                    self.surface2.data = \
                        np_ma.masked_array(self.surface2.data,
                                           mask=np.zeros(self.surface2.data.shape))
                self.surface1._regsurface=False
                self.surface2.data = np_ma.masked_array(self.surface2.data.data,
                                                        mask=self.surface1.data.mask)
                # add mask elements to where nan values are located in interpolated grid
                self.surface1.data.mask[np.isnan(self.surface1.data.data)] = True
                self.surface2.data.mask[np.isnan(self.surface1.data.data)] = True
