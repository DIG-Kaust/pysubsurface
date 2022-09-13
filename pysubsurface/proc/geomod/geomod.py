import numpy as np
import numpy as np
import numpy.ma as np_ma
import verde as vd

from scipy.interpolate import Rbf
from sklearn.gaussian_process import GaussianProcessRegressor

from pysubsurface.objects.surface import _fillgrid
from pysubsurface.visual.cmap import *


def surface_from_wells(wells, interval, level, property, tops=None,
                       addthickness=False, surface=None,
                       xin=None, yin=None, xend=None, yend=None,
                       nx=None, ny=None, intmethod='rbf', **kwargs_interp):
    """Estimate a surface from a number of well picks using various
    interpolation methods

    Parameters
    ----------
    wells : :obj:`dict`
        Set of wells to use for creating a surface
    interval : :obj:`str`
        Name of interval
    level : :obj:`str`, optional
        Level to analyze
    property : :obj:`str`, optional
        Name of property to extract from self.intervals dataframe
    tops : :obj:`list`
        Top and base picks for custom interval (i.e., not available in stratigraphic column)
    addthickness : :obj:`bool`, optional
            Find properties in the middle of an interval by adding thickness
            (``True``) or  not (``False``)
    surface : :obj:`pysubsurface.objects.Surface`, optional
        Surface used to indentify the edges of the area of interest (provide
        ``xin``, ``yin``, ``xend``, ``yend``, ``nx``, ``ny`` alternatively)
    xin : :obj:`float`, optional
        Initial value in y direction (not used if surface is provided)
    yin : :obj:`float`, optional
        Initial value in x direction (not used if surface is provided)
    xend : :obj:`float`, optional
        Final value in x direction (not used if surface is provided)
    yend : :obj:`float`, optional
        Final value in y direction (not used if surface is provided)
    nx : :obj:`int`, optional
        Number of elements in x direction (not used if surface is provided)
    ny : :obj:`int`, optional
        Number of elements in x direction (not used if surface is provided)
    intmethod: :obj:`str`, optional
        Method used for interpolation:

        * ``rbf``: :class:`scipy.interpolate.Rbf`
        * ``rbf-sklearn``: :class:`sklearn.gaussian_process.GaussianProcessRegressor`
        * ``verde``: :class:`verde.Spline`
    kwargs_interp : :obj:`dict`, optional
        Additional parameters for interpolation routines

    Returns
    -------
    intsurface : :obj:`pysubsurface.objects.Surface`, optional
        Interpolated surface from well properties

    """
    wellnames = list(wells.keys())

    # extract formation property and its location in x-y coordinates
    xcoords = np.full(len(wellnames), np.nan)
    ycoords = np.full(len(wellnames), np.nan)
    props = np.full(len(wellnames), np.nan)
    for iwell, wellname in enumerate(wellnames):
        if wells[wellname].intervals is not None:
            xcoord, ycoord, prop = \
                wells[wellname].extrac_prop_in_interval(interval, level,
                                                        property, tops=tops,
                                                        addthickness=addthickness)
            if prop is not None:
                props[iwell] = prop
                xcoords[iwell] = xcoord
                ycoords[iwell] = ycoord

    # remove nans
    mask = ~np.isnan(props)
    props = props[mask]
    xcoords = xcoords[mask]
    ycoords = ycoords[mask]

    # define extent of grid where points will be interpolated
    if surface is not None and surface._regxy:
        xin, xend = surface.x[0], surface.x[-1]
        yin, yend = surface.y[0], surface.y[-1]
        nx, ny = surface.nx, surface.ny

    # interpolate
    if intmethod == 'rbf':
        grid_x, grid_y = np.mgrid[xin:xend:nx*1j, yin:yend:ny*1j]
        rbfi = Rbf(xcoords, ycoords, props, **kwargs_interp)
        grid_prop = rbfi(grid_x, grid_y)
    elif intmethod == 'rbf-sklearn':
        grid_x, grid_y = np.mgrid[xin:xend:nx*1j, yin:yend:ny*1j]
        gp = GaussianProcessRegressor(**kwargs_interp)
        gp.fit(np.stack([xcoords, ycoords], axis=1), props)
        grid_prop = gp.predict(
            np.stack([grid_x.ravel(), grid_y.ravel()]).T).reshape(grid_x.shape)
    elif intmethod == 'verde':
        extent = (yin, yend, xin, xend)
        spline = vd.Spline(**kwargs_interp)
        spline.fit((ycoords, xcoords), props)
        grid_prop = spline.grid(region=extent, shape=(nx, ny))
        grid_prop = grid_prop.scalars.data
    else:
        raise NotImplementedError('{} interpolation '
                                  'not implemented'.format(intmethod))

    # save interpolated grid into Surface object
    if surface is not None:
        intsurface = surface.copy(empty=True)
        #intsurface.data = grid_prop
        if isinstance(intsurface.data, np_ma.core.MaskedArray):
            intsurface.data.data[:] = grid_prop
        else:
            intsurface.data = grid_prop

    return intsurface


def create_geomodel(seismic, interpretation, plotflag=False, values=None,
                    colors=None, **kwargs_view):
    """Create simple zone-based geomodel by filling an interpretation
    set of horizons with provided values (or monotonically increasing discrete
    values)

    Parameters
    ----------
    seismic : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Seismic data to identify spatial grid and t/z axis
    interpretation : :obj:`pysubsurface.objects.Interpretation`
        Visualize surface in IL-XL coordinates (``ilxl``) or
        ``yx`` y-x coordinates
    plotflag : :obj:`bool`, optional
         Quickplot
    colors : :obj:`tuple` or :obj:`list`, optional
         Colors to be used to fill between horizons (N+1 elements, where N
         is the number of surfaces in interpretation)
    values : :obj:`tuple` or :obj:`list`, optional
         Numerical values to bu used to fill intervals. If ``None``,
         monotonically increasing discrete values are used
    kwargs_view : :obj:`dict`, optional
        Additional parameters to be passed to
        :func:`pysubsurface.objects.Seismic.view`

    Returns
    -------
    geomodel : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Zone-based geomodel
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
        Axes handles
    cm : :obj:`plt.colormap`
        Colormap

    """
    nsurfaces = len(interpretation.surfaces)
    if values is None:
        background = 0
        values = np.ones(1, nsurfaces)
    else:
        background = values[0]
        values = np.diff(values)

    # create geomodel
    geomodel = seismic.copy(empty=True)
    geomodel.data[:] = background

    # fill gaps in surfaces
    filledsurfaces = []
    for tmpsurface in interpretation.surfaces:
        if isinstance(tmpsurface.data, np_ma.core.MaskedArray):
            filledsurface = _fillgrid(tmpsurface._xs, tmpsurface._ys,
                                      tmpsurface._zs, tmpsurface.x, tmpsurface.y,
                                      fine=2, mindist=-1, intmethod='linear')[2]
        else:
            filledsurface = tmpsurface.data
        filledsurfaces.append(filledsurface)
    nfills = len(filledsurfaces) + 1

    # define common IL-XL grid
    oil = geomodel.ilines[0]
    dil = geomodel.ilines[1] - geomodel.ilines[0]
    nil = geomodel.nil
    oxl = geomodel.xlines[0]
    dxl = geomodel.xlines[1] - geomodel.xlines[0]
    nxl = geomodel.nxl
    otz = geomodel.tz[0]
    dtz = geomodel.dtz
    ntz = geomodel.ntz

    # fill geomodel
    for isurface, (surface, filledsurface) in \
            enumerate(zip(interpretation.surfaces, filledsurfaces)):
        iil = ((surface.il - oil) // dil).astype(np.int16)
        ixl = ((surface.xl - oxl) // dxl).astype(np.int16)
        iil, ixl = np.meshgrid(iil, ixl, indexing='ij')
        iil, ixl = iil.flatten(), ixl.flatten()
        itz = ((filledsurface - otz) // dtz).flatten().astype(np.int16)

        # remove cells where surface is nan
        nanmask = ~np.isnan(itz)
        iil, ixl, itz = iil[nanmask], ixl[nanmask], itz[nanmask]

        # remove cells going out of geomodel dimensions
        dimmask = (iil < 0) | (iil >= nil) | (ixl < 0) | (ixl >= nxl) | (itz >= ntz)
        iil, ixl, itz = iil[~dimmask], ixl[~dimmask], itz[~dimmask]

        # slow, less memory demanding option
        #for iiil, iixl, iitz in zip(iil, ixl, itz):
        #    geomodel.data[iiil, iixl, iitz:] = isurface + 1
        # faster, more memory demanding option
        mask = np.arange(ntz)[:, None]
        mask = values[isurface] * (itz > mask).astype(int)
        geomodel.data[iil, ixl] += mask.T

    if not plotflag:
        fig = ax = None
    else:
        if colors is not None and len(colors) == nfills:
            cm = LinearSegmentedColormap.from_list('name', colors,
                                                   N=nfills)
        else:
            raise ValueError('provide list of colors of '
                             'with element more than lenght of'
                             ' interpretation set')
        fig, ax = geomodel.view(cmap=cm, **kwargs_view)

    return geomodel, fig, ax, cm