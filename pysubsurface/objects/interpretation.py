import os
import glob
import copy
import numpy as np
import numpy.ma as np_ma
import matplotlib.pyplot as plt

from pysubsurface.utils.utils import findclosest
from pysubsurface.objects.Surface import Surface, _FORMATS


plt.rc('font', family='serif')


class Interpretation:
    """Interpretation object.

    This object contains a set of surfaces representing an interpretation of
    a seismic dataset

    Parameters
    ----------
    filenames : :obj:`tuple` or :obj:`list`, optional
        Name of files containing surfaces
    directory : :obj:`str`, optional
        Name of directory containing surfaces
        (use when ``filenames=None``). If both are set to ``None`` the
        Interpretation object is simply initialized
    extension : :obj:`str`, optional
        Files extensions (use in combination with ``directory``)
    format : :obj:`str`, optional
        Format of file containing surface (available options: ``dsg5_long``)
    decimals : :obj:`int`, optional
        Number of decimals in x-y (or il-xl) coordinates
        (if ``None`` keep all)
    nanvals : :obj:`float`, optional
         Value to use to indicate non-valid numbers
         (if not provided default one from FORMATS dict in
         :class:`pysubsurface.objects.Surface` class is used)
    finegrid : :obj:`int`, optional
         Fill grid at (0) available points using nearest neighour
         interpolation, (1) at available points but use only integer
         locations to avoid multiple locations to fit at same location,
         (2) at available points and interpolate/extrapolate to finer
         grid
    gridints : :obj:`tuple` or :obj:`list`, optional
         Sampling step in IL/y and XL/x direction
         (relevant only if ``finegrid=2``, if not provided use min value
         found from interpreting surface)
    mindist : :obj:`bool`, optional
         Minimum distance allowed for extrapolation in case ``finegrid=2``
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    plotflag : :obj:`bool`, optional
         Quickplot
    verb : :obj:`bool`, optional
         Verbosity

    .. note:: This class and the :class:`pysubsurface.objects.Surface` have
        historically supported several formats. At the current stage of
        development we decided to support a single format which is likely
        to change to match the format used/introduced to store surfaces
        within Omnia data lake.

    """
    def __init__(self, filenames=None, directory='./', extension='txt',
                 format='dsg5_long', decimals=None, nanvals=False, finegrid=0,
                 gridints=[-1, -1, -1, -1], mindist=-1,
                 kind='local', plotflag=False, verb=False):
        self._decimals = decimals
        self._gridints = gridints.copy()
        self._mindist = mindist
        self._finegrid = finegrid
        self._plotflag = plotflag
        self._kind = kind
        self._verb = verb
        self._loadsurfaces = False

        if format in _FORMATS.keys():
            self._format = format
            self._format_dict = _FORMATS[self._format]
            if nanvals is None:
                self._format_dict['nanvals'] = nanvals
        else:
            raise ValueError('{} not contained in list of available'
                             ' formats=[{}]'.format(format,
                                                    ' '.join(_FORMATS)))
        # identify list of filenames
        if filenames is not None:
            self._filenames = filenames.copy()
        else:
            self._filenames = glob.glob(os.path.join(directory,
                                                     '*.' + extension))
        # load surfaces
        if len(self._filenames) == 0:
            self._filenames = []
            self._surfaces = []
            self._loadsurfaces = True
        elif self._loadsurfaces and len(self._filenames) > 0:
            self._surfaces = self._load_surfaces()
            self._loadsurfaces = True

    @property
    def surfaces(self):
        if not self._loadsurfaces:
            self._loadsurfaces = True
            self._surfaces = self._load_surfaces()
        return self._surfaces

    def __str__(self):
        descr = 'Interpretation object:\n' + \
                ''.join(['{}\n'.format(filename)
                         for filename in self._filenames])
        return descr

    def _load_surfaces(self):
        """Load surfaces
        """
        def _load_surface(surfacefile):
            if self._verb:
                print('Loading surface {}...'.format(surfacefile))
            return Surface(filename=surfacefile, format=self._format,
                           finegrid=self._finegrid, gridints=self._gridints,
                           loadsurface=True, plotflag=self._plotflag,
                           kind= self._kind, verb=False)
        print('HERE')
        return [_load_surface(surfacefile) for
                surfacefile in self._filenames]

    def add_surface(self, newsurface, name=None, index=-1):
        """Add surface.

        Parameters
        ----------
        newsurface : :obj:`float`
            New surface
        name : :obj:`str`, optional
            Name to overwrite to filename of copied surface
        index : :obj:`int`, optional
            Index of position in list where horizon will be added

        """
        # ensure surfaces object has been initialized
        if not self._loadsurfaces:
            self._surfaces = self._load_surfaces()
            self._loadsurfaces = True

        if index < 0:
            index = len(self._filenames) - index

        self._filenames.insert(index, name)
        self.surfaces.insert(index, newsurface)

    def add_constant_surface(self, constant, copyfrom=0, name=None, index=-1):
        """Add surface with constant value.

        Parameters
        ----------
        constant : :obj:`float`
            Constant value for surface
        copyfrom : :obj:`int`, optional
            Index of surface to be used as template for new one
        name : :obj:`str`, optional
            Name to overwrite to filename of copied surface
        index : :obj:`int`, optional
            Index of position in list where horizon will be added

        """
        if index < 0:
            index = len(self._filenames) - index

        newsurface = self.surfaces[copyfrom].copy(empty=True)
        newsurface.data[:] = constant
        newsurface.filename = name
        self._filenames.insert(index, name)
        self.surfaces.insert(index, newsurface)

    def add_fluidcontact(self, constant, top, name=None, index=-1):
        """Add fluid contact

        Parameters
        ----------
        constant : :obj:`float`
            Constant value for surface
        top : :obj:`int`, optional
            Index of surface to be interpreted as top horizon
        name : :obj:`str`, optional
            Name to overwrite to filename of copied surface
        index : :obj:`int`, optional
            Index of position in list where horizon will be added

        """
        if index < 0:
            index = len(self._filenames) - index

        newsurface = self.surfaces[top].copy()
        if isinstance(self.surfaces[top].data, np_ma.core.MaskedArray):
            newsurface.data.data[:] = constant
            newsurface.data.mask[newsurface.data < self.surfaces[top].data] = True
        else:
            newsurface.data[:] = constant
            newsurface.data[np.isnan(self.surfaces[top].data)] = np.nan
            newsurface.data[newsurface.data < self.surfaces[top].data] = np.nan
        newsurface.filename = name
        self._filenames.insert(index, name)
        self.surfaces.insert(index, newsurface)

    def copy(self, empty=False):
        """Return a copy of the object.

        Parameters
        ----------
        empty : :obj:`bool`
            Copy input data (``True``) or just create an empty data (``False``)

        Returns
        -------
        interpretationcopy : :obj:`pysubsurface.objects.Interpretation`
            Copy of Interpretation object

        """
        interpretationcopy = copy.deepcopy(self)
        if empty:
            for isurface,surface in enumerate(self.surfaces):
                if isinstance(surface.data, np.ndarray):
                    interpretationcopy.surfaces[isurface].data = \
                        np.zeros((surface.ny, surface.nx))
                else:
                    interpretationcopy.surfaces[isurface].data.data[:] = 0.
        return interpretationcopy

    def view_surface(self, isurface, title=None,
        polygonset=None, polygons=None, wells=None,
        lims=None, **kwargs_surface):
        """Display surface (horizon or map) with wells and polygons

        Parameters
        ----------
        isurface : :obj:`str`
            Index of of surface
        title : :obj:`str`, optional
            Figure title (if ``None`` use ``surfacename``)
        polygonset : :obj:`pysubsurface.objects.PolygonSet.PolygonSet`, optional
            Polygonset to display
        polygons : :obj:`list`, optional
            Polygons to display
        wells : :obj:`dict`, optional
            Wells to display as dict (key: wellname,
            value::obj:`pysubsurface.objects.Well.Well`
        lims : :obj:`tuple`, optional
            y-x (or il-xl) limits
        kwargs_surface : :obj:`dict`, optional
            Additional input parameters to be provided to
            :func:`pysubsurface.objects.Surface.view`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        surfaceplot = self.surfaces[isurface]

        fig, ax = surfaceplot.view(**kwargs_surface)
        ax.set_title('' if title is None else title)

        axiskm = False if 'axiskm' not in kwargs_surface \
            else kwargs_surface['axiskm']

        # add polygon set
        if polygonset is not None:
            ax = polygonset.view(ax, axiskm=axiskm, color='w')

        # add polygons
        if polygons is not None:
            for polygon in polygons:
                polygon.view(ax, color='w', flipaxis=False, bbox=False,
                             axiskm=axiskm)
        # limit extent
        if lims is not None:
            ax.set_ylim([lims[0], lims[1]])
            ax.set_xlim([lims[2], lims[3]])

        # add wells
        if wells is not None:
            for wellname in wells.keys():
                wells[wellname].trajectory.view_traj(ax=ax, color='k',
                                                     axiskm=axiskm,
                                                     labels=False,
                                                     checkwell=True)
        return fig, ax

    def view(self, which='all', ilplot=None, xlplot=None, tzshift=0.0,
             horcolors=[], scalehors=1., hornames=False, horlw=2,
             reservoir=None, axs=None, figsize=(15, 4),
             title=False, verb=False):
        """Visualize horizons on inline and crossline section

        Parameters
        ----------
        which : :obj:`str`, optional
            Slices to visualize. ``all``: il and xl slices,
            slices, ``il``: iline slice, ``xline``: x slice
        ilplot : :obj:`int`, optional
            Inline to plot (if ``None`` show inline in the middle)
        xlplot : :obj:`int`, optional
            Crossline to plot (if ``None`` show crossline in the middle)
        tzshift : :obj:`float`, optional
            Shift to apply to surfaces in tz axis in seismic
        horcolors : :obj:`list`, optional
            Horizon colors
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
        axs : :obj:`plt.axes`
           Axes handles (if ``None`` create a new figure)
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`bool`, optional
            Add title
        verb : :obj:`bool`, optional
            Verbosity

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
           Axes handles

        """
        xl = self.surfaces[0].xl
        il = self.surfaces[0].il

        if xlplot is None:
            xlplot = xl[len(xl)//2]
        if ilplot is None:
            ilplot = il[len(il)//2]

        if which == 'all':
            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
            else:
                fig = None
            if title:
                axs[0].set_title(
                    'Intepretation (IL={})'.format(int(il[findclosest(il, ilplot)])))
                axs[1].set_title(
                    'Intepretation (XL={})'.format(int(xl[findclosest(xl, xlplot)])))
        else:
            if axs is None:
                fig, axs = plt.subplots(1, 1, figsize=figsize)
            else:
                fig = None
            if title:
                if which == 'il':
                    axs.set_title('Intepretation (IL={})'.format(il[findclosest(il, ilplot)]))
                elif which == 'xl':
                    axs.set_title('Intepretation (XL={})'.format(xl[findclosest(xl, xlplot)]))

        # choose colors
        if len(horcolors) == 0:
            horcolors = ['k'] * len(self.surfaces)

        for isurface, (surface, color) in enumerate(zip(self.surfaces, horcolors)):
            if verb:
                print('Plotting {}...'.format(surface.filename))

            # extract axes
            xl = self.surfaces[isurface].xl
            il = self.surfaces[isurface].il

            # extract surface
            surface_x = surface.data[findclosest(il, ilplot)] * scalehors + tzshift
            surface_y = surface.data[:, findclosest(xl, xlplot)] * scalehors + tzshift

            # display surface
            if which == 'all':
                axs[0].plot(xl, surface_x, color, lw=horlw)
                axs[1].plot(il, surface_y, color, lw=horlw)
            elif which == 'il':
                axs.plot(xl, surface_x, color, lw=horlw)
            else:
                axs.plot(il, surface_y, color, lw=horlw)

            # add text label
            if hornames:
                # find where to add text label
                if hornames:
                    if len(np.where(surface_x.mask == 0)[0]) > 0 and \
                        len(np.where(surface_y.mask == 0)[0]) > 0:
                        surface_x_text = surface_x[0] if type(surface) == np.ndarray \
                            else surface_x.data[np.where(surface_x.mask == 0)[0][0]]
                        surface_y_text = surface_y[0] if type(surface) == np.ndarray \
                            else surface_y.data[np.where(surface_y.mask == 0)[0][0]]
                        surface_name = None if surface.filename is None else os.path.basename(surface.filename)

                        if which == 'all':
                            axs[0].text(xl[np.where(surface_x.mask==0)[0][0]],
                                        surface_x_text, surface_name,
                                        ha="center", va="center", color=color,
                                        bbox=dict(boxstyle="round", fc=(1., 1., 1.),
                                                  ec=color))
                            axs[1].text(il[np.where(surface_y.mask==0)[0][0]],
                                        surface_y_text, surface_name,
                                        ha="center", va="center", color=color,
                                        bbox=dict(boxstyle="round", fc=(1., 1., 1.),
                                                  ec=color))
                        elif which == 'il':
                            axs.text(il[np.where(surface_x.mask == 0)[0][0]],
                                     surface_x_text, surface_name,
                                     ha="center", va="center", color=color,
                                     bbox=dict(boxstyle="round", fc=(1., 1., 1.),
                                               ec=color))
                        else:
                            axs.text(il[np.where(surface_y.mask == 0)[0][0]],
                                     surface_y_text, surface_name,
                                     ha="center", va="center", color=color,
                                     bbox=dict(boxstyle="round", fc=(1., 1., 1.),
                                               ec=color))

        # fill fluids between horizons
        if reservoir is not None:
            colorfill = []
            reservoir_surfaces = [self.surfaces[reservoir['top']],
                                  self.surfaces[reservoir['base']]]
            if 'GOC' in reservoir.keys():
                reservoir_surfaces.insert(1, self.surfaces[reservoir['GOC']])
                colorfill.append('r')
            if 'OWC' in reservoir.keys():
                reservoir_surfaces.insert(len(colorfill) + 1,
                                          self.surfaces[reservoir['OWC']])
                colorfill.append('g')
            colorfill.append('b')

            for isurface in range(len(reservoir_surfaces) - 1):
                surface_base = reservoir_surfaces[isurface + 1].copy()
                if isurface == 0:
                    surface_top = reservoir_surfaces[isurface].copy()
                if not isinstance(surface.data, np_ma.core.MaskedArray):
                    if isurface == 0:
                        surface_top.data = np_ma.masked_array(
                            surface_top.data,
                            mask=np.zeros_like(surface_top)) + tzshift
                    surface_base.data = np_ma.masked_array(
                        surface_base.data,
                        mask=np.zeros_like(
                            surface_base)) + tzshift

                if isurface == 0:
                    surface_top_xl = surface_top.data[:, findclosest(surface_top.xl, xlplot)] \
                                     * scalehors
                    surface_top_il = surface_top.data[findclosest(surface_top.il, ilplot)] \
                                     * scalehors
                else:
                    surface_top_xl = np.maximum(surface_top_xl,
                                                surface_base_xl)
                    surface_top_il = np.maximum(surface_top_il,
                                                surface_base_il)

                surface_base_xl = surface_base.data[:,
                                  findclosest(surface_base.xl, xlplot)] \
                                  * scalehors
                surface_base_il = surface_base.data[
                                      findclosest(surface_base.il, ilplot)] \
                                  * scalehors
                surface_base_il = np.interp(surface_top.xl,
                                            surface_base.xl,
                                            surface_base_il)
                surface_base_xl = np.interp(surface_top.il,
                                            surface_base.il,
                                            surface_base_xl)

                if which == 'all':
                    axs[0][0].fill_between(surface_top.xl,
                                           surface_top_il,
                                           surface_base_il,
                                           where=surface_top_il < surface_base_il,
                                           color=colorfill[isurface], alpha=0.3)
                    axs[0][1].fill_between(surface_top.il,
                                           surface_top_xl,
                                           surface_base_xl,
                                           where=surface_top_xl < surface_base_xl,
                                           color=colorfill[isurface], alpha=0.3)
                elif which == 'il':
                    axs.fill_between(surface_top.xl,
                                     surface_base_il,
                                     surface_top_il,
                                     where=surface_top_il < surface_base_il,
                                     color=colorfill[isurface],
                                     alpha=0.3)
                else:
                    axs.fill_between(surface_top.il,
                                     surface_base_xl,
                                     surface_top_xl,
                                     where=surface_top_xl < surface_base_xl,
                                     color=colorfill[isurface],
                                     alpha=0.3)
        # add labels
        if which == 'all':
            axs[0].set_xlabel('XL')
            axs[1].set_xlabel('IL')
        elif which == 'il':
            axs.set_xlabel('XL')
        else:
            axs.set_xlabel('IL')

        # invert z axis
        if fig is not None:
            if which == 'all':
                axs[0].invert_yaxis()
                axs[1].invert_yaxis()
            else:
                axs.invert_yaxis()

        return fig, axs

    def view_arbitratyline(self, ils, xls, tzshift=0.0,
                           horcolors=[], scalehors=1., hornames=False, horlw=2,
                           reservoir=None, ax=None, figsize=(15, 4)):
        """Visualize horizons through arbitrary lines

        Parameters
        ----------
        ils : :obj:`list`
            List of inlines to display
        xls : :obj:`int`, optional
            List of crosslines to display
        tzshift : :obj:`float`, optional
            Shift to apply to surfaces in tz axis in seismic
        horcolors : :obj:`list`, optional
            Horizon colors
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
        ax : :obj:`plt.axes`
           Axes handle (if ``None`` create a new figure)
        figsize : :obj:`tuple`, optional
            Size of figure

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        for tmpsurface, color in zip(self.surfaces, horcolors):
            surface = tmpsurface.copy()
            if not isinstance(surface.data, np_ma.core.MaskedArray):
                surface.data = np_ma.masked_array(surface.data,
                                                  mask=np.zeros_like(
                                                      surface)) + tzshift

            iils = np.array((ils - surface.il[0]) / surface.dil).astype(int)
            ixls = np.array((xls - surface.xl[0]) / surface.dxl).astype(int)
            masksurface = np.array([True] * len(iils))
            masksurface[((iils >= surface.nil) | (iils < 0))] = False
            masksurface[((ixls >= surface.nxl) | (ixls < 0))] = False
            surfaceline = surface.data[iils[masksurface],
                                       ixls[masksurface]] * scalehors
            ax.plot(np.arange(len(ils))[masksurface],
                    surfaceline,
                    color, lw=horlw)

        # fill fluids between horizons
        if reservoir is not None:
            colorfill = []
            reservoir_surfaces = [self.surfaces[reservoir['top']],
                                  self.surfaces[reservoir['base']]]
            if 'GOC' in reservoir.keys():
                reservoir_surfaces.insert(1, self.surfaces[reservoir['GOC']])
                colorfill.append('r')
            if 'OWC' in reservoir.keys():
                reservoir_surfaces.insert(len(colorfill) + 1,
                                          self.surfaces[reservoir['OWC']])
                colorfill.append('g')
            colorfill.append('b')

            for isurface in range(len(reservoir_surfaces) - 1):
                surface_base = reservoir_surfaces[isurface + 1].copy()
                if isurface == 0:
                    surface_top = reservoir_surfaces[isurface].copy()
                if not isinstance(surface.data, np_ma.core.MaskedArray):
                    if isurface == 0:
                        surface_top.data = np_ma.masked_array(
                            surface_top.data,
                            mask=np.zeros_like(surface_top)) + tzshift
                    surface_base.data = np_ma.masked_array(
                        surface_base.data,
                        mask=np.zeros_like(
                            surface_base)) + tzshift

                if isurface == 0:
                    iils_top = np.array((ils - surface_top.il[0]) / surface_top.dil).astype(int)
                    ixls_top = np.array((xls - surface_top.xl[0]) / surface_top.dxl).astype(int)
                    masksurface_top = np.array([True] * len(iils_top))
                    masksurface_top[((iils_top >= surface_top.nil) | (iils_top < 0))] = False
                    masksurface_top[((ixls_top >= surface_top.nxl) | (ixls_top < 0))] = False
                    surface_top_line = surface_top.data[iils_top[masksurface_top],
                                                        ixls_top[masksurface_top]] \
                                       * scalehors
                else:
                    surface_top_line = np.maximum(surface_top_line,
                                                  surface_base_line)

                iils_base = np.array((ils - surface_base.il[0]) / surface_base.dil).astype(int)
                ixls_base = np.array((xls - surface_base.xl[0]) / surface_base.dxl).astype(int)
                masksurface_base = np.array([True] * len(iils_base))
                masksurface_base[((iils_base >= surface_base.nil) | (iils_base < 0))] = False
                masksurface_base[((ixls_base >= surface_base.nxl) | (ixls_base < 0))] = False
                surface_base_line = surface_base.data[iils_base[masksurface_base],
                                                      ixls_base[masksurface_base]] \
                                    * scalehors

                surface_base_line = np.interp(np.arange(len(ils))[masksurface_top],
                                              np.arange(len(ils))[masksurface_base],
                                              surface_base_line)

                ax.fill_between(np.arange(len(ils))[masksurface_top],
                                surface_base_line,
                                surface_top_line,
                                where=surface_top_line < surface_base_line,
                                color=colorfill[isurface],
                                alpha=0.3)

        if fig is not None:
            ax.invert_yaxis()
        return fig, ax


class Ensemble:
    """Ensemble object.

    This object contains an ensemble of interpretations and can be used to
    perform basic statistics on them

    Parameters
    ----------
    interpretations : :obj:`dict`
        Dictionary containing a set of interpretations

    """
    def __init__(self, interpretations):
        self.interpretations = interpretations
        self.nints = len(interpretations)
        self.intnames = list(interpretations.keys())
        self.firstintname = self.intnames[0]
        self.nhors = len(interpretations[self.firstintname].surfaces)

    def mean_std(self):
        """Mean and standard deviation

        Compute point-wise mean and standard deviation between different
        interpreations of each horizon

        Returns
        -------
        intmean : :obj:`pysubsurface.objects.Surface`
            Computed mean
        intstd : :obj:`pysubsurface.objects.Surface`
            Computed standard deviation

        """
        intmean = copy.deepcopy(self.interpretations[self.firstintname])
        intstd = copy.deepcopy(self.interpretations[self.firstintname])
        for ireal, real in enumerate(self.intnames):
            for ihor in range(self.nhors):
                if ireal == 0:
                    intmean.surfaces[ihor].data[:] = \
                        self.interpretations[real].surfaces[ihor].data[:] / self.nints
                    intstd.surfaces[ihor].data[:] = \
                        self.interpretations[real].surfaces[ihor].data[:] ** 2 / self.nints
                else:
                    intmean.surfaces[ihor].data[:] += \
                        self.interpretations[real].surfaces[ihor].data[:] / self.nints
                    intstd.surfaces[ihor].data[:] += \
                        self.interpretations[real].surfaces[ihor].data[:] ** 2 / self.nints
        for ihor in range(self.nhors):
            intstd.surfaces[ihor].data[:] -= intmean.surfaces[ihor].data[:] ** 2
            intstd.surfaces[ihor].data[:] = np.sqrt(intstd.surfaces[ihor].data[:])
        return intmean, intstd

    def copy(self, empty=False):
        """Return a copy of the object.

        Parameters
        ----------
        empty : :obj:`bool`
            Copy input data (``True``) or just create an empty data (``False``)

        Returns
        -------
        ensemblecopy : :obj:`pysubsurface.objects.Ensemble`
            Copy of Ensemble object

        """
        ensemblecopy = copy.deepcopy(self)
        if empty:
            for interpr in self.interpretations:
                for isurface,surface in enumerate(self.interpretations[interpr].surfaces):
                    if isinstance(surface.data, np.ndarray):
                        ensemblecopy.interpretations[interpr].surfaces[isurface].data = \
                            np.zeros((surface.ny, surface.nx))
                    else:
                        ensemblecopy.interpretations[interpr].surfaces[isurface].data.data[:] = 0.
        return ensemblecopy

    def view(self, which='all', ilplot=None, xlplot=None, tzshift=0.0,
             horcolors=[], scalehors=1., hornames=False, horlw=2, reservoir=None,
             axs=None, figsize=(15, 4), title=None, verb=False):
        """Visualize ensemble on inline and crossline section

        See :func:`pysubsurface.objects.Interpretation.view for more details
        """
        if axs is None:
            fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)

        else:
            fig = None

        for interpr in self.interpretations:
            self.interpretations[interpr].view(which=which, ilplot=ilplot,
                                               xlplot=xlplot, tzshift=tzshift,
                                               horcolors=horcolors,
                                               scalehors=scalehors,
                                               hornames=hornames, horlw=horlw,
                                               reservoir=None,
                                               axs=axs, verb=verb)
        if fig is not None:
            axs[0].invert_yaxis()
        return fig, axs

    def view_arbitratyline(self, ils, xls, tzshift=0.0,
                           horcolors=[], scalehors=1., hornames=False, horlw=2,
                           reservoir=None, ax=None, figsize=(15, 4), title=None):
        """Visualize horizons through arbitrary lines

        See :func:`pysubsurface.objects.Interpretation.view_arbitratyline for
        more details
        """
        if ax is None:
            fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
        else:
            fig = None

        for interpr in self.interpretations:
            self.interpretations[interpr].view_arbitratyline(ils=ils,
                                                             xls=xls,
                                                             tzshift=tzshift,
                                                             horcolors=horcolors,
                                                             scalehors=scalehors,
                                                             hornames=hornames,
                                                             horlw=horlw,
                                                             reservoir=None,
                                                             ax=ax)
        if fig is not None:
            ax.invert_yaxis()
        return fig, ax