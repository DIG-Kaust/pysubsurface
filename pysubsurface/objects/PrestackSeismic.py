import logging

import os
import copy

import numpy as np
import numpy.ma as np_ma
import matplotlib.pyplot as plt
import segyio
import pysubsurface

from azure.datalake.store import multithread
from segyio.tracefield import TraceField as stf
from pysubsurface.utils.utils import findclosest
from pysubsurface.objects.utils import _a2e, _e2a
from pysubsurface.objects.utils import _findclosest_well_seismicsections
from pysubsurface.objects.Cube import Cube
from pysubsurface.objects.Seismic import Seismic

from pysubsurface.proc.seismicmod.waveletest import statistical_wavelet
from pysubsurface.visual.utils import plot_polygon

logging.basicConfig(level=logging.WARNING)


class PrestackSeismic(Seismic):
    """Pre-stack Seismic object.

    Create object containing a pre-stack seismic 4-dimensional array
    (il, xl, offset/angle, t/z) with with regular inline
    and crossline axes.

    Parameters
    ----------
    filename : :obj:`str`
        File name
    iline : :obj:`int`, optional
        Byte location of inline
    xline : :obj:`int`, optional
        Byte location of crossline
    offset : :obj:`int`, optional
        Byte location of offset
    cdpy : :obj:`int`, optional
        Byte location of CDP_Y
    cdpx : :obj:`int`, optional
        Byte location of CDP_X
    taxis : :obj:`int`, optional
        Define either time (``True``) or depth (``False``) axis
    scale : :obj:`int`, optional
        Apply scaling to data when reading it
    loadcube : :obj:`int`, optional
        Load data into ``self.data`` variable during initialization (``True``)
        or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`bool`, optional
        Verbosity

    """
    def __init__(self, filename, iline=189, xline=193, offset=37,
                 cdpy=185, cdpx=181, tzfirst=False, taxis=True, scale=1,
                 loadcube=True, kind='local', verb=False):
        self.filename = filename
        self._iline = iline
        self._xline = xline
        self._offset = offset
        self._cdpy = cdpy
        self._cdpx = cdpx
        self._tzfirst = tzfirst
        self._taxis = taxis
        self._loadcube = loadcube
        self._scale = scale
        self._kind = kind
        self._verb = verb
        self._interpret_seismic()
        self._interpret_extrainfo_seismic()

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
        descr = 'Pre-stack Seismic object:\n' + \
                '-------------------------\n' + \
                'Filename: {}\n'.format(self.filename) + \
                'nil={}, nxl={}, noff={}, ntz={}\n'.format(self.nil, self.nxl,
                                                           self.noff,self.ntz) + \
                'il = {} - {}\n'.format(self.ilines[0], self.ilines[-1]) + \
                'xl = {} - {}\n'.format(self.xlines[0], self.xlines[-1]) + \
                'off/angle = {} - {}\n'.format(self.offsets[0], self.offsets[-1]) + \
                'tz = {} - {}\n'.format(self.tz[0], self.tz[-1])
        if self._loadcube:
            descr = descr + \
                    'min = {0:.3f},\nmax = {1:.3f}\n'.format(self.data.min(),
                                                             self.data.max()) + \
                    'mean = {0:.3f},\nstd = {1:.3f}'.format(np.mean(self.data),
                                                            np.std(self.data))
        return descr

    def _interpret_extrainfo_seismic(self):
        """Interpret extra information for pre-stack seismic. To be used after
        :func:`pysubsurface.objects.Seismic._interpret_seismic`

        Parameters
        ----------
        verb : :obj:`bool`, optional
            Verbosity

        """
        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            self.offsets = f.offsets
            self.doff = self.offsets[1] - self.offsets[0]
            self.noff = len(self.offsets)
            self.dims = (self.nil, self.nxl, self.ntz)

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
        seismiccopy = copy.deepcopy(self)

        if empty:
            seismiccopy._data = np.zeros_like(self.data)
        else:
            seismiccopy._data = np.copy(self.data)

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
        raise NotImplementedError('not yet implemented for prestack')

    def read_subcube(self, ilinein=0, ilineend=None, verb=False):
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
        raise NotImplementedError('not implemented yet!')

        #with segyio.open(self.filename, "r",
        #                 iline=self._iline, xline=self._xline) as f:
        #    subcube = segyio.collect(f.gather[self.ilines[ilinein]:
        #                                     self.ilines[ilineend], :, :]) * \
        #              self._scale
        #return subcube

    def read_inline(self, iline=0, verb=False):
        r"""Read seismic inline section directly from segy file (without
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
            Inline section of size :math:`\lbrack n_{off} \times n_{xl} \times n_t \rbrack`

        """
        if iline < 0 or iline > self.nil:
            raise ValueError('Selected inline exceeds available range....')

        if verb:
            print('Reading {} for il={}...'.format(self.filename,
                                                   self.ilines[iline]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            section = segyio.collect(f.iline[self.ilines[iline], :]) * self._scale
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
        if xline < 0 or xline > self.nil:
            raise ValueError('Selected crossline exceeds available range....')

        if verb:
            print('Reading {} for xl={}...'.format(self.filename,
                                                   self.xlines[xline]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            section = segyio.collect(f.xline[self.xlines[xline], :]) * self._scale
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

        raise NotImplementedError('not implemented yet!')
        #with segyio.open(self.filename, "r",
        #                 iline=self._iline, xline=self._xline) as f:
        #
        #    slice = segyio.collect(f.depth_slice[itz, :, :]) * self._scale
        #return slice

    def read_gather(self, iline=0, xline=0, verb=False):
        """Read seismic depth/time gather directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        iline : :obj:`int`, optional
            Index of inline to read
        xline : :obj:`int`, optional
            Index of crossline to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        gather : :obj:`np.ndarray`
            Gather

        """
        if iline < 0 or iline > self.nil:
            raise ValueError('Selected inline exceeds available range....')

        if xline < 0 or xline > self.nil:
            raise ValueError('Selected crossline exceeds available range....')

        if verb:
            print('Reading {} for il={}, xl={}...'.format(self.filename,
                                                          self.ilines[iline],
                                                          self.xlines[xline]))

        with segyio.open(self.filename, "r",
                         iline=self._iline, xline=self._xline) as f:
            slice = f.gather[f.ilines[iline], f.xlines[xline]] * self._scale
        return slice


    def extract_gather_verticalwell(self, well, verb=False):
        """Extract gather at inline and crossline closest to a
        vertical well

        Parameters
        ----------
        well : :obj:`Well`
            Well object
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        gather : :obj:`np.ndarray`
            Seismic gather at well location

        """
        if verb:
            print('Reading trace from {} closest to '
                  'well {}...'.format(self.filename, well.wellname))

        # find IL and XL the well is passing through
        ilwell, xlwell = _findclosest_well_seismicsections(well, self,
                                                           verb=verb)
        gather = self.read_gather(findclosest(self.ilines, ilwell),
                                  findclosest(self.xlines, xlwell))

        return gather


    #########
    # Viewers
    #########
    def view(self, ax=None, ilplot=None, xlplot=None,
             tzoom=None, tzoom_index=True, scale=1.,
             horizons=None, horcolors=[], scalehors=1., hornames=False,
             clip=1., clim=[], cmap='seismic', cbar=False, interp=None,
             figsize=(3, 7),  title='', savefig=None):
        """Quick visualization of single gather in PrestackSeismic object.

        Parameters
        ----------
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` create new figure)
        ilplot : :obj:`int`, optional
            Index of inline to plot (if ``None`` show inline in the middle)
        xlplot : :obj:`int`, optional
            Index of crossline to plot
            (if ``None`` show crossline in the middle)
        tzoom : :obj:`tuple`, optional
            Time/depth start and end values (or indeces) for visualization
            of time/depth axis
        tzoom_index : :obj:`bool`, optional
            Consider values in ``tzoom`` as indeces (``True``) or
            actual values (``False``)
        scale : :obj:`float`, optional
             Apply scaling to data when showing it
        horizons : :obj:`pysubsurface.objects.Interpretation` or :obj:`pysubsurface.objects.Surface`, optional
             set of horizons to plot
        scalehors : :obj:`float`, optional
             Apply scaling to horizons time/depth values when showing them
        hornames : :obj:`bool`, optional
             Add names of horizons (``True``) or not (``False``)
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
        # find out indices for data extraction
        if ilplot is None:
            ilplot = int(len(self.ilines) / 2)
        if xlplot is None:
            xlplot = int(len(self.xlines) / 2)

        if tzoom is not None:
            tzoom = sorted(tzoom, reverse=True)
            if tzoom_index:
                tzoom = self.tz[tzoom]

        # get horizons to plot
        if horizons is None:
            plothorizons = False
        elif isinstance(horizons, pysubsurface.objects.Interpretation):
            surfaces = horizons.surfaces
            plothorizons = True
        elif isinstance(horizons, pysubsurface.objects.Surface):
            surfaces = [horizons]
            plothorizons = True
        else:
            raise TypeError('horizons must be of type Surface or '
                            'Interpretation...')

        if isinstance(horcolors, str):
            horcolors = [horcolors]
        if len(horcolors) == 0:
            horcolors = ['k'] * len(surfaces)

        # make figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        # extract gather
        gath = self.read_gather(iline=ilplot, xline=xlplot)

        if len(clim) == 0:
            clim = [-clip * np.abs(scale) * np.nanmax(gath),
                    clip * np.abs(scale) * np.nanmax(gath)]

        # display gather
        im = ax.imshow(gath.T, vmax=clim[1], vmin=clim[0], cmap=cmap,
                       extent=(self.offsets[0], self.offsets[-1],
                               self.tz[-1], self.tz[0]),
                       interpolation=interp)
        ax.set_title('Gather slice at '
                     'IL={}, XL={}'.format(self.ilines[ilplot],
                                           self.xlines[xlplot]))
        ax.set_xlabel('Off/Angle')
        ax.set_ylabel('t/z')
        ax.axis('tight')
        if tzoom is not None:
            ax.set_ylim(tzoom)
        if cbar:
            plt.colorbar(im, ax=ax, orientation='horizontal',
                         pad=0.07, shrink=0.7)

        # display horizons
        if plothorizons:
            for tmpsurface, color in zip(surfaces, horcolors):
                print(tmpsurface.filename)
                surface = tmpsurface.copy()
                if not isinstance(surface.data, np_ma.core.MaskedArray):
                    surface.data = np_ma.masked_array(surface.data,
                                                      mask=np.zeros_like(surface))
                ilgath = findclosest(surface.il, self.ilines[ilplot]),
                xlgath = findclosest(surface.xl, self.xlines[xlplot])
                surface_mask = surface.data.mask[ilgath, xlgath] * scalehors

                if not surface_mask:
                    surface_gath = surface.data.data[ilgath, xlgath] * scalehors
                    ax.axhline(surface_gath, color=color, lw=4)

                    if hornames:
                        ax.text(1.05*self.offsets[-1],
                                surface_gath,
                                os.path.basename(surface.filename),
                                va="center", color=color)
        # savefig
        if fig is not None and savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')

        return fig, ax

