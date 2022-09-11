from shutil import copyfile

import os
import segyio

import numpy as np
import matplotlib.pyplot as plt

from segyio.tracefield import TraceField as stf
from pysubsurface.objects.Seismic import Seismic


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
        ``1``:m ost important header words, ``2``: all header words)

    """
    with segyio.open(filename, 'r', iline=iline, xline=xline,
                     ignore_geometry=True) as segyfile:
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
        dt = t[1] - t[0]

        # display info
        print('File: {}\n'.format(filename))
        print('Geometry:')
        print('IL: Irreg')
        print('XL: Irreg')
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


class SeismicIrregular(Seismic):
    """Irregular Seismic object.

    Create object containing a seismic cube with irregular inline
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
        ``local`` when data are stored locally in a folder,
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
        self._taxis = taxis
        self._tzfirst = tzfirst
        self._loadcube= loadcube
        self._scale = scale
        self._kind = kind
        self._verb = verb
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

    def _interpret_seismic(self):
        """Interpret seismic

        Open segy file and interpret its layout. Add useful information to the
        object about dimensions, time/depth-IL-XL axes, and absolute UTM
        coordinates

        """
        if self._verb:
            print('Interpreting {}...'.format(self.filename))

        try:
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                # time axis
                self.tz = f.samples
                self.dtz = f.samples[1]- f.samples[0] # sampling in msec / m
                self.ntz = len(f.samples)

                # extract geometry
                self.XLs = f.attributes(self._xline)[:]
                self.ILs = f.attributes(self._iline)[:]

                self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
                if self.sc < 0:
                    self.sc = 1. / abs(self.sc)
                self.cdpy = self.sc * f.attributes(self._cdpy)[:]
                self.cdpx = self.sc * f.attributes(self._cdpx)[:]

                # define a regular IL and XL axis
                ILunique = np.unique(self.ILs)
                XLunique = np.unique(self.XLs)

                ILmin, ILmax = min(ILunique), max(ILunique)
                XLmin, XLmax = min(XLunique), max(XLunique)

                dIL = min(np.unique(np.diff(ILunique)))
                dXL = min(np.unique(np.diff(XLunique)))

                self.ilines = np.arange(ILmin, ILmax + dIL, dIL)
                self.xlines = np.arange(XLmin, XLmax + dXL, dXL)
                self.dil = self.ilines[1] - self.ilines[0]
                self.dxl = self.xlines[1] - self.xlines[0]

                self.nil, self.nxl = len(self.ilines ), len(self.xlines)
                self.dims = (self.nil, self.nxl, self.ntz)

                self.IIL, self.IXL = np.meshgrid(np.arange(self.nil),
                                                 np.arange(self.nxl),
                                                 indexing='ij')
                if self.ilines[-1] != ILmax:
                    raise ValueError('ILmax is not part of ilines axis...')
                if self.xlines[-1] != XLmax:
                    raise ValueError('XLmax is not part of xlines axis...')

                # indentify look-up table between iline-xline and tracenumber
                self.traceindeces = np.full((self.nil, self.nxl), np.nan)
                iXLs = (self.XLs - XLmin) / dXL
                iILs = (self.ILs - ILmin) / dIL

                if np.abs(iILs - iILs.astype(np.int)).sum() > 0:
                    raise ValueError('At least one IL value does not '
                                     'fit with created IL axis...')
                if np.abs(iXLs - iXLs.astype(np.int)).sum() > 0:
                    raise ValueError('At least one XL value does '
                                     'not fit with created XL axis...')
                iXLs = iXLs.astype(np.int)
                iILs = iILs.astype(np.int)
                self.traceindeces[iILs, iXLs] = np.arange(len(self.XLs))
        except:
            raise ValueError('{} not available...'.format(self.filename))

    def _read_cube(self):
        """Read seismic cube

        Parameters
        ----------
        plotflag : :obj:`bool`, optional
            Quickplot

        Returns
        -------
        data : :obj:`np.ndarray`
            Data

        """
        if self._verb:
            print('Reading %s...' % self.filename)

        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            # create cube
            data = np.zeros(self.dims)

            # select all ilines
            ilinein, ilineend = 0, self.nil

            iil_sub = self.IIL[ilinein:ilineend].flatten()
            ixl_sub = self.IXL[ilinein:ilineend].flatten()
            traceindexes_sub = self.traceindeces[ilinein:ilineend].flatten()
            traces_available = np.logical_not(np.isnan(traceindexes_sub))
            tracesread = np.array([f.trace.raw[int(traceread)]
                                   for traceread in
                                   traceindexes_sub[traces_available]])
            data[iil_sub[traces_available],
                      ixl_sub[traces_available]] = tracesread
            data = data*self._scale

            if self._tzfirst:
                self.data = np.moveaxis(self.data, -1, 0)

            return data

    def info(self, level=2):
        """Print summary of segy file binary headers

        Parameters
        ----------
        level : :obj:`int`, optional
            Level of details (``0``: only geometry,
            ``1``: most important header words, ``2``: all header words)

        """
        _segyinfo(self.filename, iline=self._iline, xline=self._xline,
                  level=level)

    def save(self, outputfilename, verb=False):
        """Save object to segy file.

        Parameters
        ----------
        outputfilename : :obj:`str`
            Name of output file
        verb : :obj:`bool`, optional
            Verbosity

        """
        if verb:
            print('Saving seismic {}'.format(outputfilename))

        # find out indeces for all available traces
        traceindeces = self.traceindeces.ravel()
        traces_available = np.logical_not(np.isnan(traceindeces))
        traceindeces_available = traceindeces[traces_available]
        iil_available = (self.IIL.ravel())[traces_available]
        ixl_available = (self.IXL.ravel())[traces_available]

        copyfile(self.filename, outputfilename)
        with segyio.open(outputfilename, "r+", ignore_geometry=True) as dst:
            for itrace, trace_available in enumerate(traceindeces_available):
                if verb:
                    percdone = 100.*(float(itrace)/len(traceindeces_available))
                    if(np.round(percdone,4) % 10 == 0):
                        print('{} perc done'.format(percdone))
                if self._tzfirst == True:
                    dst.trace[int(trace_available)] = \
                        self.data[:, iil_available[itrace],
                        ixl_available[itrace]].astype('float32')
                else:
                    dst.trace[int(trace_available)] = \
                        self.data[iil_available[itrace],
                                  ixl_available[itrace]].astype('float32')

    def read_subcube(self, ilinein=0, ilineend=-1, verb=False):
        """Read seismic subcube

        Read subset of ilines directly from segy file (without reading the
        entire cube first)

        Parameters
        ----------
        ilinein : :obj:`int`, optional
            Index of first inline (included) to read
        ilineend : :obj:`int`, optional
            Index of last inline (excluded) to read
        verb : :obj:`int`, optional
            Verbosity

        Returns
        -------
        subcube : :obj:`np.ndarray`
            Subcube
        """
        if ilineend is None:
            ilineend = (self.nil-1)

        if ilinein<0 or ilineend>self.nil:
            raise ValueError('Selected inlines exceed available range....')

        if verb:
            print('Reading {} for il={}-{}...'.format(self.filename,
                                                      self.ilines[ilinein],
                                                      self.ilines[ilineend]))

        with segyio.open(self.filename, "r", ignore_geometry=True) as f:

            # create cube/subcube
            subcube = np.zeros((ilineend-ilinein, self.nxl, self.dims[-1]))

            # select subset of ilines
            iil_sub = self.IIL[ilinein:ilineend].flatten()
            ixl_sub  = self.IXL[ilinein:ilineend].flatten()
            traceindexes_sub = self.traceindeces[ilinein:ilineend].flatten()
            traces_available = np.logical_not(np.isnan(traceindexes_sub))

            tracesread = [f.trace.raw[int(traceread)] for traceread in
                          traceindexes_sub[traces_available]]

            subcube[iil_sub[traces_available]- ilinein,
                    ixl_sub[traces_available]] = tracesread
            subcube = subcube*self._scale
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
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            # create section
            section = np.zeros((self.nxl, self.dims[-1]))

            # select subset of ilines
            ixl_sub = self.IXL[iline]
            traceindexes_sub = self.traceindeces[iline].flatten()
            traces_available = np.logical_not(np.isnan(traceindexes_sub))

            tracesread = [f.trace.raw[int(traceread)] for
                          traceread in traceindexes_sub[traces_available]]

            section[ixl_sub[traces_available]] = tracesread
            section = section*self._scale
        return section

    def read_crossline(self, xline=0, verb=False):
        """Read seismic crossline section directly from segy file (without
        reading the entire cube first)

        Parameters
        ----------
        crossline : :obj:`int`, optional
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
            print('Reading {} for il={}...'.format(self.filename,
                                                   self.xlines[xline]))
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
        # create section
            section = np.zeros((self.nil, self.dims[-1]))

            # select subset of ilines
            iil_sub = self.IIL[:, xline]
            traceindexes_sub = self.traceindeces[:, xline].flatten()
            traces_available = np.logical_not(np.isnan(traceindexes_sub))
            print(len(iil_sub))
            print(len(traces_available))
            tracesread = [f.trace.raw[int(traceread)] for
                          traceread in traceindexes_sub[traces_available]]
            section[iil_sub[traces_available]] = tracesread
            section = section * self._scale
        return section

    def read_slice(self, itz=0, verb=False):
        """Read seismic depth/time slice (in this case data is first read and
        slice is extracted afterwards)

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

        slice = self.data[:, :, itz]
        return slice
