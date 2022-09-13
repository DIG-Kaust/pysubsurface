import logging
import os
import copy

import numpy as np
import numpy.ma as np_ma
import scipy.interpolate as spint
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree as KDTree
from pysubsurface.utils.utils import findclosest

logging.basicConfig(level=logging.WARNING)

plt.rc('font', family='serif')


_FORMATS = {'dsg': {'skiprows_in': 123, 'skiprows_end': 0,
                    'xpos': 0, 'ypos': 1, 'zpos': 2}}


class Fault:
    r"""Fault object.

    This object contains a fault. The fault can be imported from dsg formatted
    file with 4 columns (x, y, z, id). Note that internally the fault will be
    kept as scatter points and returned interpolated within a grid or a line
    only upon request.

    Parameters
    ----------
    filename : :obj:`str`
        Name of files containing surfaces (use ``None`` to create an
        empty surface)
    loadpoints : :obj:`bool`, optional
         Read points during initialization (``True``) or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`bool`, optional
         Verbosity

    """
    def __init__(self, filename, loadpoints=True,
                 kind='local', verb=False):
        self.filename = filename
        self.format = 'dsg'
        self.format = 'dsg'
        self.format_dict = _FORMATS[self.format].copy()
        self._kind = kind
        self._verb = verb

        # load points
        if loadpoints:
            self._read_data()

    def __str__(self):
        descr = 'Fault object:\n' + \
                '---------------\n' + \
                'Filename: {}\n'.format(self.filename)
        return descr

    def _read_data(self):
        """Read data
        """
        if self._verb:
            print('Loading fault {}'.format(self.filename))

        data = np.loadtxt(self.filename,
                          skiprows=self.format_dict['skiprows_in'])

        self.x = data[:, self.format_dict['xpos']]
        self.y = data[:, self.format_dict['ypos']]
        self.z = data[:, self.format_dict['zpos']]

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
        faultcopy = copy.deepcopy(self)
        if empty:
            faultcopy.z[:] = 0.
        return faultcopy

    def grid(self, xgrid, ygrid, mindist=None, quickplot=False, clim=None):
        """Grid fault

        Return fault in a regular grid with axis ``xgrid`` and ``ygrid``. Note
        that if ``xgrid`` and ``ygrid`` are single numbers, gridding will be
        performed along the closest 1d line in the fault cloud points
        (instead for a 2d grid).

        Parameters
        ----------
        xgrid : :obj:`np.ndarray` or :obj:`float`
            X axis for grid
        ygrid : :obj:`np.ndarray` or :obj:`float`, optional
            Y axis for grid
        mindist : :obj:`float`, optional
            Minimum distance from fault point for interpolated grid (points
            whose distance is bigger than mindist will be masked out)
        quickplot : :obj:`bool`, optional
            Quickplot
        clim : :obj:`float`, optional
             Colorbar limits (if ``None`` infer from data and
             apply ``clip`` to those)

        Returns
        -------
        gridded_fault : :obj:`np.ndarray`
            Gridded fault

        """
        if isinstance(xgrid, float):
            yline, xline = True, False
        elif isinstance(ygrid, float):
            yline, xline = False, True
        else:
            yline, xline = False, False

        if not yline and not xline:
            y, x, z = self.y, self.x, self.z
            ny, nx = len(ygrid), len(xgrid)
            Ygrid, Xgrid = np.meshgrid(ygrid, xgrid, indexing='ij')
            Ygrid, Xgrid = Ygrid.flatten(), Xgrid.flatten()
            gridded_fault = spint.griddata((y, x), z, (Ygrid, Xgrid),
                                           fill_value=np.nan, method='linear')
        elif yline:
            ny, nx = len(ygrid), 1
            xclosest = np.unique(self.x)[findclosest(np.unique(self.x), xgrid)]
            if sum(self.x == xclosest) >= 2:
                if np.abs(xclosest - xgrid) > mindist:
                    return np.full(ny, np.nan)
                y = self.y[self.x == xclosest]
                z = self.z[self.x == xclosest]
                sorty = np.argsort(y)
                y, z = y[sorty], z[sorty]
                gridded_fault = np.interp(ygrid, y, z, left=np.nan, right=np.nan)
            else:
                Xgrid = xgrid * np.ones(ny)
                Ygrid = ygrid
                gridded_fault = spint.griddata((self.y, self.x), self.z,
                                               (Ygrid, Xgrid),
                                               fill_value=np.nan,
                                               method='linear')
        else:
            ny, nx = 1, len(xgrid)
            yclosest = np.unique(self.y)[findclosest(np.unique(self.y), ygrid)]
            if sum(self.y == yclosest) >= 2:
                if np.abs(yclosest - ygrid) > mindist:
                    return np.full(nx, np.nan)
                x = self.x[self.y == yclosest]
                z = self.z[self.y == yclosest]
                sortx = np.argsort(x)
                x, z = x[sortx], z[sortx]
                gridded_fault = np.interp(xgrid, x, z, left=np.nan, right=np.nan)
            else:
                ny, nx = 1, len(xgrid)
                Xgrid = xgrid
                Ygrid = ygrid * np.ones(nx)
                gridded_fault = spint.griddata((self.y, self.x), self.z,
                                               (Ygrid, Xgrid),
                                               fill_value=np.nan,
                                               method='linear')

        if ny == 1 or nx == 1:
            if quickplot:
                plt.figure(figsize=(10, 10))
                plt.plot(gridded_fault)
        else:
            gridded_fault = gridded_fault.reshape(ny, nx)
            if mindist is not None:
                tree = KDTree(np.c_[self.y, self.x])
                dist, _ = tree.query(np.c_[Ygrid.ravel(), Xgrid.ravel()], k=1)
                dist = dist.reshape(Ygrid.shape)
                mask = np.zeros_like(dist)
                mask[dist > mindist] = 1
                gridded_fault = np_ma.masked_array(gridded_fault, mask=mask)
            if quickplot:
                if clim is None:
                    clim = [np.nanmin(gridded_fault),
                            np.nanmax(gridded_fault)]
                plt.figure(figsize=(10, 10))
                plt.scatter(Xgrid.ravel(), Ygrid.ravel(),
                            c=gridded_fault.flatten(),
                            vmin=clim[0], vmax=clim[1])
                plt.scatter(self.x,
                            self.y,
                            c=self.z,
                            s=50, edgecolors='k',
                            vmin=clim[0], vmax=clim[1])
                plt.xlabel('X')
                plt.ylabel('Y')
        return gridded_fault

    def resample(self, xpoints, ypoints, mindist=None, quickplot=False,
                 ax=None, clim=None):
        """Resample fault

        Return fault depths given a new set of x-y points

        Parameters
        ----------
        xpoints : :obj:`np.ndarray`
            X coordinates of points where fault is resampled
        ypoints : :obj:`int`, optional
            Y coordinates of points where fault is resampled
        mindist : :obj:`float`, optional
            Minimum distance from fault point for interpolated grid (points
            whose distance is bigger than mindist will be masked out)
        quickplot : :obj:`bool`, optional
            Quickplot
        ax : :obj:`plt.axes`
            Axes handle
        clim : :obj:`float`, optional
             Colorbar limits (if ``None`` infer from data and
             apply ``clip`` to those)

        Returns
        -------
        resampled_fault : :obj:`np.ndarray`
            Resampled fault

        """
        resampled_fault = spint.griddata((self.y, self.x), self.z,
                                         (ypoints, xpoints),
                                         fill_value=np.nan,
                                         method='cubic')
        if mindist is not None:
            tree = KDTree(np.c_[self.y, self.x])
            dist, _ = tree.query(np.c_[ypoints, xpoints], k=1)
            mask = np.zeros_like(dist)
            mask[dist > mindist] = 1
            resampled_fault = np_ma.masked_array(resampled_fault, mask=mask)
        if quickplot:
            if clim is None:
                clim = [np.nanmin(resampled_fault), np.nanmax(resampled_fault)]
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(xpoints, ypoints,
                       c=resampled_fault.flatten(),
                       vmin=clim[0], vmax=clim[1])
            ax.scatter(self.x, self.y, c=self.z,
                       s=50, edgecolors='k',
                       vmin=clim[0], vmax=clim[1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        return resampled_fault

    #########
    # Viewers
    #########
    def view(self, ax=None, cmap='rainbow', color=None, axiskm=False,
             figsize=(10, 7), title=None):
        """Quick visualization of Fault object.

        Parameters
        ----------
        ax : :obj:`plt.axes`
            Axes handle. If ``None`` make a new figure.
        cmap : :obj:`str`, optional
             Colormap to use to display fault
        color : :obj:`str`, optional
             Color to use to display fault. If not provided use ``cmap``
             instead
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)
        interp : :obj:`str`, optional
             imshow interpolation
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``ax`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handle

        """
        scaling = 1000. if axiskm else 1.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        if color is None:
            ax.scatter(self.x / scaling, self.y / scaling, c=self.z, cmap=cmap)
        else:
            ax.scatter(self.x / scaling, self.y / scaling, c=color)
            #ax.plot(self.x / scaling, self.y / scaling, c=color, lw=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if title is not None:
            ax.set_title(title)
        return fig, ax
