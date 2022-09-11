import copy
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')


class Slice:
    r"""Slice object.

    Create object containing three-dimensional data along unitary axis.
    For simplicity, x axis is centered around zero. This object is a
    convenience object to be used in various algorithms when information about
    the actual axes is not stricly needed.

    Parameters
    ----------

    data : :obj:`np.ndarray`
        Data organized as math:`[nx \times nt]`

    Attributes
    ----------

    x : :obj:`np.ndarray`
        x-axis
    t : :obj:`np.ndarray`
        t-axis

    """
    def __init__(self, data):
        # dimensions
        self.nx, self.nt = data.shape

        # data
        self.data = data.copy()

        # axes
        self.x = np.arange(self.nx) - self.xmid
        self.t = np.arange(self.nt)

        # sampling
        self.dx, self.dt = 1, 1

    def __str__(self):
        descr = 'Slice object:\n' + \
                'nx={}, nt={}\n'.format(self.nx, self.nt) + \
                'min={0:.3f},\nmax={1:.3f}\n'.format(self.data.min(),
                                                     self.data.max()) + \
                'mean={0:.3f},\nstd={1:.3f}'.format(np.mean(self.data),
                                                    np.std(self.data))
        return descr

    @property
    def xmid(self):
        if self.nx % 2 == 1:
            _xmid = (self.nx - 1) / 2
        else:
            _xmid = self.nx / 2
        return _xmid

    def copy(self, empty=False):
        """Return a copy of the object.

        Parameters
        ----------
        empty : :obj:`bool`
            Copy input data (``True``) or just create an empty data (``False``)

        Returns
        -------
        cubecopy : :obj:`pysubsurface.objects.Slice`
            Copy of Slice object

        """
        slicecopy = copy.deepcopy(self)
        if empty:
            slicecopy.data = np.zeros((self.nx, self.nt))
        else:
            slicecopy.data = np.copy(self.data)
        return slicecopy


    #########
    # Viewers
    #########
    def view(self, ax=None, x=None, tz=None,
             clip=1., clim=None, cmap='seismic',
             cbar=False, interp=None,
             figsize=(20, 6), title='', savefig=None):
        """Quick visualization of Slice object.

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axis handle for visualization (if ``None`` draw a new figure)
        x : :obj:`np.ndarray`
            x-axis for visualization (if ``None`` infer from data)
        tz : :obj:`np.ndarray`, optional
            time/depth-axis for visualization (if ``None`` infer from data)
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
        annotation : :obj:`str`, optional
             Text to write in bottom right corner (if ``None`` infer from data)
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

        # define tz axis
        if tz is None:
            tmin, tmax = self.t.min() - self.dt/2., self.t.max() + self.dt/2.
        else:
            dtz = abs(tz[1]-tz[0])
            tmin, tmax = tz[0] - dtz/2., tz[-1] + dtz/2.

        # define x axis
        if x is None:
            xmin, xmax = self.x.min() - self.dx/2., self.x.max() + self.dx/2.
        else:
            dx = abs(x[1]-x[0])
            xmin, xmax = x[0] - dx/2., x[-1] + dx/2.

        # define limits of colorbar
        if clim is None:
            clim = [-clip * np.nanmax(self.data), clip * np.nanmax(self.data)]

        # plotting
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig= None
        im = ax.imshow(self.data.T, aspect='auto', interpolation=interp,
                       cmap=cmap, vmax=clim[1], vmin=clim[0],
                       extent=(xmin, xmax, tmax, tmin))
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$t$', fontsize=14)
        if cbar:
            plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig, dpi=300)
        return fig, ax
