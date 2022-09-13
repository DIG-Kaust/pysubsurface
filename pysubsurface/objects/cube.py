import copy
import numpy as np
import matplotlib.pyplot as plt

def _cube_view(data, axs=None, y=None, x=None, tz=None,
               iy=None, ix=None, it=None, scale=1.,
               clip=1., clim=None, cmap='seismic',
               cbar=False, interp=None, annotation=None,
               figsize=(13, 6), title=None, savefig=None):
    """Quick visualization of Cube object. Refer to
    :func:`pysubsurface.objects.Cube.view` for detailed documentation.
    """
    if annotation is None:
        annotation = \
            'Cube dimensions: ny={}, nx={}, nt={}\n\n' \
            'Figpath: {}:'.format(data.ny, data.nx, data.nt,
                                  savefig if savefig is not None
                                  else 'fig not saved')

    # define y axis
    if y is None:
        y = data.y
        ymin, ymax = data.y.min() - data.dy / 2., data.y.max() + data.dy / 2.
    else:
        dy = abs(y[1] - y[0])
        ymin, ymax = y[0] - dy / 2., y[-1] + dy / 2.

    # define x axis
    if x is None:
        x = data.x
        xmin, xmax = data.x.min() - data.dx / 2., data.x.max() + data.dx / 2.
    else:
        dx = abs(x[1] - x[0])
        xmin, xmax = x[0] - dx / 2., x[-1] + dx / 2.

    # define tz axis
    if tz is None:
        tz = data.t
        tmin, tmax = data.t.min() - data.dt / 2., data.t.max() + data.dt / 2.
    else:
        dtz = abs(tz[1] - tz[0])
        tmin, tmax = tz[0] - dtz / 2., tz[-1] + dtz / 2.

    if iy is None:
        iy = int(data.ny / 2)
    if ix is None:
        ix = int(data.nx / 2)
    if it is None:
        it = int(data.nt / 2)

    # define clims
    if clim is None:
        clim = [-clip * np.nanmax(scale * data.data),
                clip * np.nanmax(scale * data.data)]

    # create figure if axs is not provided
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=20, fontweight='bold')
    else:
        fig = None
        axs = [[axs[0], axs[1]],
               [axs[2], axs[3]]]

    # fill figure with different cube slices
    im = axs[0][0].imshow(scale * data.data[iy, :, :].T, aspect='auto',
                          interpolation=interp, cmap=cmap,
                          vmax=clim[1], vmin=clim[0],
                          extent=(xmin, xmax, tmax, tmin))
    axs[0][0].axhline(tz[it], c='k', ls='--')
    axs[0][0].axvline(x[ix], c='k', ls='--')
    axs[0][0].set_title('Slice at iy=%d' % y[iy], fontsize=13)
    axs[0][0].set_xlabel(r'$x$', fontsize=12)
    axs[0][0].set_ylabel(r'$t$', fontsize=12)
    axs[0][0].axis('tight')

    im = axs[0][1].imshow(scale * data.data[:, ix, :].T, aspect='auto',
                          interpolation=interp, cmap=cmap,
                          vmax=clim[1], vmin=clim[0],
                          extent=(ymin, ymax, tmax, tmin))
    axs[0][1].axhline(tz[it], c='k', ls='--')
    axs[0][1].axvline(y[iy], c='k', ls='--')
    axs[0][1].set_title('Slice at ix=%d' % x[ix], fontsize=13)
    axs[0][1].set_xlabel(r'$y$', fontsize=12)
    axs[0][1].set_ylabel(r'$t$', fontsize=12)
    axs[0][1].axis('tight')

    im = axs[1][0].imshow(scale * data.data[:, :, it], aspect='auto',
                          interpolation=interp, cmap=cmap,
                          vmax=clim[1], vmin=clim[0],
                          extent=(xmin, xmax, ymax, ymin))
    axs[1][0].axhline(y[iy], c='k', ls='--')
    axs[1][0].axvline(x[ix], c='k', ls='--')
    axs[1][0].set_title('Slice at it=%d' % tz[it], fontsize=13)
    axs[1][0].set_xlabel(r'$x$', fontsize=12)
    axs[1][0].set_ylabel(r'$y$', fontsize=12)
    axs[1][0].axis('tight')

    axs[1][1].set_axis_off()
    axs[1][1].text(0.0, 0.4, annotation, color='black', fontsize=11)
    if cbar:
        plt.colorbar(im, ax=axs[1][0])
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)

    return fig, axs


class Cube:
    r"""Cube object.

    Create object containing three-dimensional data along unitary axis.
    For simplicity, y and x axes are centered around zero. This object is a
    convenience object to be used in various algorithms when information about
    the actual axes is not stricly needed.

    Parameters
    ----------

    data : :obj:`np.ndarray`
        Data organized as math:`[ny \times nx \times nt]`

    Attributes
    ----------

    y : :obj:`np.ndarray`
        y-axis
    x : :obj:`np.ndarray`
        x-axis
    t : :obj:`np.ndarray`
        t-axis

    """
    def __init__(self, data):
        # dimensions
        self.ny, self.nx, self.nt = data.shape

        # data
        self.data = data.copy()

        # axes
        self.y = np.arange(self.ny) - self.ymid
        self.x = np.arange(self.nx) - self.xmid
        self.t = np.arange(self.nt)

        # sampling
        self.dy, self.dx, self.dt = 1, 1, 1

    def __str__(self):
        descr = 'Cube object:\n' + \
                'ny={}, nx={}, nt={}\n'.format(self.ny, self.nx, self.nt) + \
                'min={0:.3f},\nmax={1:.3f}\n'.format(self.data.min(),
                                                     self.data.max()) + \
                'mean={0:.3f},\nstd={1:.3f}'.format(np.mean(self.data),
                                                    np.std(self.data))
        return descr

    @property
    def ymid(self):
        if self.ny % 2 == 1:
            _ymid = (self.ny - 1) / 2
        else:
            _ymid = self.ny / 2
        return _ymid

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
        cubecopy : :obj:`pysubsurface.objects.Cube`
            Copy of Cube object

        """
        cubecopy = copy.deepcopy(self)
        if empty:
            cubecopy.data = np.zeros((self.ny, self.nx, self.nt))
        else:
            cubecopy.data = np.copy(self.data)
        return cubecopy


    #########
    # Viewers
    #########
    def view(self, axs=None, y=None, x=None, tz=None,
             iy=None, ix=None, it=None, scale=1.,
             clip=1., clim=None, cmap='seismic',
             cbar=False, interp=None, annotation=None,
             figsize=(9, 9), title=None, savefig=None):
        """Quick visualization of Cube object.

        Parameters
        ----------
        axs : :obj:`tuple` or :obj:`list`, optional
            Four axes handles for visualization of y-slice, x-slice, and t-slice
            and auxiliary text (if ``None`` draw a new figure)
        y : :obj:`np.ndarray`, optional
            y-axis for visualization (if ``None`` infer from data)
        x : :obj:`np.ndarray`
            x-axis for visualization (if ``None`` infer from data)
        tz : :obj:`np.ndarray`, optional
            time/depth-axis for visualization (if ``None`` infer from data)
        iy : :obj:`np.ndarray`, optional
            Index of slice to visualize in y (1st) direction
            (if ``None`` display slice in the middle)
        ix : :obj:`np.ndarray`, optional
            Index of slice to visualize in x (2nd) direction
            (if ``None`` display slice in the middle)
        it : :obj:`np.ndarray`, optional
            index of slice to visualize in time/depth (3rd) direction
            (if ``None`` display slice in the middle)
        scale : :obj:`float`, optional
            Apply temporary scaling to data for visualization
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
        return _cube_view(self, axs=axs, y=y, x=x, tz=tz,
                          iy=iy, ix=ix, it=it, scale=scale,
                          clip=clip, clim=clim, cmap=cmap, cbar=cbar,
                          interp=interp, annotation=annotation,
                          figsize=figsize, title=title, savefig=savefig)
