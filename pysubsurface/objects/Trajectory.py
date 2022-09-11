import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


class Trajectory:
    """Trajectory object.

    This object contains a well trajectory arranged
    it into a :obj:`pd.DataFrame`.

    Parameters
    ----------
    filename : :obj:`str`
        Name of file containing picks to be read (if ``None``, create an empty
        :class:`pysubsurface.objects.Picks` object and fill it with ``pick`` dataframe
        or leave it to the user to fill it manually)
    wellname : :obj:`str`, optional
        Name of well (if ``None`` use filename)
    loadtraj : :obj:`bool`, optional
        Read data and load it into ``df`` attribute of class as part of
        initialization (``True``) or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`int`, optional
        Verbosity

    """
    def __init__(self, filename=None, wellname=None, loadtraj=True,
                 kind='local', verb=False):
        self.filename = filename
        self.wellname = \
            filename.split('/')[-1] if wellname is None else wellname
        self._loadtraj = loadtraj
        self._kind = kind
        self._verb = verb

        if self._loadtraj:
            self._read_traj()

    def __str__(self):
        descr = 'Trajectory (original file {})\n\n'.format(self.filename) + \
                str(self.df)
        return descr

    def _read_traj(self):
        """Read trajectory from file
        """
        if self._verb:
            print('Reading well {} trajectory...'.format(self.wellname))
        if self._kind == 'local':
            self.df = pd.read_csv(self.filename, header=1)
            self.df['TVDSS'] = -self.df['TVDSS']
        else:
            raise NotImplementedError('kind must be local')

    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display trajectory table

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        if ipython_flag:
            display(self.df.head(nrows))
        else:
            print(self.df.head(nrows))

    def view_traj(self, ax=None, color='k',
                  labels=True, labelcoords=None,
                  wellname=True, flipaxis=False, shift=(0, 0),
                  fontsize=10, bbox=False,
                  grid=False, axiskm=False, checkwell=False,
                  figsize=(6, 5), title=None):
        """Visualize well trajectory

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handle
        color : :obj:`str`, optional
            Trajectory color
        labels : :obj:`bool`, optional
            Add labels to x and y axis (``True``) or not (``False``)
        labelcoords : :obj:`tuple`, optional
            x and y coordinates for label (if ``None`` use well head)
        wellname : :obj:`tuple`, optional
            Add wellname (``True``) or not (``False``)
        flipaxis : :obj:`bool`, optional
            Plot x coordinates on x-axis and y coordinates on y-axis (``False``)
            or viceversa (``True``)
        shift : :obj:`tuple`, optional
            Shift to be applied to label in (x, y)
        fontsize : :obj:`float`, optional
            Size of wellname
        bbox : :obj:`bool`, optional
            Add box around labels
        grid : :obj:`bool`, optional
            Add grid (``True``) or not (``False``)
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)
        checkwell : :obj:`bool`, optional
            Check if well is inside axis limits (``True``) or plot
            anyways (``False``)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if flipaxis:
            shift = (shift[0], shift[1])
            if labelcoords is not None:
                labelcoords = (labelcoords[1], labelcoords[0])
        axisscale = 1000. if axiskm else 1.

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.suptitle('Trajectory curve from {}'.format(self.filename.split('/')[-1])
                         if title is None else title, fontweight='bold')
            wellinside = True
        else:
            ax_x, ax_y = sorted(ax.get_xlim()), sorted(ax.get_ylim())
            fig = None
            # check if wells is inside axis when provided
            if checkwell:
                wellinside = self.df['X Absolute'].iloc[0] / axisscale > ax_x[0] and \
                             self.df['X Absolute'].iloc[0]  / axisscale < ax_x[1] and \
                             self.df['Y Absolute'].iloc[0]  / axisscale > ax_y[0] and \
                             self.df['Y Absolute'].iloc[0]  / axisscale < ax_y[1]
            else:
                wellinside = True
        if wellinside:
            ax.plot(self.df['Y Absolute']/axisscale if flipaxis else
                    self.df['X Absolute']/axisscale,
                    self.df['X Absolute']/axisscale if flipaxis else
                    self.df['Y Absolute']/axisscale,
                    color, lw=2)
            ax.scatter(self.df['Y Absolute'].iloc[0]/axisscale if flipaxis else
                       self.df['X Absolute'].iloc[0]/axisscale,
                       self.df['X Absolute'].iloc[0]/axisscale if flipaxis else
                       self.df['Y Absolute'].iloc[0]/axisscale, c=color, s=50)

            if wellname:
                if labelcoords is None:
                    labelcoords = (self.df['Y Absolute'].iloc[-1]/axisscale if flipaxis else
                                   self.df['X Absolute'].iloc[-1] / axisscale,
                                   self.df['X Absolute'].iloc[-1]/axisscale if flipaxis else
                                   self.df['Y Absolute'].iloc[-1] / axisscale)
                ax.text(labelcoords[0] + shift[0],
                        labelcoords[1] + shift[1],
                        self.wellname,
                        ha="center", va="center", color=color, fontsize=fontsize,
                        bbox=None if bbox is False else dict(boxstyle="round",
                                                            fc=(1., 1., 1.),
                                                            ec='k', alpha=0.9))
            if labels:
                if axiskm:
                    if flipaxis:
                        ax.set_xlabel('Y (km)'), ax.set_ylabel('X (km)')
                    else:
                        ax.set_xlabel('X (km)'), ax.set_ylabel('Y (km)')
                else:
                    if flipaxis:
                        ax.set_xlabel('Y (m)'), ax.set_ylabel('X (m)')
                    else:
                        ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)')
            if grid:
                ax.grid()
        if fig is not None:
            plt.tight_layout()
        return fig, ax

    def view_mdtvdss(self, ax=None,  color='k', labels=True,
                     figsize=(6, 5), title=None):
        """Visualize MD-TVDSS relation

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handle
        color : :obj:`str`, optional
            Curve color
        labels : :obj:`tuple`, optional
            Add labels to x and y axis (``True``) or not (``False``)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.suptitle('MD-TVDSS relation from {}'.format(
                self.filename.split('/')[-1]) if title is None else title,
                         fontweight='bold')
        else:
            fig = False
        ax.plot(self.df['MD (meters)'], self.df['TVDSS'], color, lw=2)
        if labels:
            ax.set_xlabel('MD (m)'), ax.set_ylabel('TVDSS (m)')
        ax.grid()
        ax.invert_yaxis()
        return fig, ax

    def view(self, ax=None, figsize=(10, 5), title=None, savefig=None):
        """Visualize well trajectory and MD-TVDSS relation
        in two adjacient plots

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handle
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        ax : :obj:`plt.axes`, optional
             Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)

        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle('Trajectory and MD-TVDSS relation '
                         'from {}'.format(self.filename.split('/')[-1])
                         if title is None else title,
                         y=0.98, fontweight='bold')
        else:
            fig = False
        self.view_traj(ax=ax[0], color='k')
        self.view_mdtvdss(ax=ax[1], color='k')
        plt.subplots_adjust(bottom=0.2)
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax
