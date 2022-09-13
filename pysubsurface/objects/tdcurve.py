import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


class TDcurve:
    """TD curve (or checkshots) object.

    This object simply stores a Time-Depth (TD) curve (or checkshots)
    for a single well.

    Parameters
    ----------
    filename : :obj:`str`, optional
        Name of full path to file containing TD curve or checkshots to be read
    name : :obj:`str`, optional
            Name to give to TD curve or checkshots (if ``None`` use filename)
    loadtd : :obj:`int`, optional
        Load data into ``self.tdcurve`` variable during initialization
        (``True``) or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`int`, optional
        Verbosity

    """
    def __init__(self, filename, name=None, loadtd=True,
                 kind='local', ads=None, verb=False):
        self.filename = filename
        self.name = filename if name is None else name
        self._loadtd = loadtd
        self._kind = kind
        self._verb = verb

        if self._loadtd:
            self._read_td()

    def __str__(self):
        descr = 'TD Curve (original file {})\n\n'.format(self.filename) + \
                str(self.df)
        return descr

    def _read_td(self):
        """Read TD curve from file
        """
        if self._verb:
            print('Reading {} TD curve...'.format(self.filename))
        if self._kind == 'local':
            self.df = pd.read_csv(self.filename, header=1)
        else:
            raise NotImplementedError('kind must be local')

    def display(self, nrows=10):
        """Display TD curve table

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        if ipython_flag:
            display(self.df.head(nrows))
        else:
            print(self.df.head(nrows))

    def view_td(self, ax=None, figsize=(3, 8), title=None):
        """Visualize TD curve as function of depth

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handles (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``ax`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handles

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
        else:
            fig = None

        ax.plot(self.df['Time (ms)'], self.df['Depth (meters)'],
                '.-k', lw=2)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Depth (meters)')
        ax.set_title('TD curve from {}'.format(self.filename.split('/')[-1])
                     if title is None else title, fontweight='bold')
        ax.grid()

        return fig, ax

    def view_vd(self, ax=None, figsize=(3, 8), title=None):
        """Visualize TD curve and velocity as function of depth

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handles (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``ax`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handles

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
        else:
            fig = None

        ax.plot(self.df['Velocity'], self.df['Depth (meters)'],
                    '.-k', lw=2)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_title('Vint-depth curve from {}'.format(self.filename.split('/')[-1])
                     if title is None else title, fontweight='bold')
        ax.grid()
        return fig, ax

    def view(self, ax=None, figsize=(6, 8), title=None, savefig=None):
        """Visualize TD curve and velocity as function of depth

        Parameters
        ----------
        ax : :obj:`plt.axes`, optional
            Axes handles (if ``None`` draw a new figure)
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
            Figure handle (``None`` if ``ax`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handles

        """
        if ax is None:
            fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
            fig.suptitle('TD curve from {}'.format(self.filename.split('/')[-1])
                         if title is None else title, fontweight='bold', y=0.95)
        else:
            fig = None

        axs[0].plot(self.df['Time (ms)'], self.df['Depth (meters)'],
                    '.-k', lw=2)
        axs[0].set_xlabel('Time (ms)')
        axs[0].set_ylabel('Depth (meters)')
        axs[0].grid()
        axs[1].plot(self.df['Velocity'], self.df['Depth (meters)'],
                    '.-k', lw=2)
        axs[1].set_xlabel('Velocity (m/s)')
        axs[1].grid()
        if fig is None:
            axs[0].invert_yaxis()
        axs[1].invert_yaxis()
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax
