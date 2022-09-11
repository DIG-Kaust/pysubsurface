import pandas as pd
import matplotlib.pyplot as plt


try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


_FORMATS = {'dsg': {'skiprows': 166,
                    'xpos': 0, 'ypos': 1, 'zpos':4,
                    'nanvals':-999, 'ilxl': True}}


class Polygon:
    """Polygon object.

    This object contains a polygon arranged
    it into a :obj:`pd.DataFrame`.

    Parameters
    ----------
    filename : :obj:`str`
        Name of file containing polygon to be read
    polygonname : :obj:`str`, optional
        Name of polygon (if ``None`` use filename)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`int`, optional
        Verbosity

    """
    def __init__(self, filename, polygonname=None,
                 format='dsg', kind='local', verb=False):
        self.filename = filename
        self.polygonname = \
            filename.split('/')[-1] if polygonname is None else polygonname
        self._kind = kind
        self._verb = verb

        # evaluate format
        if format in _FORMATS.keys():
            self.format = format
            self.format_dict = _FORMATS[self.format].copy()
        else:
            raise ValueError('{} not contained in list of available'
                             ' formats=[{}]'.format(format, ' '.join(_FORMATS)))
        self._read_polygon()

    def __str__(self):
        descr = 'Polygon (original file {})\n\n'.format(self.filename) + \
                str(self.df)
        return descr

    def _read_polygon(self):
        """Read polygon from file
        """
        if self._verb:
            print('Reading polygon {}...'.format(self.filename))
        if self._kind == 'local':
            self.df = \
                pd.read_csv(self.filename,
                            skiprows=int(self.format_dict['skiprows']),
                            delim_whitespace=True, index_col=False,
                            names=['X Absolute', 'Y Absolute', 'TVDSS'])

        else:
            raise NotImplementedError('kind must be local')

    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display polygon table

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        if ipython_flag:
            display(self.df.head(nrows))
        else:
            print(self.df.head(nrows))

    def view(self, ax, color='k', polyname=True,
             flipaxis=False, bbox=False, axiskm=False):
        """Visualize polygon

        Parameters
        ----------
        ax : :obj:`plt.axes`
            Axes handle
        color : :obj:`str`, optional
            Trajectory color
        polyname : :obj:`bool`, optional
            Add polygon name (``True``) or not (``False``)
        flipaxis : :obj:`bool`, optional
            Plot x coordinates on x-axis and y coordinates on y-axis (``False``)
            or viceversa (``True``)
        bbox : :obj:`bool`, optional
            Add box around labels
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)

        Returns
        -------
        ax : :obj:`plt.axes`
            Axes handle

        """
        axisscale = 1000. if axiskm else 1.

        if flipaxis:
            ax.plot(self.df['Y Absolute']/axisscale,
                    self.df['X Absolute']/axisscale, color=color)
        else:
            ax.plot(self.df['X Absolute']/axisscale,
                    self.df['Y Absolute']/axisscale, color=color)
        if polyname:
            ax.text(self.df.iloc[0]['Y Absolute']/axisscale if flipaxis else
                    self.df.iloc[0]['X Absolute']/axisscale,
                    self.df.iloc[0]['X Absolute']/axisscale if flipaxis else
                    self.df.iloc[0]['Y Absolute']/axisscale,
                    self.polygonname,
                    ha="center", va="center", color=color,
                    bbox=None if bbox is False else dict(boxstyle="round",
                                                         fc=(1., 1., 1.),
                                                         ec='k', alpha=0.9))
        return ax
