import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display

    ipython_flag = True
except:
    ipython_flag = False


def _split_lines(lines):
    return [(line.split()) for line in lines]


class PolygonSet():
    def __init__(self, filename, polygonname=None,
                 kind='local', verb=False):
        self.filename = filename
        self.polygonname = \
            filename.split('/')[-1] if polygonname is None else polygonname
        self._kind = kind
        self._verb = verb
        self._read_polygonset()

    def __str__(self):
        descr = 'Polygon set (original file {}): '.format(self.filename) + \
                '{} polygons'.format(len(self.polys))
        return descr

    def _read_polygonset(self):
        """Read polygon set from file
        """
        if self._verb:
            print('Reading polygon set {}...'.format(self.filename))
        if self._kind == 'local':
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        else:
            raise NotImplementedError('kind must be local')
        indices = [i for i, line in enumerate(lines) if
                   'Mapping Polygon' in line][3:]

        poly_ins = np.array(indices) + 2
        poly_ends = np.array(indices[1:]) - 1
        poly_ends = np.append(poly_ends, -1)

        self.polys = [lines[poly_in:poly_end] for poly_in, poly_end in
                      zip(poly_ins, poly_ends)]
        self.polys = [pd.DataFrame(np.array(_split_lines(poly)).astype(np.float),
                                   columns=['X Absolute', 'Y Absolute',
                                            'TVDSS']) for poly in self.polys]

    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display polygon set tables

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        if ipython_flag:
            for ipoly, poly in enumerate(self.polys):
                print('Polygon {}'.format(ipoly))
                display(poly.head(nrows))
        else:
            for ipoly, poly in enumerate(self.polys):
                print('Polygon {}'.format(ipoly))
                print(poly.head(nrows))

    def view(self, ax, color='k', flipaxis=False, axiskm=False, lw=2):
        """Visualize polygon set

        Parameters
        ----------
        ax : :obj:`plt.axes`
            Axes handle
        color : :obj:`str`, optional
            Trajectory color
        flipaxis : :obj:`bool`, optional
            Plot x coordinates on x-axis and y coordinates on y-axis (``False``)
            or viceversa (``True``)
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)
        lw : :obj:`float`, optional
            Linewidth
        Returns
        -------
        ax : :obj:`plt.axes`
            Axes handle

        """
        axisscale = 1000. if axiskm else 1.

        for poly in self.polys:
            if flipaxis:
                ax.fill(poly['Y Absolute'] / axisscale,
                        poly['X Absolute'] / axisscale, color=color, lw=lw)
            else:
                ax.fill(poly['X Absolute'] / axisscale,
                        poly['Y Absolute'] / axisscale, color=color, lw=lw)
        return ax
