import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysubsurface.visual.utils import _rgb2hex

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


def _rgb2hex_protected(r,g,b):
    """From rgb to hexadecimal color with check if

    Parameters
    ----------
    r : :obj:`str`
        Red
    g : :obj:`str`
        Green
    b : :obj:`str`
        Blue

    Returns
    -------
    hex : :obj:`str`
        Hexadecimal color
    """
    if np.isnan(r) or np.isnan(r) or np.isnan(r):
        #force black
        hex = '#000000'
    else:
        hex = _rgb2hex(r,g,b)
    return hex


class Intervals:
    """Intervals object.

    This object stores a dataframe with one interval per row and the
    following columns: ``Name``, ``Top``, ``Base``, ``Level``, and ``Color``.
    See :func:`pysubsurface.objects.Intervals.add_interval` routine for details on
    the meaning of each column.

    Parameters
    ----------
    intervals : :obj:`pd.DataFrame`, optional
        Dataframe containing intervals (if ``None`` create empty dataframe)
    kind : :obj:`str`, optional
        ``local`` when intervals are provided via ``intervals`` input or add
        manually later on via :func:`pysubsurface.objects.Intervals.add_interval`,

    Attributes
    ----------
    df : :obj:`pd.DataFrame`
        Intervals

    """
    def __init__(self, intervals=None, kind='local'):
        self._kind = kind

        if self._kind == 'local':
            if intervals is None:
                self.df = pd.DataFrame(columns=['Name', 'Top', 'Base', 'Level',
                                                'Order', 'Parent', 'Color'])
            else:
                self.df = intervals.copy()

    def __str__(self):
        descr = 'Intervals\n\n' + \
                str(self.df)
        return descr

    @staticmethod
    def _define_order(df):
        """Identify order or intervals within each level based on their top age
        """
        levels = list(df[_stratunitlevel].unique())
        for level in levels:
            df_level = df[df[_stratunitlevel] == level]
            iarg = np.argsort(list(df_level['TOP_AGE']))
            order = np.empty(len(iarg))
            order[iarg] = np.arange(len(iarg))
            df['ORDER'][df[_stratunitlevel] == level] = order
        return df

    def add_interval(self, name, top, base, level, color,
                     order=None, parent=None, field=None):
        """Add interval

        Parameters
        ----------
        name : :obj:`str`
            Name of interval
        top : :obj:`str`
            Name of top pick
        base : :obj:`str`
            Name of base pick
        level : :obj:`str`
            Level of interval (increasing number means increasing detail,
            e.g. from unit to formation to subformation). Use ``level=None``
            when adding an interval which will not be considered as part of the
            unit-formation-subformation-etc. tree (e.g., seismic interval
            between two surfaces)
        color : :obj:`str`
            Color of interval
        order : :obj:`int`
            Order of interval within its level
        parent : :obj:`str`
            Name of parent interval, i.e. inverval in level-1
            (use ``None`` for level 0)
        field : :obj:`str`, optional
            Name of field

        """
        self.df = self.df.append(pd.DataFrame({'Name': name,
                                               'Top': top,
                                               'Base': base,
                                               'Level': level,
                                               'Order': order,
                                               'Parent': parent,
                                               'Color': color,
                                               'Field': field,
                                               'Govern Area': None}, index=[0]),
                                 ignore_index=True)


    #########
    # Viewers
    #########
    def display(self, filtname=None, nrows=10):
        """Display intervals

        Parameters
        ----------
        filtname : :obj:`str`, optional
            Regex filter on Name column
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        # filter intervals
        if filtname is not None:
            intervals_display = self.df[self.df['Name'].apply(lambda x:
                                                          filtname in x)]
        else:
            intervals_display = self.df

        # display
        if ipython_flag:
            display(intervals_display.head(nrows).style.applymap(lambda x:
                                                                 'background-color: {}'.format(x),
                                                                 subset=['Color']))
        else:
            print(intervals_display.head(nrows))

    def view(self, levelmax=None, field=None, govern_area=None, alpha=0.7,
             fontsize=10, figsize=(12, 10), title=None, savefig=None):
        """Visualize stratigraphic column

        Parameters
        ----------
        levelmax : :obj:`int`, optional
            Maximum level to display
        field : :obj:`str`, optional
            Name of field to visualize. If ``None``, the first field
            in alphabetic order will be selected
        govern_area : :obj:`str`, optional
            Name of govern area to visualize. If ``None``, the first govern area
            in alphabetic order will be selected
        alpha : :obj:`float`, optional
            Transparency
        fontsize : :obj:`float`, optional
             Font size for interval names
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
        # choose field
        if field is None:
            field = sorted(self.df['Field'].unique())[0]
        if field is not None:
            df_view = self.df[self.df['Field'] == field]
        else:
            df_view = self.df.copy()

        # choose govern area
        if govern_area is None:
            govern_area = sorted(df_view['Govern Area'].unique())[0]
        if govern_area is not None:
            df_view = df_view[df_view['Govern Area'] == govern_area]

        # remove intervals without level
        df_view = df_view.dropna(axis=0, subset=['Level'])

        # select only interval of interest
        if levelmax is not None:
            df_view = df_view[df_view['Level'] <= levelmax]

        # find out unique intervals
        levels = list(df_view['Level'].unique())
        levels.sort(reverse=True)
        nlevels = len(levels)
        df_view['_nsubintervals'] = 0
        df_view.loc[df_view['Level'] == max(levels), '_nsubintervals'] = 1

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle('Stratigraphy' if title is None else title,
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(len(df_view[df_view['Level'] == nlevels-1]), 0)

        # loop over levels and work on plotting on level column
        for level in levels:
            icum = 0  # cumulative counter to stack intervals
            intervals_in_level = df_view[df_view['Level'] == level]
            intervals_in_level.sort_values('Order', inplace=True)
            top_pick = intervals_in_level[intervals_in_level['Order'] == 0].iloc[0]['Top']
            # loop over intervals and add them to level column
            for i in range(len(intervals_in_level)):
                interval = intervals_in_level[intervals_in_level['Top'] ==
                                              top_pick]
                if len(interval) != 1:
                    raise ValueError('Cannot find interval '
                                     'for pick {}'.format(top_pick))
                if i < len(intervals_in_level)-1:
                    top_pick = intervals_in_level.iloc[i+1]['Top']

                # accumulate the number of subintervals for each interval
                if level>0:
                    df_view.loc[((df_view['Name'] == interval.iloc[0]['Parent']) & (df_view['Level'] == level-1)), '_nsubintervals'] = \
                        int(df_view[((df_view['Name'] == interval.iloc[0]['Parent'])
                                     & (df_view['Level'] == level-1))]['_nsubintervals']) + \
                        int(interval['_nsubintervals'])
                # plot interval
                if level == nlevels-1:
                    ax.fill_between([level/(nlevels+1), (level+2.)/(nlevels+1)],
                                    i, i+1,  facecolor=interval['Color'],
                                    alpha=alpha)
                    ax.axhline(i + 1, level / (nlevels+1), (level + 2.) /
                               (nlevels+1), color='k', lw=1)
                    ax.text((2*level + 2.) / (2*nlevels+2), (2*i + 1)/2.,
                            interval.iloc[0]['Name'], color='k',
                            ha='center',va='center',
                            family='sans-serif', fontweight='bold', fontsize=fontsize)
                else:
                    # find out out many subintervals in level-1 belong to this interval
                    nsubintervals = int(df_view[(df_view['Level'] == level) & (df_view['Name'] == interval.iloc[0]['Name'])].iloc[0]['_nsubintervals'])
                    ax.fill_between([level / (nlevels+1.), (level + 1.) / (nlevels+1)],
                                    icum, icum + nsubintervals,
                                    facecolor=interval['Color'],
                                    alpha=alpha)
                    ax.axhline(icum + nsubintervals,
                               level / (nlevels+1.), (level + 1.) / (nlevels+1.),
                               color='k', lw=1)
                    ax.text((2*level+1) / (2*nlevels+2), (2 * icum + nsubintervals)/2.,
                            interval.iloc[0]['Name'], color='k',
                            ha='center', va='center',
                            family='sans-serif', fontweight='bold', fontsize=fontsize)
                    icum = icum + nsubintervals
            ax.axvline((level)/(nlevels+1.), 0,
                       len(df_view[df_view['Level'] == nlevels-1]),
                       color='k', lw=1)

        ax.axvline(0.999, 0, len(df_view[df_view['Level'] == nlevels - 1]),
                   color='k', lw=1)
        ax.axhline(len(df_view[df_view['Level'] == nlevels-1])-0.01, 0, 1,
                   color='k', lw=1)
        ax.axhline(0.01, 0, 1, color='k', lw=1)
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax