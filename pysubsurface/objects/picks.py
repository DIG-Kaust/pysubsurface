import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


class Picks:
    """Well picks object.

    This object loads well picks for entire field and arranges them into
    a :obj:`pd.DataFrame`.

    Parameters
    ----------
    filename : :obj:`str`, optional
        Name of file containing picks to be read (if ``None``, create an empty
        :class:`pysubsurface.objects.Picks` object and fill it with ``pick`` dataframe
        or leave it to the user to fill it manually)
    area : :obj:`str`, optional
        Name of area - Deprecated
    field : :obj:`str`, optional
        Name of field - Deprecated
    loadpicks : :obj:`bool`, optional
        Read data and load it into ``df`` attribute of class as part of
        initialization (``True``) or not (``False``)
    picks : :obj:`pd.DataFrame`, optional
        Dataframe containing picks (required when ``filename=None``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,
    verb : :obj:`int`, optional
        Verbosity

    """
    def __init__(self, filename=None, area=None, field=None,
                 loadpicks=True, picks=None,  kind='local',
                 verb=False):
        self.filename = filename
        self.area = area
        self.field = field
        self._loadpicks = loadpicks
        self._kind = kind
        self._verb = verb
        self.df = self._initialize_empty_picks()

        if picks is not None:
            self._create_picks(picks)
        elif filename is not None:
            if self._loadpicks:
                self._read_picks()

    def __str__(self):
        descr = 'Picks (original file' \
                ' {})\n\n'.format(self.filename) + \
                str(self.df)
        return descr

    @staticmethod
    def _initialize_empty_picks():
        """Initialize dataframe with empty picks
        """
        df = pd.DataFrame(columns=['Name',
                                   'Well UWI',
                                   'Depth (meters)',
                                   'Interp',
                                   'Color'])
        return df

    def _read_picks(self):
        """Read picks from input file
        """
        if self._verb:
            print('Reading picks from {}...'.format(self.filename))
        if self._kind == 'local':
            self.df = pd.read_csv(self.filename, delim_whitespace=True,
                                  index_col=False,
                                  names=['Name', 'Well UWI',
                                         'Depth (meters)', 'Interp'])
            self.df['Obs no'] = 1
            self.df['Field'] = self.field
        else:
            raise NotImplementedError('kind must be local')
        self.df['Well UWI'] = self.df['Well UWI'].str.replace('_', ' ')
        self.df['Color'] = '#000000'

    def _create_picks(self, picks):
        """Load picks from dataframe

        Parameters
        ----------
        pick : :obj:`pd.DataFrame`, optional
            Dataframe containing picks
        """
        if self._verb:
            print('Creating picks from dataframe')
        self.df = picks

    def concat_picks(self, other):
        """Concatenate picks from another object

        Parameters
        ----------
        other : :obj:`pysubsurface.objects.Picks`
            Picks object to concatenate

        """
        self.df = pd.concat([self.df, other.df])

    def add_pick(self, name, well, depth, tvdss=None, interp=None, color=None):
        """Add single pick manually

        Parameters
        ----------
        name : :obj:`str`
            Pick name
        well : :obj:`str`
            Pick name
        depth : :obj:`float`
            Measured depth in meters
        tvdss : :obj:`str`, optional
            TVDSS in meters
        interp : :obj:`str`, optional
            Interpreter
        color : :obj:`str`, optional
            Pick color

        """
        self.df = self.df.append(pd.DataFrame({'Name': name,
                                               'Well UWI': well,
                                               'Depth (meters)': depth,
                                               'TVDSS (meters)': tvdss,
                                               'Interp': interp,
                                               'Color': color}, index=[0]),
                                 ignore_index=True,
                                 sort=False).reset_index(drop=True)

    def select_interpreters(self, interpreters):
        """Select interpreter(s)

        Filter picks that belong to an interpreter or a list of interpreters.
        All picks belonging to other interpreters will be removed from the
        project

        Parameters
        ----------
        interpreters : :obj:`tuple` or  :obj:`str`
            List of interpreters to select

        """
        if isinstance(interpreters, str):
            interpreters = (interpreters, )
        if self._verb:
            print('Selecting picks from intepreters {}...'.format(interpreters))

        self.df = self.df[self.df['Interp'].apply(lambda x: x in interpreters)]
        self.df.reset_index(drop=True, inplace=True)

    def discard_on_keywords(self, keywords):
        """Remove picks that contain any string from keywords list in their Name

        Parameters
        ----------
        interpreters : :obj:`tuple` or  :obj:`str`
            List of keywords to exclude

        """
        if isinstance(keywords, str):
            keywords = (keywords,)
        self.df = \
            self.df[self.df['Name'].apply(lambda x:
                                          not any(keyword in x for
                                                  keyword in keywords))]
        self.df.reset_index(drop=True, inplace=True)

    def assign_color(self, name, color=None, intervals=None, verb=False):
        """Assign a color to a specific pick

        Parameters
        ----------
        name : :obj:`str`
            Pick name
        color : :obj:`str`, optional
            Pick color (if ``None`` infer from intervals object)
        intervals : :obj:`pysubsurface.objects.Intervals`, optional
            Intervals object
        verb : :obj:`bool`,optional
            Verbosity

        """
        if verb:
            print('Change color to {} into {}'.format(name, color))
        if color is None:
            interval = intervals.df[intervals.df['Top'] == name]
            if len(interval) > 0:
                self.df['Color'][self.df['Name'] == name] = interval.iloc[0]['Color']
            else:
                self.df['Color'][self.df['Name'] == name] = 'k'
        else:
            self.df['Color'][self.df['Name'] == name] = color

    def extract_from_well(self, well):
        """Extract picks for single well

        Parameters
        ----------
        well : :obj:`str`
            Well UWI name

        Returns
        -------
        wellpicks : :obj:`pysubsurface.object.Picks`
            New Picks object with picks from single well

        """
        wellpicks = Picks(picks = self.df[self.df['Well UWI'] == well])
        return wellpicks


    #########
    # Viewers
    #########
    def display(self, filtname=None, nrows=10):
        """Display picks table

        Parameters
        ----------
        filtname : :obj:`str`, optional
            Regex filter on Name column
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        # filter picks
        if filtname is not None:
            picks_display = self.df[self.df['Name'].apply(lambda x:
                                                          filtname in x)]
        else:
            picks_display = self.df

        # display
        if ipython_flag:
            if nrows is None:
                display(picks_display.style.applymap(lambda x:
                                                     'background-color: {}'.format(
                                                         x), subset=['Color']))
            else:
                display(picks_display.head(nrows).style.applymap(lambda x:
                                                                 'background-color: {}'.format(
                                                                     x),
                                                                 subset=['Color']))
        else:
            if nrows is None:
                print(picks_display)
            else:
                print(picks_display.head(nrows))

    def count_picks(self, nrows=10, plotflag=False, title=None, savefig=None):
        """Display (or plot) a count of all picks

        Picks with same name (from different wells) are grouped and a summary
        of the number of picks with unique name is created.

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)
        plotflag : :obj:`bool`, optional
            Plot histogram (``True``) or simply print the summary (``False``)
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)

        """
        picks_count = self.df['Name'].value_counts()
        if nrows is not None:
            picks_count = picks_count.iloc[:nrows]
        if plotflag:
            clrs = ['grey']*len(picks_count)
            clrs[0] = 'red'
            fig, ax = plt.subplots(1, 1, figsize=(9, 15))
            sns.barplot(picks_count, palette=clrs)
            ax.set_title('' if title is None else title,
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.05)
            if savefig is not None:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
        else:
            print(picks_count)

    def view(self, axs, depth='Depth', ylim=None, labels=True, pad=1.02):
        """Visualize picks on top of log curves

        For this routine to work sensibly, axes handles
        ``axs`` should be previously obtained from the
        :func:`pysubsurface.objects.Logs.visualize_logcurve` or
        :func:`pysubsurface.objects.Logs.visualize_logcurves` routines.

        Parameters
        ----------
        axs : :obj:`plt.axes`
            Axes handles
        depth : :obj:`str`, optional
            Domain to use for plotting of picks
            (``Depth``: MD, ``TVDSS``: TVDSS, ``TWT``: TWT)
        ylim : :obj:`tuple`, optional
            Extremes of y-axis (picks outside those extremes will be discarded)
        labels : :obj:`bool`, optional
            Display name of picks
        pad : :obj:`float`, optional
            Padding for labels

        Returns
        -------
        axs : :obj:`plt.axes`
            Axes handles

        """
        if depth == 'Depth':
            depth = depth+' (meters)'
        elif depth == 'TVDSS':
            depth = depth + ' (meters)'
        if depth == 'TWT':
            depth = depth + ' (ms)'

        if len(self.df) > 0:
            if ylim is None:
                picks_plot = self.df.copy()
            else:
                picks_plot = self.df[(self.df[depth] > ylim[0]) &
                                     (self.df[depth] < ylim[1])]

            # add picks
            for ipick, pick in picks_plot.iterrows():
                for ax in axs:
                    ax.axhline(pick[depth], color=pick['Color'], lw=4)

            # add pick labels
            if labels:
                for ipick, pick in picks_plot.iterrows():
                    axs[-1].text(pad * axs[-1].get_xlim()[1],
                                 pick[depth], pick['Name'],
                                 fontsize=8, fontweight='bold',
                                 color=pick['Color'],
                                 bbox=dict(fc='w', ec='w'))
        return axs