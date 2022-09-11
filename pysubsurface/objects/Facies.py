import logging

import numpy as np
from pysubsurface.utils.utils import findclosest
from pysubsurface.objects.Logs import _filters_curves


try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


class Facies:
    """Facies object.

    This object contains a facies definition composed of an interval and filters
    on properties.

    Parameters
    ----------
    faciesname : :obj:`str`
        Name of facies
    color : :obj:`str`
        Color to assign to facies
    intervalname : :obj:`str`, optional
        Name of interval or intervals to filter on
         (if ``None`` consider entire depth range and do not filter on interval)
    level : :obj:`str`
            Level of interval(s)
    filters : :obj:`list` or :obj:`tuple`
        Filters to be applied
        (each filter is a dictionary with logname and rule, e.g.
        logname='LFP_COAL', rule='<0.1' will keep all values where values
        in  LFP_COAL logs are <0.1)

    """
    def __init__(self, faciesname, color, intervalname=None,
                 level=None, filters=None):
        self.faciesname = faciesname
        self.color = color
        self.intervalname = intervalname
        self.level = level
        self.filters = filters
        if intervalname is None and filters is None:
            raise ValueError('provide at least one filtering conditions, either'
                             'interval or log filters...')

    def __str__(self):
        descr = 'Facies\n' \
                '------\n' + \
                'Name: {},\n' \
                'Color: {},\n' \
                'Interval: {},\n' \
                'Filters: {}'.format(self.faciesname, self.color,
                                     self.intervalname,
                                     self.filters)
        return descr

    def extract_mask_from_well(self, well):
        """Extract mask based on facies conditions from a well

        Parameters
        ----------
        well : :obj:`pysubsurface.objects.Well`
            Well
        level : :obj:`int`
            Interval level

        Returns
        -------
        mask : :obj:`np.ndarray`
            Mask indicating locations in well logs where conditions (interval
            and filters) are satisfied

        """
        # mask on filters
        if self.filters is not None:
            # mask on nans
            logs = [filter['logname'] for filter in self.filters]
            df_logs = well.welllogs.df[logs]
            maskava = ~df_logs.isnull().any(axis=1).values
            _, masklog =_filters_curves(well.welllogs.df, self.filters)
            mask = maskava & masklog
        # mask on interval
        if self.intervalname is not None:
            if well.intervals is None:
                raise ValueError('Well should contain intervals to be able to '
                                 'filter on interval')
            else:
                maskint = np.full(len(mask), False)
                if self.level is not None:
                    wellintervals = \
                        well.intervals[well.intervals['Level'] == self.level]
                else:
                    wellintervals = well.intervals.copy()
                if isinstance(self.intervalname, str):
                    self.intervalname = (self.intervalname, )
                for intervalname in self.intervalname:
                    interval = \
                    wellintervals[wellintervals['Name'] == intervalname]
                    md_top = interval['Top MD (meters)'].values
                    md_base = interval['Base MD (meters)'].values
                    if len(md_top)==0 or len(md_top)==0:
                        logging.warning('Cannot find '
                                        'interval {}'.format(self.intervalname))
                    elif len(md_top) > 1 or \
                            len(md_top) > 1:
                            logging.warning('More than one interval was '
                                            'found for interval '
                                            '{}'.format(self.intervalname))
                    else:
                        itop = findclosest(well.welllogs.df.index.values,
                                           md_top[0])
                        ibase = findclosest(well.welllogs.df.index.values,
                                            md_base[0])
                        maskint[itop:ibase] = True
                mask = mask & maskint

        return mask


    #########
    # Viewers
    #########
    def view_on_wellog(self, well, curve=None, depth='MD',
                       level=0, xlim=None, ylim=None,
                       ax=None, **kwargs_logplot):
        """Display well log with mask where facies conditions are satisfied

        Parameters
        ----------
        well : :obj:`pysubsurface.objects.Well`
            Well to display
        curve : :obj:`str`
            Name of log curve to display
        depth : :obj:`str`, optional
            Name of curve to use for depth axis
        xlim : :obj:`tuple`, optional
            Limits for x axis
        ylim : :obj:`tuple`, optional
            Limits for depth axis
        ax : :obj:`plt.axes`
            Axes handle (if provided only add the facies mask,
            if ``None`` draw a new figure and also add well curve)
        kwargs_logplot : :obj:`dict`, optional
            Additional parameters for
            :func:`pysubsurface.objects.Logs.visualize_logcurve`

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        mask = self.extract_mask_from_well(well)
        if well.welllogs.df.index.name == depth:
            depth_facies = well.welllogs.df.index
        else:
            depth_facies = well.welllogs.df[depth]

        if ax is None:
            fig, ax = well.welllogs.visualize_logcurve(curve, depth=depth,
                                                       xlim=xlim, ylim=ylim,
                                                       **kwargs_logplot)
            well.view_picks_and_intervals(ax,
                                          depth=depth,
                                          level=level, ylim=ylim,
                                          showintervals=False)
        else:
            fig = None
        if xlim is None:
            xlim = ax.get_xlim()
        ax.fill_betweenx(depth_facies,
                         xlim[0], xlim[1],
                         where=mask, alpha=1., color=self.color)
        if fig is not None:
            fig.subplots_adjust(left=0.2, right=0.5)

        return fig, ax