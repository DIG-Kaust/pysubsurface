import logging

import copy
import lasio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.colors as cplt

from pysubsurface.proc.seismicmod.poststack import zerooffset_wellmod
from pysubsurface.proc.seismicmod.avo import prestack_wellmod
from pysubsurface.visual.utils import _discrete_cmap, _discrete_cmap_indexed, \
    _wiggletrace, _wiggletracecomb

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


def _threshold_curve(curve, thresh, greater=True):
    """Apply thresholding to a curve

    Parameters
    ----------
    curve : :obj:`np.ndarray`
        Curve to be thresholded
    thresh : :obj:`float`
        Maximum allowed value (values above will be set to non-valid
    greater : :obj:`bool`, optional
        Apply threshold for values greater than ``thresh`` (``True``) or
        smaller than ``thresh`` (``False``)

    Returns
    -------
    threshcurve : :obj:`np.ndarray`
        Thresholded curve

    """
    threshcurve = np.copy(curve)
    if thresh is not None:
        if greater:
            threshcurve[threshcurve > thresh] = np.nan
        else:
            threshcurve[threshcurve < thresh] = np.nan
    return threshcurve

def _filters_curves(curves, filters):
    """Apply conditional filters to a set of log curves

    Parameters
    ----------
    curves : :obj:`pd.DataFrame`
        Set of log curves
    filters : :obj:`list` or :obj:`tuple`
        Filters to be applied
        (each filter is a dictionary with logname and rule, e.g.
        logname='LFP_COAL', rule='<0.1' will keep all values where values
        in  LFP_COAL logs are <0.1)

    Returns
    -------
    filtered_curves : :obj:`pd.DataFrame`
        Filtered curves
    cond : :obj:`pd.DataFrame`
        Filtering mask
    """
    if isinstance(filters, dict):
        filters = (filters, )
    cond = eval("curves['" + filters[0]['logname'] + "']" +
                filters[0]['rule']).values
    cond = cond | (np.isnan(curves[filters[0]['logname']].values))
    for filter in filters[1:]:
        if filter['chain'] == 'and':
            cond = cond & eval("curves['" + filter['logname'] + "']" +
                               filter['rule']).values
        else:
            cond = cond | eval("curves['" + filter['logname'] + "']" +
                               filter['rule']).values
        cond = cond | (np.isnan(curves[filter['logname']].values))
    filtered_curves = curves[cond]
    return filtered_curves, cond

def _visualize_curve(ax, logs, curve, depth='MD', thresh=None, shift=None,
                     verticalshift=0., scale=1., color='k', lw=2,
                     logscale=False, grid=False, inverty=True, ylabel=True,
                     xlabelpos=0, xlim=None, ylim=None, title=None, **kwargs):
    """Visualize single curve track in axis ``ax``

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`str`
        Keyword of log curve to be visualized
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    thresh : :obj:`float`, optional
        Maximum allowed value (values above will be set to non-valid)
    shift : :obj:`np.ndarray`, optional
        Depth-dependent shift to apply to the curve to visualize
    verticalshift : :obj:`np.ndarray`, optional
        Bulk vertical shift to apply to the curve to visualize
    scale : :obj:`float`, optional
        Scaling to apply to log curve
    color : :obj:`str`, optional
        Curve color
    lw : :obj:`int`, optional
        Line width
    semilog : :obj:`bool`, optional
        Use log scale in log direction
    grid : :obj:`bool`, optional
        Add grid to plot
    inverty : :obj:`bool`, optional
        Invert y-axis
    ylabel : :obj:`str`, optional
        Show y-label
    xlabelpos : :obj:`str`, optional
        Position of xlabel outside of axes (if ``None`` keep it as original)
    xlim : :obj:`tuple`, optional
        x-axis extremes
    ylim : :obj:`tuple`, optional
        y-axis extremes
    title : :obj:`str`, optional
        Title of figure
    kwargs : :obj:`dict`, optional
        Additional plotting keywords

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    try:
        logcurve = _threshold_curve(logs[curve], thresh)
        logcurve *= scale

        if shift is not None:
            logcurve += shift

        plot = True
    except:
        logging.warning('logs object does not contain {}...'.format(curve))
        plot = False

    if plot:
        if grid:
            ax.grid()
        if logscale:
            ax.semilogx(logcurve, logs[depth] + verticalshift, c=color,
                        lw=lw, **kwargs)
        else:
            ax.plot(logcurve, logs[depth] + verticalshift, c=color,
                    lw=lw, **kwargs)
        if ylabel:
            ax.set_ylabel(depth)
        if xlabelpos is not None:
            ax.set_xlabel(title if title is not None else curve, color=color)
            ax.tick_params(direction='in', width=2, colors=color,
                           bottom=False, labelbottom=False, top=True, labeltop=True)
            ax.spines['top'].set_position(('outward', xlabelpos*80))

        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim[1], ylim[0])
        else:
            if inverty:
                ax.invert_yaxis()
    return ax

def _visualize_colorcodedcurve(ax, logs, curve, depth='MD',
                               thresh=None, shift=None, verticalshift=0.,
                               scale=1., envelope=None,
                               leftfill=True, cmap='seismic', clim=None,
                               step=100):
    """Visualize filling of single curve track in colorcode in axis ``ax``.
    Generally used in combination with _visualize_colorcode.

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`str`
        Keyword of log curve to be visualized
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    thresh : :obj:`float`, optional
        Maximum allowed value (values above will be set to non-valid)
    shift : :obj:`np.ndarray`, optional
        Depth-dependent shift to apply to the curve to visualize
    verticalshift : :obj:`np.ndarray`, optional
        Bulk vertical shift to apply to the curve to visualize
    scale : :obj:`float`, optional
        Scaling to apply to log curve
    envelope : :obj:`float`, optional
        Value to use as envelope in color-coded display (if ``None``
        use curve itself)
    leftfill : :obj:`bool`, optional
        Fill on left side of curve (``True``) or right side of curve (``False``)
    cmap : :obj:`str`, optional
        Colormap for colorcoding
    clim : :obj:`tuple`, optional
        Limits of colorbar (if ``None`` use min-max of curve)
    step : :obj:`str`, optional
        Step for colorcoding

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    try:
        logcurve = _threshold_curve(logs[curve], thresh)
        logcurve_color = logcurve.copy()
        logcurve *= scale

        if shift is not None:
            logcurve += shift
        plot = True
    except:
        logging.warning('logs object does not contain {}...'.format(curve))
        plot = False

    if plot:
        # get temporary curves and subsampled them
        x = logcurve[::step]
        y = logs[depth][::step] + verticalshift
        z = logcurve_color[::step]
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        if clim is None:
            normalize = mpl.colors.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
        else:
            normalize = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

        for i in range(x.size - 1):
            if leftfill:
                ax.fill_betweenx([y[i], y[i + 1]], shift,
                                 x2=[x[i], x[i + 1]] if envelope is None \
                                     else [shift + envelope, shift + envelope],
                                 color=cmap(normalize(z[i])))
            else:
                ax.fill_betweenx([y[i], y[i + 1]],
                                 [x[i], x[i + 1]] if envelope is None \
                                     else [np.nanmax(z) - envelope, np.nanmax(z) - envelope],
                                 x2=np.nanmax(z), color=cmap(normalize(z[i])))
        #if envelope is not None:
        #    ax.plot([shift + envelope]*2, [y[i], y[i + 1]], 'k', lw=1)
    return ax

def _visualize_filled(ax, logs, curves, colors, depth='MD', envelope=None,
                      grid=False, inverty=True, ylabel=True, xlim=None,
                      title=None, **kwargs):
    """Visualize filled set of curves

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curves : :obj:`tuple`
        Keywords of N log curve to be visualized
    colors : :obj:`tuple`
        N+1 colors to be used for filling between curves
        (last one used as complement)
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    envelope : :obj:`str`, optional
        keyword of log curve to be used as envelope
    grid : :obj:`bool`, optional
        Add grid to plot
    inverty : :obj:`bool`, optional
        Invert y-axis
    ylabel : :obj:`str`, optional
        Show y-label
    xlim : :obj:`tuple`, optional
        x-axis extremes
    title : :obj:`str`, optional
        Title of figure
    kwargs : :obj:`dict`, optional
        Additional plotting keywords

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    # check that sum of volumes does not exceed 1
    filllogs = np.array([logs[curve] for curve in curves])
    cumfilllogs = np.cumsum(np.array(filllogs), axis=0)
    exceedvol = np.sum(cumfilllogs[-1][~np.isnan(cumfilllogs[-1])]>1.)
    if exceedvol > 0:
        logging.warning('Sum of volumes exceeds '
                        '1 for {} samples'.format(exceedvol))
    if envelope is not None: cumfilllogs = cumfilllogs * logs[envelope]

    # plotting
    if grid:
        ax.grid()
    ax.fill_betweenx(logs[depth], cumfilllogs[0], facecolor=colors[0])
    ax.plot(cumfilllogs[0], logs[depth], 'k', lw=0.5)
    for icurve in range(len(curves)-1):
        ax.fill_betweenx(logs[depth], cumfilllogs[icurve],
                         cumfilllogs[icurve+1],
                         facecolor=colors[icurve+1], **kwargs)
        ax.plot(cumfilllogs[icurve], logs[depth], 'k', lw=0.5)
    if envelope is None:
        ax.fill_betweenx(logs[depth], cumfilllogs[-1], 1, facecolor=colors[-1])
        ax.plot(cumfilllogs[-1], logs[depth], 'k', lw=0.5)
    else:
        ax.fill_betweenx(logs[depth], cumfilllogs[-1], logs[envelope],
                         facecolor=colors[-1])
        ax.plot(cumfilllogs[-1], logs[depth], 'k', lw=0.5)
        ax.plot(logs[envelope], logs[depth], 'k', lw=1.5)
    if ylabel:
        ax.set_ylabel(depth)
    ax.set_title(title if title is not None else '', pad=20)
    ax.tick_params(direction='in', width=2, colors='k',
                   bottom=False, labelbottom=False, top=True, labeltop=True)
    if xlim is not None and len(xlim)==2:
        ax.set_xlim(xlim[0], xlim[1])
    if inverty:
        ax.invert_yaxis()
    return ax


def _visualize_facies(ax, logs, curve, colors, names, depth='MD',
                      cbar=False, title=None):
    """Visualize facies curve as image

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`tuple`
        Keywords oflog curve to be visualized
    colors : :obj:`tuple`
        Colors to be used for facies
    colors : :obj:`tuple`
        Names to be used for facies
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    cbar : :obj:`bool`, optional
        Show colorbar (``True``) or not (``False``)
    title : :obj:`str`, optional
        Title of figure
    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    nfacies = len(colors)
    faciesext, zfaciesest = \
        logs.resample_curve(curve, zaxis=depth)
    faciesext = np.repeat(np.expand_dims(faciesext, 1),
                          nfacies, 1)
    cmap_facies = cplt.ListedColormap(colors,
                                      'indexed')
    im = ax.imshow(faciesext, interpolation='none',
                       aspect='auto', origin='lower',
                       extent=(0, nfacies,
                               zfaciesest[0],
                               zfaciesest[-1]),
                       cmap=cmap_facies, vmin=-0.5,
                       vmax=nfacies - 0.5)
    ax.set_title(title if title is not None else '', pad=20)
    if cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(np.arange(0, nfacies))
        cbar.set_ticklabels(names)
    return ax


class Logs:
    """Log curves object.

    This object contains a set of log curves for a single well from a .LAS file

    Parameters
    ----------
    filename : :obj:`str`
        Name of file containing logs to be read
    wellname : :obj:`str`, optional
        Name of file containing logs to be read
    loadlogs : :obj:`bool`, optional
        Load data into ``self.logs`` variable during initialization (``True``)
        or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,'
    verb : :obj:`str`, optional
        Verbosity

    """
    def __init__(self, filename, wellname=None, loadlogs=True,
                 kind='local', verb=False):
        self.filename = filename
        self.df = None
        self.wellname = filename if wellname is None else wellname
        self._loadlogs = loadlogs
        self._kind = kind
        self._verb = verb
        if self._loadlogs:
            self._logs = self._read_logs()

    @property
    def logs(self):
        if not self._loadlogs:
            self._loadlogs = True
            self._logs = self._read_logs()

        return self._logs

    @property
    def avestep(self):
        return np.median(np.unique(np.diff(self.logs.index)))

    """
    @property
    def df(self):
        self._df = self.logs.df()
        return self._df
    """

    def __str__(self):
        descr = 'Logs {})\n\n'.format(self.wellname) + \
                'Curves: {}\n'.format(list(self.logs.keys()))
        return descr

    def _read_logs(self):
        """Read a set of logs from file
        """
        if self._verb:
            print('Reading {} logs...'.format(self.filename))
        if self._kind == 'local':
            logs = lasio.read(self.filename)
        else:
            raise NotImplementedError('kind must be local')
        # ensure there is no TWT curve leaking in... we only want to get them
        # from TD curves or checkshots so we can keep track of them...
        if 'TWT' in logs.keys():
            logs.delete_curve('TWT')
        return logs

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        logscopy : :obj:`pysubsurface.objects.Logs`
            Copy of Logs object

        """
        logscopy = copy.deepcopy(self)
        return logscopy

    def dataframe(self, resetindex=False):
        """Return log curves into a :obj:`pd.DataFrame`

        Parameters
        ----------
        resetindex : :obj:`bool`, optional
            Move index to curve DEPTH and reset index to consecutive numbers
        """
        self.df = self.logs.df()
        if resetindex:
            depth = self.df.index
            self.df.reset_index(inplace=True)
            self.df['DEPTH'] = depth

    def startsample(self, curve=None):
        "index of first available sample in log"
        if curve is None:
            curve = self.logs.curves[1].mnemonic
        mask = np.cumsum(~np.isnan(self.logs[curve]))
        return np.where(mask == 1)[0][0]

    def endsample(self, curve=None):
        "index of last available sample in log"
        if curve is None:
            curve = self.logs.curves[1].mnemonic
        mask = np.cumsum(np.flipud(~np.isnan(self.logs[curve])))
        return len(mask) - np.where(mask == 1)[0][0]

    def add_curve(self, curve, mnemonic, unit=None, descr=None, value=None,
                  delete=True):
        """Add curve to logset

        Parameters
        ----------
        curve : :obj:`np.ndarray`
            Curve to be added
        mnemonic : :obj:`str`
            Curve mnemonic
        unit : :obj:`str`, optional
            Curve unit
        descr : :obj:`str`, optional
            Curve description
        value : :obj:`int`, optional
            Curve value
        delete : :obj:`bool`, optional
            Delete curve with same name if present (``True``) or not (``False``)
        """
        if delete:
            self.delete_curve(mnemonic)
        self.logs.append_curve(mnemonic, curve,
                               unit='' if unit is None else unit,
                               descr='' if descr is None else descr,
                               value='' if value is None else value)
        self.dataframe()

    def add_tvdss(self, trajectory):
        """Add TVDSS curve (and interpolate from trajectory to logs sampling)

        Parameters
        ----------
        trajectory : :obj:`pysubsurface.objects.Trajectory`
            Curve to be added

        """
        # create regular tvdss axis for mapping of picks
        md = trajectory.df['MD (meters)']
        tvdss = trajectory.df['TVDSS']

        f = interp1d(md, tvdss, kind='linear',
                     bounds_error=False, assume_sorted=True)
        tvdss_log = f(self.logs.index)
        self.logs.append_curve('TVDSS', tvdss_log, unit='m', descr='TVDSS')
        self.dataframe()

    def add_twt(self, tdcurve, name):
        """Add TWT curve (and interpolate from trajectory to logs sampling)

        Parameters
        ----------
        tdcurve : :obj:`pysubsurface.objects.TDcurve`
            TD curve or checkshots
        tdcurve : :obj:`pysubsurface.objects.TDcurve`
            name of TD or checkshot curve to be used within Logs object

        """
        # create regular tvdss axis for mapping of picks
        md = tdcurve.df['Md (meters)']
        twt = tdcurve.df['Time (ms)']

        f = interp1d(md, twt, kind='linear',
                     bounds_error=False, assume_sorted=True)
        twt_log = f(self.logs.index)
        self.logs.append_curve('TWT - {}'.format(name),
                               twt_log, unit='ms', descr='TWT')
        self.dataframe()

    def delete_curve(self, mnemonic, verb=False):
        """Delete curve to logset

        Parameters
        ----------
        mnemonic : :obj:`str`
            Curve mnemonic
        verb : :obj:`bool`, optional
            Verbosity

        """
        if mnemonic in self.logs.keys():
            if verb:
                print('Deleted {} from {} well'.format(mnemonic,
                                                       self.wellname))
            self.logs.delete_curve(mnemonic)
        else:
            if verb: print('Curve {} not present for '
                           '{} well'.format(mnemonic, self.wellname))

    def resample_curve(self, mnemonic, zaxis=None, mask=None, step=None):
        """Return resampled curve with constant step in depth axis.

        Parameters
        ----------
        mnemonic : :obj:`str`
            Curve mnemonic
        zaxis : :obj:`str`, optional
            Label of log to use as z-axis
        mask : :obj:`np.ndarray`, optional
            Mask to apply prior to resampling (values where mask is ``True``
            will be put to np.nan)s
        step : :obj:`float`, optional
            Step. If ``None`` estimated as median value of different steps
            in current depth axis

        Returns
        -------
        loginterp : :obj:`str`
            Interpolated log
        regaxis : :obj:`np.ndarray`
            Regularly sampled depth axis

        """
        if zaxis is None:
            start = self.logs.index[0]
            end = self.logs.index[-1]
        else:
            zaxis_nonan = self.logs[zaxis][~np.isnan(self.logs[zaxis])]
            start = zaxis_nonan[0]
            end = zaxis_nonan[-1]
        if step is None:
            if zaxis is None:
                step = self.avestep
            else:
                steps = np.unique(np.diff(self.logs[zaxis]))
                step = np.max(steps[~np.isnan(steps)])
        regaxis = np.arange(start, end + step, step)

        # Resample the logs to the new axis using linear interpolation
        log = self.logs[mnemonic].copy()
        if mask is not None:
            log[mask] = np.nan
        loginterp = np.interp(regaxis,
                              self.logs.index if zaxis is None else
                              self.logs[zaxis], log)
        return loginterp, regaxis


    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display logs as table

        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        self.dataframe()

        if ipython_flag:
            display(self.df.head(nrows))
        else:
            print(self.df.head(nrows))

    def describe(self):
        """Display statistics of logs
        """
        self.dataframe()

        if ipython_flag:
            display(self.df.describe())
        else:
            print(self.df.describe())

    def visualize_logcurve(self, curve, depth='MD',
                           thresh=None, shift=None, verticalshift=0.,
                           scale=1.,
                           color='k', lw=2,
                           grid=True, xlabelpos=0,
                           inverty=True, curveline=True,
                           colorcode=False, envelope=None,
                           cmap='seismic', clim=None,
                           step=40, leftfill=True,
                           ax=None, figsize=(4, 15), title=None,
                           savefig=None, **kwargs):
        """Visualize log track as function of certain depth curve

        curve : :obj:`str`
            Keyword of log curve to be visualized
        depth : :obj:`str`, optional
            Keyword of log curve to be used for vertical axis
        thresh : :obj:`float`, optional
            Maximum allowed value (values above will be set to non-valid)
        shift : :obj:`np.ndarray`, optional
            Depth-dependent lateral shift to apply to the curve to visualize
        verticalshift : :obj:`np.ndarray`, optional
            Bulk vertical shift to apply to the curve to visualize
        scale : :obj:`float`, optional
            Scaling to apply to log curve
        color : :obj:`str`, optional
            Curve color
        lw : :obj:`int`, optional
            Line width
        grid : :obj:`bool`, optional
            Add grid to plot
        xlabelpos : :obj:`str`, optional
            Position of xlabel outside of axes
        inverty : :obj:`bool`, optional
            Invert y-axis
        curveline : :obj:`bool`, optional
            Display curve as line between curve and max value of curve
        colorcode : :obj:`bool`, optional
            Display curve color-coded between well trajectory and ``envelope``
        envelope : :obj:`float`, optional
            Value to use as envelope in color-coded display (if ``None``
            use curve itself)
        cmap : :obj:`str`, optional
            Colormap for colorcoding
        clim : :obj:`tuple`, optional
            Limits of colorbar (if ``None`` use min-max of curve)
        step : :obj:`str`, optional
            Step for colorcoding
        leftfill : :obj:`bool`, optional
            Fill on left side of curve (``True``) or right side of curve (``False``)
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        if colorcode:
            ax = _visualize_colorcodedcurve(ax, self.logs, curve, depth=depth,
                                            thresh=thresh, shift=shift,
                                            verticalshift=verticalshift,
                                            scale=scale, envelope=envelope,
                                            cmap=cmap, clim=clim, step=step,
                                            leftfill=leftfill)
        if curveline:
            ax = _visualize_curve(ax, self.logs, curve, depth=depth,
                                  thresh=thresh, shift=shift,
                                  verticalshift=verticalshift, scale=scale,
                                  color=color, lw=lw,
                                  xlabelpos=xlabelpos, inverty=inverty,
                                  grid=grid, title=title, **kwargs)

        if savefig is not None:
            fig.savefig(savefig, dpi=300)

        return fig, ax

    def visualize_logcurves(self, curves, depth='MD', ylim=None,
                            grid=True, ylabel=True, seisreverse=False,
                            prestack_wiggles=True,
                            axs=None, figsize=(9, 7),
                            title=None, savefig=None, **kwargs):
        """Visualize multiple logs curves using a common depth axis and
        different layouts (e.g., line curve, filled curve)
        depending on the name given to the curve.

        The parameter ``curves`` needs to be a dictionary of dictionaries
        whose keys can be:

        * 'Volume': volume plot with filled curves from ``xlim[0]`` to
          ``xlim[1]``. The internal dictionary needs to contain the
          following keys, ``logs`` as a list of log curves, ``colors`` as a
          list of colors for filling and ``xlim`` to be the limit of x-axis
        * 'Sat': saturatuion plot with filled curves from ``xlim[0]`` to
          ``xlim[1]`` (or envelope curve). The internal dictionary needs to
          contain the following keys, ``logs`` as a list of log curves,
          ``colors`` as a list of colors for filling, ``envelope`` as a string
          containing the name of the envolope curve and ``xlim`` to be the
          limit of x-axis.
        * 'Stack*': modelled seismic trace. The internal dictionary needs to
          contain the following keys, ``log`` as the AI log used to model
          seismic data, ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * 'Diff*': modelled difference between two seismic traces.
          The internal dictionary needs to contain the following keys,
          ``logs`` as the AI logs used to model seismic data (subtraction
          convention is first - second),
          ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * 'Prestack*': modelled pre-stack seismic gather. The internal
          dictionary needs to contain the following keys, ``vp`` and
          ``vs`` and ``rho`` as VP, VS and density logs used to model
          seismic data, ``theta`` for the angles to be modelled,
          ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * Anything else: treated as single log curve or multiple overlayed
          log curves.  The internal dictionary needs to
          contain the following keys, ``logs`` as a list of log curves,
          ``colors`` as a list of colors for filling, ``lw`` as a list of
          line-widths, ``xlim`` to be the limit of x-axis

        curves : :obj:`dict`
            Dictionary of curve types and names
        depth : :obj:`str`, optional
            Keyword of log curve to be used for vertical axis
        ylim : :obj:`tuple`, optional
            Limits for depth axis
        grid : :obj:`bool`, optional
            Add grid to plots
        ylabel : :obj:`bool`, optional
            Add ylabel to first plot
        seisreverse : :obj:`bool`, optional
            Reverse colors for seismic plots
        prestack_wiggles : :obj:`bool`, optional
            Use wiggles to display pre-stack seismic (``True``) or imshow
            (``False``)
        axs : :obj:`plt.axes`
            Axes handles (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if seisreverse:
            cpos, cneg = 'b', 'r'
        else:
            cpos, cneg = 'r', 'b'

        N = len(curves)
        if axs is None:
            fig, axs = plt.subplots(1, N, sharey=True, figsize=figsize)
        else:
            fig = None

        # Plot the specified curves
        ncurves_max = 1
        for i, key in enumerate(curves.keys()):
            if key is 'Volume':
                axs[i] = _visualize_filled(axs[i], self.logs, curves[key]['logs'],
                                           colors=curves[key]['colors'], xlim=curves[key]['xlim'],
                                           depth=depth, grid=grid, inverty=False, title='Vol',
                                           ylabel=True if i==0 else False)
            elif 'Sat' in key:
                axs[i] = _visualize_filled(axs[i], self.logs, curves[key]['logs'],
                                           envelope = curves[key]['envelope'],
                                           colors=curves[key]['colors'], xlim=curves[key]['xlim'],
                                           depth=depth, grid=grid, inverty=False, title='Sat',
                                           ylabel=True if i==0 else False)
            elif 'Stack' in key:
                trace, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                       ai=curves[key]['log'])[:2]
                axs[i] = _wiggletrace(axs[i], tz, trace, cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Seismic' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=12)

                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Diff' in key:
                # identify common mask where samples from both logs are not nan
                ai1 = self.logs[curves[key]['logs'][0]]
                ai2 = self.logs[curves[key]['logs'][1]]
                mask = (np.isnan(ai1) | np.isnan(ai2))

                trace1, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                       ai=curves[key]['logs'][0], mask=mask)[:2]
                trace2, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in
                                                         curves[key].keys() else
                                       curves[key]['wavcenter'],
                                       ai=curves[key]['logs'][1], mask=mask)[:2]
                dtrace = trace1 - trace2
                axs[i] = _wiggletrace(axs[i], tz, dtrace, cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Seismic difference' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=7)
                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Prestack' in key:
                traces, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'], curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                     vp=curves[key]['vp'], vs=curves[key]['vs'],
                                     rho=curves[key]['rho'],
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     title='Modelled Pre-stack Seismic',
                                     plotflag=False)[0:2]
                if prestack_wiggles:
                    axs[i] = _wiggletracecomb(axs[i], tz, curves[key]['theta'],
                                              traces, scaling=curves[key]['scaling'],
                                              cpos=cpos, cneg=cneg)
                else:
                    axs[i].imshow(traces.T, vmin=-np.abs(traces).max(),
                                  vmax=np.abs(traces).max(),
                                  extent=(curves[key]['theta'][0],
                                          curves[key]['theta'][-1],
                                          tz[-1], tz[0]), cmap='seismic')
                    axs[i].axis('tight')
                axs[i].set_title('Modelled Pre-stack Seismic' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=7)
                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Prediff' in key:
                # identify common mask where samples from both logs are not nan
                vp1 = self.logs[curves[key]['vp'][0]]
                vp2 = self.logs[curves[key]['vp'][1]]
                vs1 = self.logs[curves[key]['vs'][0]]
                vs2 = self.logs[curves[key]['vs'][1]]
                rho1 = self.logs[curves[key]['rho'][0]]
                rho2 = self.logs[curves[key]['rho'][1]]
                mask = (np.isnan(vp1) | np.isnan(vp2) |
                        np.isnan(vs1) | np.isnan(vs2) |
                        np.isnan(rho1) | np.isnan(rho2))

                traces1, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'], curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                     vp=curves[key]['vp'][0], vs=curves[key]['vs'][0],
                                     rho=curves[key]['rho'][0], mask=mask,
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     plotflag=False)[0:2]
                traces2, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'],
                                     curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else
                                     curves[key]['wavcenter'],
                                     vp=curves[key]['vp'][1], vs=curves[key]['vs'][1],
                                     rho=curves[key]['rho'][1], mask=mask,
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     plotflag=False)[0:2]
                axs[i] = _wiggletracecomb(axs[i], tz, curves[key]['theta'],
                                          traces1 - traces2,
                                          scaling=curves[key]['scaling'],
                                          cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Pre-stack Seismic difference'
                                 if 'title' not in curves[key].keys() else curves[key]['title'],
                                 fontsize = 7)
                if 'xlim' in curves[key].keys():
                    axs[i].set_xlim(curves[key]['xlim'])

            elif 'Facies' in key:
                axs[i] = _visualize_facies(axs[i], self,
                                           curves[key]['log'],
                                           curves[key]['colors'],
                                           curves[key]['names'],
                                           depth=depth,
                                           cbar=False if 'cbar' not in \
                                                curves[key].keys() \
                                                else curves[key]['cbar'],
                                           title=key)

            else:
                ncurves = len(curves[key]['logs'])
                ncurves_max = ncurves if ncurves > ncurves_max else ncurves_max
                for icurve, (curve, color) in enumerate(zip(curves[key]['logs'],
                                                            curves[key]['colors'])):
                    if 'lw' not in curves[key].keys(): curves[key]['lw'] = [int(2)] * ncurves
                    if icurve == 0:
                        axs[i].tick_params(which='both', width=0, bottom=False,
                                           labelbottom=False, top=False, labeltop=False)

                    axs_tmp = axs[i].twiny()
                    axs_tmp = self.visualize_logcurve(curve, depth=depth, thresh=None,
                                                      color=color, lw=curves[key]['lw'][icurve],
                                                      xlim=curves[key]['xlim'],
                                                      logscale = False if 'logscale' not in curves[key] else curves[key]['logscale'],
                                                      grid=grid,
                                                      inverty=False, ylabel=True if i==0 else False,
                                                      xlabelpos=icurve/ncurves,
                                                      ax=axs_tmp, title=curve, **kwargs)
                axs[i].set_xlim(curves[key]['xlim'])
        axs[0].invert_yaxis()
        if ylabel:
            axs[0].set_ylabel(depth)
        if ylim is not None:
            axs[0].set_ylim(ylim[1], ylim[0])
        if fig is not None:
            fig.suptitle(self.filename if title is None else title, y=0.93+ncurves_max*0.02,
                         fontsize=20, fontweight='bold')

        if savefig is not None and fig is not None:
            fig.savefig(savefig, dpi=300)

        return fig, axs

    def visualize_histogram(self, curve, thresh=None, thresh1=None, bins=None, color='k',
                            grid=True, ax=None, figsize=(9, 7), title=None,
                            savefig=None):
        """Visualize histogram of log curve

        Parameters
        ----------
        curve : :obj:`str`
            Keyword of log curve to be visualized
        thresh : :obj:`float`, optional
            Maximum allowed value (values above will be set to non-valid)
        thresh1 : :obj:`float`, optional
            Minimum allowed value (values above will be set to non-valid)
        color : :obj:`str`, optional
            Curve color
        grid : :obj:`bool`, optional
            Add grid to plot
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
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
        ax : :obj:`plt.axes`
            Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        try:
            logcurve = _threshold_curve(self.logs[curve], thresh)
            logcurve = _threshold_curve(logcurve, thresh1, greater=False)
        except:
            raise ValueError('{} does not contain {}...'.format(self.filename,
                                                                curve))
        # remove nans
        logcurve = logcurve[~np.isnan(logcurve)]

        # plot samples
        if grid:
            ax.grid()
        sns.distplot(logcurve, fit=norm, rug=False, bins=bins,
                     hist_kws={'color': color, 'alpha': 0.5},
                     kde_kws={'color':color, 'lw': 3},
                     fit_kws={'color': color, 'lw': 3, 'ls':'--'},
                     ax=ax)
        ax.set_xlabel(curve)
        if title is not None: ax.set_title(title)
        if bins is not None: ax.set_xlim(bins[0], bins[-1])
        ax.text(0.95 * ax.get_xlim()[1], 0.85 * ax.get_ylim()[1],
                'mean: {0:%.3f},\nstd: {1:%.3f}' % (np.mean(logcurve),
                                                    np.std(logcurve)),
                fontsize=14,
                ha="right", va="center",
                bbox=dict(boxstyle="square",
                          ec=(0., 0., 0.),
                          fc=(1., 1., 1.)))

        if savefig is not None:
            plt.tight_layout()
            fig.savefig(savefig, dpi=300)
        return fig, ax

    def visualize_crossplot(self, curve1, curve2, curvecolor=None,
                            thresh1=None, thresh2=None, threshcolor=None,
                            cmap='jet', cbar=True, cbarlabels=None,
                            grid=True, ax=None, figsize=(9, 7),
                            title = None, savefig = None, **kwargs):
        """Crossplot two log curves (possibly color-coded using another curve)

        curve1 : :obj:`str`
            Keyword of log curve to be visualized along x-axis
        curve2 : :obj:`str`
            Keyword of log curve to be visualized along y-axis
        curvecolor : :obj:`str`
            Keyword of log curve to be color-coded
        thresh1 : :obj:`float`, optional
            Maximum allowed value for curve1
            (values above will be set to non-valid)
        thresh2 : :obj:`float`, optional
            Maximum allowed value for curve2
            (values above will be set to non-valid)
        threshcolor : :obj:`float`, optional
            Maximum allowed value for curvecolor
            (values above will be set to non-valid)
        cmap : :obj:`str` or :obj:`list`, optional
            Colormap name or list of colors for discrete map
        cbar : :obj:`bool`, optional
            Add colorbar
        cbarlabels : :obj:`list` or :obj:`tuple`, optional
            Labels to be added to colorbar. To be used for discrete colorbars
        grid : :obj:`bool`, optional
            Add grid to plot
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle
        scatax : :obj:`matplotlib.collections.PathCollection`
            Scatterplot handle
        cbar : :obj:`matplotlib.colorbar.Colorbar`
            Colorbar handle
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        try:
            logcurve1 = _threshold_curve(self.logs[curve1], thresh1)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curve1))
        try:
            logcurve2 = _threshold_curve(self.logs[curve2], thresh2)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curve2))
        try:
            if len(curvecolor):
                logcurvecolor = _threshold_curve(self.logs[curvecolor], threshcolor)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curvecolor))

        if cbarlabels:
            if isinstance(cmap, str):
                cmap = _discrete_cmap(len(cbarlabels), cmap)
            else:
                cmap = _discrete_cmap_indexed(cmap)

        if grid:
            ax.grid()
        if curvecolor is not None:
            scatax = ax.scatter(logcurve1, logcurve2, c=logcurvecolor,
                                marker='o', edgecolors='none', alpha=0.7,
                                vmin=np.nanmin(logcurvecolor),
                                vmax=np.nanmax(logcurvecolor),
                                cmap=cmap)
        else:
            scatax = ax.scatter(logcurve1, logcurve2, marker='o',
                                edgecolors='none', alpha=0.7,
                                cmap=cmap, **kwargs)
        ax.set_xlabel(curve1), ax.set_ylabel(curve2)
        ax.set_title(title if title is None else '{} - {}'.format(curve1,
                                                                  curve2),
                     weight='bold')

        # add colorbar
        if cbar:
            cbar = plt.colorbar(scatax, ax=ax)
            if curvecolor is not None:
                cbar.ax.set_ylabel(curvecolor, rotation=270)
            if cbarlabels:
                scatax.set_clim(vmin=np.nanmin(logcurvecolor) - 0.5,
                                vmax=np.nanmax(logcurvecolor) + 0.5)
                cbar.set_ticks(np.arange(np.nanmin(logcurvecolor),
                                         np.nanmax(logcurvecolor)+1))
                cbar.set_ticklabels(cbarlabels)
        else:
            cbar = None

        if savefig is not None:
            fig.savefig(savefig, dpi=300)
        return fig, ax, scatax, cbar

