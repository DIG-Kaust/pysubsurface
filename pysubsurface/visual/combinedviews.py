import logging
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns

from pysubsurface.utils.utils import findclosest
from pysubsurface.objects.utils import _findclosest_well_seismicsections
from pysubsurface.objects.seismic import _extract_arbitrary_path
from pysubsurface.proc.seismicmod.avo import _methods
from pysubsurface.proc.seismicmod.avo import *


def categorical_statistics(df, colors, name='Values', linecol='w', alpha=1.,
                           xlim=None, sharey=False, title=None, figsize=(10, 7)):
    """Display statistical distribution of each row of a dataframe
    (assuming that each row represents a category and columns represent
    different data points in this category)

    Parameters
    ----------

    Returns
    -------

    """
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    df_plot = []
    for ipick, (category, values) in enumerate(df.iterrows()):
        values_valid = values.dropna()
        df_plot.append(pd.DataFrame({name:values_valid, 'category':category}))
    df_plot = pd.concat(df_plot).reset_index(drop=True)

    fig = sns.FacetGrid(df_plot, row="category", hue="category", aspect=7,
                        height=1.4, palette=colors, sharey=sharey)
    fig.map(sns.kdeplot, name, clip_on=True, shade=True, alpha=alpha, lw=1.5)
    fig.map(sns.kdeplot, name, clip_on=True, color=linecol, lw=4)
    fig.map(plt.axhline, y=0, lw=2, clip_on=True)
    fig.map(label, name)

    # Set the subplots to overlap
    fig.fig.subplots_adjust(hspace=-0.05)

    # Define xlims
    if xlim is not None:
        fig.axes[0][-1].set_xlim(xlim)

    # Remove axes details that don't play well with overlap
    fig.set_titles("")
    fig.set(yticks=[])
    fig.despine(bottom=True, left=True)
    fig.fig.suptitle(title)
    fig.fig.set_size_inches(figsize)
    return fig


def scatter_well(x, y, wellnames,
                 xlabel=None, ylabel=None, cmap='Spectral',
                 regplot=True, diagplot=False, ax=None,
                 figsize=(9, 7), title=None, savefig=None):
    """Display scatter plot and create legend based on well names

    Parameters
    ----------
    x : :obj:`list`
        Values to display on x-axis
    y : :obj:`list` objects
        Values to display on y-axis
    wellnames : :obj:`list`
        Interval level to display (used in case the same ``interval`` is
        present in different levels)
    xlabel : :obj:`str`, optional
        X label
    ylabel : :obj:`str`, optional
        Y label
    cmap : :obj:`str`, optional
         Colormap
    regplot : :obj:`bool`, optional
         Display regression line
    diagplot : :obj:`bool`, optional
         Display main diagonal
    ax : :obj:`plt.axes`
       Axes handle (if ``None`` create a new figure)
    figsize : :obj:`tuple`, optional
        Size of figure
    title : :obj:`str`, optional
        Title of figure
    savefig : :obj:`str`, optional
        Figure filename (if ``None``, figure is not saved)

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
    cmap = cm.get_cmap(cmap, len(x))

    # build the plot
    for iwell, wellname in enumerate(wellnames):
        if not np.isnan(x[iwell]) and not np.isnan(y[iwell]):
            ax.scatter(x[iwell], y[iwell], c=[cmap(iwell)],
                       edgecolor='k', linewidth=1, s=100, label=wellname)
    # add regression line
    if regplot:
        try: # done to avoid case where and x and y are constant and regression is undefined
            sns.regplot(x, y, scatter=False, color=".1", ax=ax)
        except:
            logging.warning('Cannot compute regression line')
    # add main diagonal
    if diagplot:
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        ax.plot(xlims, xlims, '--k', lw=2)
        ax.set_xlim(xlims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
              fancybox=True, shadow=True, ncol=max(1, int(len(x)/20)))
    """
    if regplot:
        ax.text(0.77, 0.94,
                'pearson={0:.2f}'.format(pearsonr(x[~np.isnan(x) & ~np.isnan(y)],
                                              y[~np.isnan(x) & ~np.isnan(y)])[0]),
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc=(1., 1., 1.), ec='k'))
    """
    plt.subplots_adjust(bottom=0.15)

    # savefig
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, ax


def intervals_on_map(wells, surface, level, interval, tops=None,
                     property='Thickness (meters)', addthickness=False,
                     xlim=None, ylim=None,
                     checkwell=False, shift=(0, 0),
                     bubblesize=500, climbubbles=None, cmapproperty='jet',
                     ax=None, savefig=None, **kwargs_surface):
    """Display properties contained in in a specified interval
    for all `wells` as bubble on top of a surface

    Parameters
    ----------
    wells : :obj:`dict`
        Suite of :obj:`pysubsurface.objects.Well` objects
    surface : :obj:`pysubsurface.objects.Surface` objects
        Surface to display
    level : :obj:`int`
        Interval level to display (used in case the same ``interval`` is
        present in different levels)
    interval : :obj:`str`
        Name of interval to display
    tops : :obj:`list`
        Top and base picks for custom interval (i.e., not available in stratigraphic column)
    property : :obj:`tuple`, optional
        Property to visualize (must be present in interval dataframe)
    addthickness : :obj:`bool`, optional
            Find properties in the middle of an interval by adding thickness
            (``True``) or  not (``False``)
    xlim : :obj:`tuple`, optional
        Plotting limits in x axis
    ylim : :obj:`tuple`, optional
        Plotting limits in y axis
    checkwell : :obj:`bool`, optional
        Check if well is inside axis limits (``True``) or plot
        anyways (``False``)
    shift : :obj:`tuple`, optional
        Shift to be applied to well label
    bubblesize : :obj:`int`, optional
        Size of bubbles
    climbubbles : :obj:`float`, optional
        Colorbar limits for bubbles (if ``None`` infer from data)
    cmapinterval : :obj:`tuple`, optional
        Colormap for property
    ax : :obj:`plt.axes`
        Axes handle (if provided the axes should already contain a map and
        this function only plots intervals)
    savefig : :obj:`str`, optional
        Figure filename (if ``None``, figure is not saved)
    kwargs_surface : :obj:`dict`, optional
        Additional parameters for surface plot

    Returns
    -------
    fig : :obj:`plt.figure`
       Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
       Axes handle

    """
    wellnames = list(wells.keys())

    # check if surface will have flipped axes
    try:
        flipaxis = kwargs_surface['flipaxis']
    except:
        flipaxis = False

    if ax is None:
        fig, ax = \
            surface.view(**kwargs_surface)
    else:
        fig = None

    # fix axes limits
    if xlim is not None:
        ax.set_xlim(sorted(xlim))
    if ylim is not None:
        ax.set_ylim(sorted(ylim))

    # extract formation property and its location in x-y coordinates
    xcoords = np.full(len(wellnames), np.nan)
    ycoords = np.full(len(wellnames), np.nan)
    props = np.full(len(wellnames), np.nan)
    for iwell, wellname in enumerate(wellnames):
        if wells[wellname].intervals is not None:
            xcoord, ycoord, prop = \
                wells[wellname].extrac_prop_in_interval(interval, level,
                                                        property, tops=tops,
                                                        addthickness=addthickness)
            if prop is not None:
                props[iwell] = prop
                xcoords[iwell] = xcoord
                ycoords[iwell] = ycoord

    if climbubbles is None:
        climbubbles = [np.nanmin(props), np.nanmax(props)]

    im = ax.scatter(ycoords if flipaxis else xcoords,
                    xcoords if flipaxis else ycoords, c=props, s=bubblesize,
                    cmap=cmapproperty,
                    vmin=climbubbles[0], vmax=climbubbles[1],
                    edgecolors='k', linewidths=2)
    if fig is not None:
        axcbar = fig.add_axes([0.71, 0.41, 0.2, 0.5])
        axcbar.axis('off')
        cbar = fig.colorbar(im, ax=axcbar, orientation='horizontal', shrink=0.6)
        cbar.ax.set_title('%s %s' % (property, interval))

    # add wells
    for iwell, wellname in enumerate(wellnames):
        if wells[wellname].intervals is not None and not np.isnan(xcoords[iwell]):
            wells[wellname].trajectory.view_traj(labelcoords=(xcoords[iwell],
                                                              ycoords[iwell]),
                                                 flipaxis=flipaxis,
                                                 shift=shift, ax=ax,
                                                 checkwell=checkwell,
                                                 color='k', bbox=False,
                                                 labels=True)

    # savefig
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
    return fig, ax


def seismic_and_map(seismic, surface, wells=None, whichseismic='il',
                    whichsurface='ilxl', ilplot=None, xlplot=None,
                    ilvertices=None, xlvertices=None,
                    kargs_seismicplot={}, kargs_surfaceplot={},
                    figsize=(20, 7), title=None, savefig=None):
    """Display seismic section and surface with line used to create the seismic
    section.

    Parameters
    ----------
    seismic : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Seismic to display
    surface : :obj:`pysubsurface.objects.Surface`
        Surface to display
    wells : :obj:`dict`, optional
        Wells to display in surface map
    whichseismic : :obj:`str`, optional
        ``IL``: display inline section passing through well,
        ``XL``: display crossline section passing through well.
        ``arbitrary``: display crossline section passing an arbitrary path.
    whichsurface : :obj:`str`, optional
        Visualize surface in IL-XL coordinates (``ilxl``) or
        ``yx`` y-x coordinates
    ilplot : :obj:`int`, optional
        Index of inline to plot if ``whichseismic='IL'``
        (if ``None`` show inline in the middle)
    xlplot : :obj:`int`, optional
        Index of crossline to plot if ``whichseismic='XL'``
        (if ``None`` show crossline in the middle)
    ilvertices : :obj:`tuple` or :obj:`list`
        Vertices of arbitrary path in inline direction
        if ``whichseismic='arbitrary'``. Currently, ``ilvertices`` must be in
        increasing order or will be internally reorganized.
    xlvertices : :obj:`plt.axes`, optional
        Vertices of arbitrary path in crossline direction
        if ``whichseismic='arbitrary'``. ``xlvertices`` can be in any order
    kargs_seismicplot : :obj:`dict`, optional
        additional input parameters to be provided to
        :func:`pysubsurface.objects.Seismic.view` or
        :func:`pysubsurface.objects.Seismic.view_arbitraryline`
    kargs_surfaceplot : :obj:`dict`, optional
        additional input parameters to be provided to
        :func:`pysubsurface.objects.Surface.view`
    figsize : :obj:`tuple`, optional
         Size of figure
    title : :obj:`str`, optional
         Title of figure
    savefig : :obj:`str`, optional
         Figure filename (if ``None``, figure is not saved)

    Returns
    -------
    fig : :obj:`plt.figure`
       Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
       Axes handles

    """
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title if title is not None else '',
                 fontsize=14, fontweight='bold', y=0.99)

    # plot seismic
    if whichseismic == 'arbitrary':
        # sort ilines in increasing order
        ilxlsort = np.array(ilvertices).argsort()
        ilvertices = np.array(ilvertices)[ilxlsort]
        xlvertices = np.array(xlvertices)[ilxlsort]
        _, ax1 = seismic.view_arbitraryline(ilvertices=ilvertices,
                                            xlvertices=xlvertices,
                                            ax=ax1, fig=fig,
                                            **kargs_seismicplot)
    else:
        _, ax1 = seismic.view(ilplot=ilplot, xlplot=xlplot,
                              which=whichseismic, axs=ax1,
                              **kargs_seismicplot)

    # plot surface
    ax1pos = ax1.get_position()

    ax2 = fig.add_axes([ax1pos.x0 + 0.01, ax1pos.y0 + 0.01,
                        0.15, 0.25])
    surface.view(ax=ax2, which=whichsurface, **kargs_surfaceplot)

    # plot line on surface
    if whichsurface == 'yx':
        cdpx = seismic.cdpx.reshape(seismic.nil, seismic.nxl)
        cdpy = seismic.cdpy.reshape(seismic.nil, seismic.nxl)

        if whichseismic == 'arbitrary':
            cdpxvertices = np.zeros(len(ilvertices))
            cdpyvertices = np.zeros(len(ilvertices))

            for i, (ilvertex, xlvertex) in enumerate(
                    zip(ilvertices, xlvertices)):
                iil = findclosest(ilvertex, seismic.ilines)
                ixl = findclosest(xlvertex, seismic.xlines)
                cdpxvertices[i] = cdpx[iil, ixl]
                cdpyvertices[i] = cdpy[iil, ixl]
            ax2.plot(np.array(cdpxvertices), np.array(cdpyvertices),
                     color='k', linestyle='--', lw=2)
        elif whichseismic == 'il':
            if ilplot is None:
                ilplot = len(seismic.ilines)//2
            iil = findclosest(seismic.ilines[ilplot], seismic.ilines)
            ixl_start = 0
            ixl_end = seismic.nxl-1
            ax2.plot([cdpx[iil, ixl_start]] * 2,
                     [cdpy[iil, ixl_start], cdpy[iil, ixl_end]], \
                     color='k', linestyle='--', lw=2)
        elif whichseismic == 'xl':
            if xlplot is None:
                xlplot = len(seismic.xlines) // 2
            ixl = findclosest(seismic.xlines[xlplot], seismic.xlines)
            iil_start = 0
            iil_end = seismic.nil-1
            ax2.plot([cdpx[iil_start, ixl], cdpx[iil_end, ixl]],
                     [cdpy[iil_start, ixl]] * 2, \
                     color='k', linestyle='--', lw=2)

    elif whichsurface == 'ilxl':
        if whichseismic == 'arbitrary':
            ax2.plot(np.array(ilvertices), np.array(xlvertices),
                     color='k', linestyle='--', lw=2)
        elif whichseismic == 'il':
            if ilplot is None:
                ilplot = len(seismic.ilines) // 2
            ax2.plot([seismic.ilines[ilplot]]*2,
                     [seismic.xlines[0], seismic.xlines[-1]], \
                     color='k', linestyle='--', lw=2)
        elif whichseismic == 'xl':
            if xlplot is None:
                xlplot = len(seismic.xlines) // 2
            ax2.plot([seismic.ilines[0], seismic.ilines[-1]],
                     [seismic.xlines[xlplot]] * 2, \
                     color='k', linestyle='--', lw=2)

    # add wells (need to convert arbitraty path in x and y and show surface with x and y)
    if wells is not None and whichsurface == 'yx':
        wellnames = list(wells.keys())
        for wellname in wellnames:
            wells[wellname].trajectory.view_traj(shift=(1e3, 1e3), ax=ax2,
                                                 color='k', labels=False,
                                                 wellname=False)

    ax2.axis('off')

    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def seismic_through_wells(seismic, wells,  domain='depth',
                          ilinein=None, xlinein=None,
                          ilineend=None, xlineend=None,
                          surface=None, cmapsurface='gist_rainbow',
                          surfacescale=1., welllabel=True,
                          bbox=False, level=2, twtcurve=None,
                          logcurve=None, logcurvethresh=None,
                          logcurvescale=1., logcurvecmap=None,
                          logcurveclim=None, logcurveenvelope=None,
                          logcurveshifts=None,
                          seismicalpha=None, seismicalpha_args={},
                          verb=False, **kwargs_seismic):
    """Display seismic section passing through a list of vertical wells.

    Parameters
    ----------
    seismic : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Seismic to display
    wells : :obj:`dict`
        Wells to display. The fence will be created using the ordering in
        the input dictionary.
    domain : :obj:`str`, optional
            Domain of seismic data, ``depth`` or ``time``
    ilinein : :obj:`int`, optional
        Inline to use for first vertex of the path
        (if ``None`` use first inline in seismic data
    xlinein : :obj:`int`, optional
        Crossline to use for first vertex of the path
        (if ``None`` use first crossline in seismic data
    ilineend : :obj:`int`, optional
        Inline to use for last vertex of the path
        (if ``None`` use first inline in seismic data
    xlineend : :obj:`int`, optional
        Crossline to use for last vertex of the path
        (if ``None`` use first crossline in seismic data
    surface : :obj:`pysubsurface.objects.Surface`, optional
        Surface to visualize with line visualized in seismic section
    cmapsurface : :obj:`str`, optional
        Colormap of surface
    surfacescale :obj:`float`, optional
        Scale to apply to vertical axis of surface
    welllabel : :obj:`bool`, optional
        Add well name labels
    bbox : :obj:`bool`, optional
        Add box around well name labels
    level : :obj:`int` or :obj:`list`
            Interval level(s) to use for display of picks
    twtcurve : :obj:`str` or :obj:`list`, optional
        Name of TWT curve(s) as defined in ``self.tdcurve`` or
        ``self.checkshots`` to be used to plot well trajectory when
        seismic is in time domain
    logcurve : :obj:`str`, optional
        Name of log curve to display
    logcurvethresh : :obj:`float`, optional
        Maximum allowed value for log curve (values above set to non-valid)
    logcurvescale : :obj:`float`, optional
        Scaling to apply to log curve
    logcurvecmap : :obj:`float`, optional
        Cmap to use to fill log curve (if ``None`` do not fill)
    logcurveclim : :obj:`tuple`, optional
        Limits of colorbar (if ``None`` use min-max of curve)
    logcurveenvelope : :obj:`float`, optional
        Value to use as envelope when filling curve (if ``None`` use curve itself)
    logcurveshifts : :obj:`tuple`, optional
        Vertical shifts to apply to log curves for each well
        (if ``None`` no shift is applied)
    seismicalpha : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Seismic to display in transparency
    seismicalpha_args : :obj:`dict`, optional
        additional input parameters to be provided to
        :func:`pysubsurface.objects.Seismic.view_arbitraryline` for ``seismicalpha``
    verb: :obj:`bool`, optional
        Verbosity.
    kwargs_seismic : :obj:`dict`, optional
        additional input parameters to be provided to
        :func:`pysubsurface.objects.Seismic.view_arbitraryline` for ``seismicalpha``

    Returns
    -------
    fig : :obj:`plt.figure`
       Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
       Axes handles

    """
    # ensure one twtcurve per well
    if twtcurve is None:
        twtcurve = [None] * len(wells.keys())
    if isinstance(twtcurve, str):
        twtcurve = [twtcurve, ]

    # find out vertices
    ilvertices = [seismic.ilines[0] if ilinein is None else ilinein]
    xlvertices = [seismic.xlines[0] if xlinein is None else xlinein]
    chosenwells = []
    for wellname in wells.keys():
        ilwell, xlwell = \
            _findclosest_well_seismicsections(wells[wellname],
                                              seismic, traj=False)
        # find out if well is inside
        if ilwell == seismic.ilines[0] or ilwell == seismic.ilines[-1] or \
                xlwell == seismic.xlines[0] or xlwell == seismic.xlines[-1]:
            logging.warning('Well {} may be outside of seismic area as '
                            'closest inline (or crossline) is at the edge '
                            'of seismic inline (or crossline) '
                            'axis'.format(wellname))
        else:
            ilvertices.append(ilwell)
            xlvertices.append(xlwell)
            chosenwells.append(wellname)
    ilvertices.append(seismic.ilines[-1] if ilineend is None else ilineend)
    xlvertices.append(seismic.xlines[-1] if xlineend is None else xlineend)
    if verb:
        print('ILvertex={}, XLvertex={}...'.format(ilvertices, xlvertices))

    ils, xls, iils, ixls, nedges = \
        _extract_arbitrary_path(ilvertices, xlvertices,
                                seismic.dil, seismic.dxl,
                                seismic.ilines[0], seismic.xlines[0])

    # plot seismic
    fig, ax = seismic.view_arbitraryline(ils, xls, addlines=False,
                                         usevertices=True, jumplabel=100,
                                         **kwargs_seismic)
    # plot seismic in transparency
    if seismicalpha is not None:
        fig, ax = seismicalpha.view_arbitraryline(ils, xls, addlines=False,
                                                  usevertices=True, jumplabel=100,
                                                  ax=ax, fig=fig,
                                                  **seismicalpha_args)
    # add wells, picks, contacts and log curves
    if logcurveshifts is None:
        logcurveshifts = [0.] * len(chosenwells)
    for wellname, edge, logcurveshift, twtc in zip(chosenwells, nedges, logcurveshifts, twtcurve):
        ax.plot([edge]*2,
                [wells[wellname].trajectory.df.iloc[0]['TVDSS'] + logcurveshift,
                 wells[wellname].trajectory.df.iloc[-1]['TVDSS'] + logcurveshift], 'k', lw=2)
        if welllabel:
            ax.text(edge, wells[wellname].trajectory.df.iloc[-1]['TVDSS'] + logcurveshift,
                    wellname, ha="center", va="center", color='k',
                    bbox=None if bbox is False else dict(boxstyle="round",
                                                         fc=(1., 1., 1.),
                                                         ec='k', alpha=0.9))
        if domain == 'depth':
            topdepth = 'Top TVDSS (meters)'
            contactdepth = 'TVDSS (meters)'
        else:
            topdepth = 'Top TWT - ' + twtc + ' (ms)'
            contactdepth = 'TWT - ' + twtc + ' (ms)' # 'TWT (meters)'

        if not isinstance(level, (list, tuple)):
            level = [level]
        intervals_plot = \
            wells[wellname].intervals[wells[wellname].intervals['Level'].isin(level)]

        for ipick, pick in intervals_plot.iterrows():
            ax.plot(edge, float(pick[topdepth]) + logcurveshift, marker='_',
                    color=pick['Color'], ms=15, mew=7)

        for icontact, contact in wells[wellname].contacts.df.iterrows():
            ax.plot(edge, float(contact[contactdepth]) + logcurveshift, marker='_',
                    color=contact['Color'], ms=15, mew=7)

        if logcurve is not None:
            wells[wellname].welllogs.visualize_logcurve(curve=logcurve,
                                                        depth='TVDSS' if domain == 'depth' else 'TWT - ' + twtc,
                                                        thresh=logcurvethresh,
                                                        shift=edge,
                                                        verticalshift=logcurveshift,
                                                        scale=logcurvescale,
                                                        color='k',
                                                        colorcode=False if logcurvecmap is None else True,
                                                        cmap=logcurvecmap,
                                                        clim=logcurveclim,
                                                        envelope = logcurveenvelope if logcurvecmap is not None else None,
                                                        curveline = False if logcurveenvelope is not None else True,
                                                        leftfill=True,
                                                        grid=False,
                                                        title=None,
                                                        xlabelpos=None,
                                                        inverty=False,
                                                        ax=ax)
    if domain != 'depth':
        ax.set_ylabel('TWT')

    # add surface and line along which seismic is displayed
    axsurface = None
    if surface is not None:
        axpos = ax.get_position()
        axsurface = fig.add_axes([axpos.x0 + 0.01, axpos.y0 + 0.01,
                                  0.15, 0.15 * surfacescale])

        _, axsurface = surface.view(ax=axsurface, cmap=cmapsurface,
                                    originlower=True, ncountour=3,
                                    lwcountour=0.5)
        axsurface.plot(xlvertices, ilvertices, color='w', lw=2)
        axsurface.scatter(xlvertices[1:-1], ilvertices[1:-1], s=40, c='w')
        axsurface.axis('off')
    return fig, (ax, axsurface)


def correlation_panel(wells, curves, ylim, depth='TVDSS', level=None,
                      wellabelsshift=100, figsize=None, savefig=None):
    """Display well correlation panel.

    Display well logs from multiple wells sharing the same vertical axis.

    Parameters
    ----------
    wells : :obj:`dict`
        Wells to display. The fence will be created using the ordering in
        the input dictionary.
    curves : :obj:`dict`
        Dictionary of curve types and names
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    wellabelsshift : :obj:`float`, optional
        Shift in depth coordinate to use for welllabels
    figsize : :obj:`tuple`, optional
             Size of figure
    savefig : :obj:`str`, optional
         Figure filename (if ``None``, figure is not saved)

    Returns
    -------
    fig : :obj:`plt.figure`
       Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
       Axes handles

    """
    nwells = len(wells.keys())
    ncurves = len(curves.keys())
    nlogsmax = [len(curves[curve]['logs']) for curve in curves.keys() if curve not in ['Volume', 'Sat']]
    nlogsmax = 1 if len(nlogsmax) == 0 else max(nlogsmax)
    gap = 0.01

    if figsize is None:
        figsize = (ncurves * nwells * 2, 10)
    fig = plt.figure(figsize=figsize)
    #fig.suptitle(title, y=1+0.05+0.04*(nlogsmax-1), fontsize=20, fontweight='bold')
    gs = [fig.add_gridspec(nrows=1, ncols=ncurves, wspace=0., left=iwell/nwells+gap,
                           right=(iwell+1)/nwells-gap) for iwell in range(nwells)]
    axs = [fig.add_subplot(gs[iwell][:, ilog]) for iwell in range(nwells) for ilog in range(ncurves)]
    for iwell, wellname in enumerate(wells.keys()):
        wells[wellname].welllogs.visualize_logcurves(curves, depth=depth, ylim=ylim,
                                                     ylabel=True if iwell == 0 else False,
                                                     axs=axs[iwell*ncurves:iwell*ncurves+ncurves])
        for icurve in range(ncurves):
            axs[iwell*ncurves+icurve].set_ylim(ylim[1], ylim[0])
            if iwell > 0 or icurve > 0:
                axs[iwell*ncurves+icurve].set_yticklabels([])
        if level is not None:
            wells[wellname].view_picks_and_intervals(axs=axs[iwell*ncurves:iwell*ncurves+ncurves],
                                                     depth=depth,
                                                     ylim=ylim, level=level,
                                                     labels=False if iwell*ncurves+icurve < ncurves * nwells -1 else True,
                                                     pad=1.05)
        axs[iwell * ncurves].text(axs[iwell * ncurves].get_xlim()[0],
                                  ylim[0] - wellabelsshift,#0.3*(1+1.*(nlogsmax-1))*(10**int(np.log10(ylim[1]-ylim[0]))),
                                  wellname, fontsize=11, fontweight='bold',
                                  ha='left', va="center", color='w',
                                  bbox=dict(boxstyle="square",
                                            ec=(0., 0., 0.),
                                            fc=(0., 0., 0.)))

    if fig is not None:
        plt.subplots_adjust(bottom=0.15)
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
    return fig, axs


def ava_modelling(vp1, vs1, rho1, vp2, vs2, rho2, theta,
                  methods,  colorsrefl, lwrefl=3,
                  colorpar='k', lwpar=3, axs=None,
                  figsize=(10, 5), title=None, savefig=None):
    """Display properties and AVA curves for an interface
    with two infinite half-space layers

    Parameters
    ----------
    vp1 : :obj:`float`
        P-wave velocity of the upper medium
    vs1 : :obj:`float`
        S-wave velocity of the upper medium
    rho1 : :obj:`float`
        Density of the upper medium
    vp2 : :obj:`float`
        P-wave velocity of the lower medium
    vs2 : :obj:`float`
        S-wave velocity of the lower medium
    rho2 : :obj:`float`
        Density of the lower medium
    theta : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
        (i.e. the second term divided by :math:`sin^2(\theta)`).
    methods : :obj:`tuple`
        Name of methods to be used
    colorsrefl : :obj:`tuple`
        Color to be used to display the reflection response for each method
    lwrefl : :obj:`tuple`
        Linewidth to be used to display the reflection responses
    colorpar : :obj:`str`
        Color to be used to display the parameters
    lwpar : :obj:`tuple`
        Linewidth to be used to display the parameters
    axs : :obj:`plt.axes`
        Axes handles. Note that in this case the left plot contains traces
        instead of a cartoon of the properties
    figsize : :obj:`tuple`, optional
             Size of figure
    title : :obj:`str`, optional
         Title of figure
    savefig : :obj:`str`, optional
         Figure filename (if ``None``, figure is not saved)

    Returns
    -------
    fig : :obj:`plt.figure`
       Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
       Axes handles

    """
    if isinstance(methods, str):
        methods = (methods, )
        colorsrefl = (colorsrefl, )

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('AVO modelling' if title is None else title, fontsize=18,
                     fontweight='bold', y=1.05)
        axs[0].fill_between(np.arange(0, 2), 1, 0.5, color='r',
                            interpolate=True, alpha=0.6)
        axs[0].fill_between(np.arange(0, 2), 0.5, 0, color='g',
                            interpolate=True, alpha=0.6)
        axs[0].text(0.5, 0.75, 'VP={}\nVS={}\nRHO={}'.format(vp1, vs1, rho1),
                    size=13,
                    ha='center', va='center', fontweight='bold')
        axs[0].text(0.5, 0.25, 'VP={}\nVS={}\nRHO={}'.format(vp2, vs2, rho2),
                    size=13,
                    ha='center', va='center', fontweight='bold')
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[0].set_title('Medium', fontweight='bold', fontsize=18)
        axs[0].axis('off')
        icurve = 1
    else:
        fig = None
        axs[0].plot([vp1, vp1, vp2, vp2], [0, 0.5, 0.5, 1], colorpar, lw=lwpar)
        axs[0].invert_yaxis()
        axs[1].plot([vs1, vs1, vs2, vs2], [0, 0.5, 0.5, 1], colorpar, lw=lwpar)
        axs[1].invert_yaxis()
        axs[2].plot([rho1, rho1, rho2, rho2], [0, 0.5, 0.5, 1], colorpar, lw=lwpar)
        axs[2].invert_yaxis()
        axs[0].set_title('VP', fontweight='bold', fontsize=14)
        axs[1].set_title('VS', fontweight='bold', fontsize=14)
        axs[2].set_title('Rho', fontweight='bold', fontsize=14)
        icurve = 3
    for method, color in zip(methods, colorsrefl):
        refl = _methods[method](vp1, vs1, rho1, vp2, vs2, rho2,
                                theta1=theta)
        axs[icurve].plot(theta, refl, color=color, lw=lwrefl, label=method)

    axs[icurve].set_title('Reflectivities', fontweight='bold', fontsize=14)
    axs[icurve].set_ylabel('Angles', fontweight='bold', fontsize=14)
    if len(methods)>1:
        axs[icurve].legend()
    plt.tight_layout()

    if fig is not None:
        plt.subplots_adjust(bottom=0.1)
        if savefig is not None:
            fig.savefig(savefig,
                        dpi=300, bbox_inches='tight')
    return fig, axs

