import os
import getpass
import datetime

import numpy as np
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import shapely.geometry as geometry

from descartes import PolygonPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pysubsurface.proc.seismicmod.poststack import zerooffset_mod

try:
    from IPython.display import display_html, display, Javascript
    displayflag=1
except:
    displayflag=0

FILEPATH = os.path.dirname(os.path.realpath(__file__))


###########
# private #
###########
def _rgb2hex(r,g,b):
    """From rgb to hexadecimal color

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
    hex = f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'
    return hex


def _set_black(fig, ax=None, cb=None):
    """Set figure and axis style to black (with white labels)

    Parameters
    ----------
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`, optional
        Axes handle
    cb : :obj:`matplotlib.colorbar.Colorbar`, optional
        Colorbar

    Returns
    -------
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
        Axes handle
    cb : :obj:`matplotlib.colorbar.Colorbar`
        Colorbar

    """
    fig.patch.set_facecolor('black')
    if ax is not None:
        ax.patch.set_facecolor('black')
        ax.xaxis.set_tick_params(color='white', labelcolor='white')
        ax.yaxis.set_tick_params(color='white', labelcolor='white')
        ax.grid(ls='--', alpha=0.5)
    if cb is not None:
        cb.ax.yaxis.set_tick_params(color='white')
        cb.outline.set_edgecolor(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    return fig, ax, cb


def _discrete_cmap(N, base_cmap='jet'):
    """Create an N-bin discrete colormap from the specified input map

    Parameters
    ----------
    N : :obj:`int`
        Number of elements
    base_cmap : :obj:`str`, optional
        Base colormap

    Returns
    -------
    cmap : :obj:`matplotlib.colors.LinearSegmentedColormap`
        Discrete colorbar

    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    cmap = base.from_list(cmap_name, color_list, N)
    return cmap


def _discrete_cmap_indexed(colors):
    """Create an N-bin discrete colormap from the set of colors

    Parameters
    ----------
    colors : :obj:`list` or :obj:`tuple`
        Colors

    Returns
    -------
    cmap : :obj:`matplotlib.colors.LinearSegmentedColormap`
        Discrete colorbar

    """
    cmap = pltcolors.ListedColormap(colors, 'indexed')
    return cmap


def _wiggletrace(ax, tz, trace, center=0, cpos='r', cneg='b'):
    """Plot a seismic wiggle trace onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    trace : :obj:`np.ndarray `
        Wiggle trace
    center : :obj:`float`, optional
        Center of x-axis where to switch color
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    ax.fill_betweenx(tz, center, trace, where=(trace > center),
                     color=cpos, interpolate=True)
    ax.fill_betweenx(tz, center, trace, where=(trace <= center),
                     color=cneg, interpolate=True)
    ax.plot(trace, tz, 'k', lw=1)
    return ax


def _wiggletracecomb(ax, tz, x, traces, scaling=None,
                     cpos='r', cneg='b'):
    """Plot a comb of seismic wiggle traces onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    x : :obj:`np.ndarray `
        Lateral axis
    traces : :obj:`np.ndarray `
        Wiggle traces
    scaling : :obj:`float`, optional
        Scaling to apply to each trace
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    dx = np.abs(x[1]-x[0])
    tracesmax = np.max(traces, axis=1)
    tracesmin = np.min(traces, axis=1)
    dynrange = tracesmax - tracesmin
    maxdynrange = np.max(dynrange)
    if scaling is None:
        scaling = 2*dx/maxdynrange
    else:
        scaling = scaling*2*dx/maxdynrange

    for ix, xx in enumerate(x):
        trace = traces[ix]
        _wiggletrace(ax, tz, xx + trace * scaling, center=xx,
                     cpos=cpos, cneg=cneg)
    ax.set_xlim(x[0]-1.5*dx, x[-1]+1.5*dx)
    ax.set_ylim(tz[-1], tz[0])
    ax.set_xticks(x)
    ax.set_xticklabels([str(xx) for xx in x])

    return ax


def _seismic_polarity(ax, poscolor='red', negcolor='blue', cmap=None,
                      normal=True, lw=7, fs=30):
    """Display cartoon of seismic polarity plot within axes

    Parameters
    ----------
    ax : :obj:`plt.axes`
         Axes handle
    poscolor : :obj:`str`, optional
         Color of positive wiggle
    negcolor : :obj:`str`, optional
         Color of negative wiggle
    cmap : :obj:`str`, optional
         Colormap to use to identify color for positive and negative wiggle
         (if ``None`` use those from ``poscolor`` and ``negcolor``
    normal : :obj:`str`, optional
         Use normal (``True``) or reverse (``False``) polarity
    lw : :obj:`int`, optional
         Linewidth
    fs : :obj:`str`, optional
         Fontsize

    Returns
    -------
    ax : :obj:`plt.axes`, optional
        Axes handle

    """
    nz = 150
    ai = np.zeros(nz)
    ai[nz // 3:2 * nz // 3] = 1

    wav = np.cos(np.linspace(-np.pi / 2, np.pi / 2, nz // 3 + 1))
    if not normal:
        wav = -wav
    seis, refl = zerooffset_mod(ai, wav)

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        poscolor = cmap(cmap.N)
        negcolor = cmap(0)

    ax.plot(ai, np.arange(nz), 'k', lw=lw)
    ax.axhline(100, 0.1, 0.5 if normal else 0.95, color='k', lw=lw//2, ls='--')
    ax.axhline(50, 0.1, 0.95 if normal else 0.5, color='k', lw=lw//2, ls='--')
    ax.text(0.1, 10, '- AI +', fontsize=fs, fontweight='bold')
    ax.fill_betweenx(np.arange(nz), 3, 3 + seis, where=seis >= 0,
                     facecolor=poscolor, interpolate=True)
    ax.fill_betweenx(np.arange(nz), 3, 3 + seis, where=seis <= 0,
                     facecolor=negcolor, interpolate=True)
    ax.plot(3 + seis, np.arange(nz), 'k', lw=lw)
    ax.text(3., 10, '-  +', fontsize=fs, fontweight='bold',
            horizontalalignment='center')
    ax.text(3., 10, '-  +', fontsize=fs, fontweight='bold',
            horizontalalignment='center')
    ax.set_xticks([]), ax.set_yticks([])
    ax.invert_yaxis()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    return ax


def _pie_in_axes(ax, data, labels, colors, x, y, size,
                 xlim=None, ylim=None,
                 autopct='%1.1f%%'):
    """Display pie plot within axes

    Parameters
    ----------
    ax : :obj:`plt.axes`
         Axes handle
    data : :obj:`np.ndarray`
         Data to display
    labels : :obj:`list`, optional
         Labels to be used for each data point in pie
    colors : :obj:`list`, optional
         Colors to be used for each data point in pie
    x : :obj:`float`, optional
         x-coordinate of center of pie
    y : :obj:`float`, optional
         y-coordinate of center of pie
    size : :obj:`float`, optional
         Size of pie
    xlim : :obj:`tuple`, optional
        Plotting limits in x axis
    ylim : :obj:`tuple`, optional
        Plotting limits in y axis
    autopct : :obj:`str`, optional
        Format of numbers in pie

    Returns
    -------
    ax_sub : :obj:`plt.axes`, optional
         Subset axes handle

    """
    if xlim is None and ylim is None:
        drawfig = True
    else:
        # check if wells is inside axis
        ax_x, ax_y = ax.get_xlim(), ax.get_ylim()
        drawfig = x > ax_x[0] and \
                  x < ax_x[1] and \
                  y > ax_y[0] and \
                  y < ax_y[1]

    if drawfig:
        ax_sub = inset_axes(ax, width=size, height=size, loc=10,
                            bbox_to_anchor=(x, y),
                            bbox_transform=ax.transData,
                            borderpad=0)
        ax_sub.pie(data, labels=labels, colors=colors,
               autopct=autopct, shadow=True, startangle=140)
        ax_sub.set_aspect("equal")
    else:
        ax_sub = None
    return ax_sub


##########
# public #
##########
def display_joint_dataframes(dfs, titles=[], displayflag=False):
    """Return (and display) set of dataframes with title alongside each others

    Parameters
    ----------
    dfs : :obj:`list`
        Dataframes to display
    tz : :obj:`list`, optional
        Titles to add to dataframes
    displayflag : :obj:`bool`, optional
        Display joint dataframes

    Returns
    -------
    html_str : :obj:`str`
        Html for visualization via ``display_html`` routine
        (e.g., ``display_html(html_str, raw=True)``)

    """
    html_str = ''
    if titles:
        html_str += '<tr>' + ''.join(f'<td style="text-align:center">{name}</td>' for name in titles) + '</tr>'
    html_str += '<tr>' + ''.join(
        f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' for df in dfs) + '</tr>'
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table', 'table style="display:inline"')

    if displayflag:
        display_html(html_str, raw=True)
    return html_str


def plot_polygon(ax, x, y, scatter=False, color='k', label=None):
    """Display set of dataframes with title alongside each others

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
        Axes handle
    x : :obj:`np.ndarray`
        x-coordinates
    y : :obj:`np.ndarray`
        y-coordinates
    scatter : :obj:`bool`, optional
        Scatter points (``True``) or simply display polygon (``False``)
    color : :obj:`str`, optional
        Color
    label : :obj:`str`, optional
        Label

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    points = [geometry.shape({"type": "Point", "coordinates": (x, y)})
              for x, y in zip(x, y)]
    point_collection = geometry.MultiPoint(list(points))
    convex_hull = point_collection.convex_hull
    x_min, y_min, x_max, y_max = convex_hull.bounds
    patch = PolygonPatch(convex_hull, fc=color, ec=color, lw=2,
                         fill=True, alpha=0.2)
    ax.add_patch(patch)
    if label is not None:
        ax.text((x_min+x_max)/2, (y_min+y_max)/2, label,
                ha= "center", va = "center",
                color= color,
                bbox= dict(boxstyle="round",
                           fc=(1., 1., 1.),
                           ec=color, ))
    if scatter:
        ax.plot(x, y, '.', ms=1, color=color, alpha=0.3)
    return ax


def plot_table(df, ax=None, fig=None, figsize=(10, 12),
               nrowstable=50, colors=None,
               title=None, savefig=None):
    """Plot dataframe as table or multiple tables (and save into png)

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame to display
    ax : :obj:`np.ndarray`, optional
        Axes handle (if ``None`` create a new figure)
    figsize : :obj:`tuple`, optional
        Figure size
    nrowstable : :obj:`int`, optional
        Number of rows per table size
    colors : :obj:`list` or :obj:`numpy.ndarray`, optional
        Colors of each cell of the dataframe as 2d list
    title : :obj:`str`, optional
        Figure title
    savefig : :obj:`str`, optional
        Path of figure to save

    Returns
    -------
    fig : :obj:`np.ndarray`, optional
        Figure handle
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    nrows = df.shape[0]
    if nrowstable < nrows:
            startrows = np.arange(0, nrows, nrowstable)
            endrows = startrows + nrowstable
            endrows[-1] = nrows
    else:
        startrows = [0,]
        endrows = [nrows]
    nsubplots = len(startrows)
    if colors is None:
        colors = np.full(df.shape, 'w')
    pass
    if ax is None:
        fig, axs = plt.subplots(1, nsubplots, figsize=figsize)
    if nsubplots == 1:
        axs = [axs, ]
    for i, (startrow, endrow) in enumerate(zip(startrows, endrows)):
        axs[i].axis("off")
        axs[i].table(cellText=df.values[startrow:endrow],
                     rowLabels=df.index[startrow:endrow],
                     colLabels=df.columns,
                     cellColours=colors[startrow:endrow],
                     fontsize=12,
                     bbox=(0, 0, 1, 1))
    if len(axs) == 1:
        axs[i].set_title(title, fontsize=12, fontweight='bold')
    else:
        fig.suptitle(title, fontsize=12, y=1.01, fontweight='bold')
    if savefig is not None:
        if fig is None:
            raise ValueError('Provide fig handle or omit axes handle '
                             '(in this case) a new figure will be created...')
        fig.savefig(savefig, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), transparent=True)
    return fig, axs