import numpy as np
import matplotlib.cm as cm
import seaborn as sns

from pysubsurface.proc.seismicmod.avo import _methods
from pysubsurface.proc.seismicmod.avo import *
from pysubsurface.utils.stats import drawsamples


def ava_modelling_sensitivity(vp, vs, rho, theta, colors, method='shuey',
                              figsize=(12, 5), title=None, savefig=None):
    r"""Perform sensitivity analysis of AVA curves and intercept gradient for
    a set of profiles. The first profile is assumed to be the base case.

    Parameters
    ----------
    vp : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Set of P-wave velocities of a stack of layers of size
        :math:`[n_ {reals} \times n_{layers}]`
    vs : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Set of S-wave velocities of a stack of layers of size
        :math:`[n_ {reals} \times n_{layers}]`
    rho : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Set of densities of a stack of layers of size
        :math:`[n_ {reals} \times n_{layers}]`
    theta : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
        (i.e. the second term divided by :math:`sin^2(\theta)`).
    colors : :obj:`tuple`
        Color to be used for each layer (will use color of layer above to
        identify the same interface)
    method : :obj:`str`
        Name of methods to be used
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
    fig = plt.figure(figsize=figsize)
    fig.suptitle('AVO sensitivity' if title is None else title,
                 fontsize=18, fontweight='bold', y=1.05)
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((4, 4), (0, 1), rowspan=4)
    ax3 = plt.subplot2grid((4, 4), (0, 2), rowspan=4)
    ax4 = plt.subplot2grid((4, 4), (0, 3), rowspan=2)
    ax5 = plt.subplot2grid((4, 4), (2, 3), rowspan=2)

    nreals, nlayers = vp.shape
    zreal = \
        np.append(np.insert(np.array([[i/nlayers, i/nlayers]
                                      for i in range(1, nlayers)]).flatten(),
                            0, 0), 1)

    for ireal in range(nreals-1, -1, -1):
        vpreal = vp[ireal]
        vsreal = vs[ireal]
        rhoreal = rho[ireal]

        vp1, vp2 = vpreal[:-1], vpreal[1:]
        vs1, vs2 = vsreal[:-1], vsreal[1:]
        rho1, rho2 = rhoreal[:-1], rhoreal[1:]

        if method == 'shuey':
            refl, gi = _methods[method](vp1, vs1, rho1, vp2, vs2, rho2,
                                        theta1=theta, return_gradient=True)
        else:
            refl = _methods[method](vp1, vs1, rho1, vp2, vs2, rho2,
                                    theta1=theta)
            _, gi = _methods['shuey'](vp1, vs1, rho1, vp2, vs2, rho2,
                                      theta1=theta, return_gradient=True)
        for iint in range(nlayers-1):
            ax4.plot(theta, refl[:, iint], colors[iint],
                     lw=5 if ireal==0 else 2, alpha=1 if ireal==0 else 0.4)
            ax5.plot(gi['intercept'][iint], gi['gradient'][iint], '.',
                     color=colors[iint],
                     ms=15 if ireal == 0 else 10,
                     alpha=1 if ireal == 0 else 0.4)

        vpreal = np.array([[i, i] for i in vpreal]).flatten()
        vsreal = np.array([[i, i] for i in vsreal]).flatten()
        rhoreal = np.array([[i, i] for i in rhoreal]).flatten()

        ax1.plot(vpreal, zreal, 'k' if ireal == 0 else '#80aaff',
                 lw=5 if ireal == 0 else 0.5)
        ax2.plot(vsreal, zreal, 'k' if ireal == 0 else '#80aaff',
                 lw=5 if ireal == 0 else 0.5)
        ax3.plot(rhoreal, zreal, 'k' if ireal == 0 else '#80aaff',
                 lw=5 if ireal == 0 else 0.5)
        if ireal == 0:
            xlims = ax1.get_xlim()
            for ilayer in range(1, nlayers+1):
                ax1.fill_between(xlims, (ilayer-1)/nlayers,
                                 ilayer/nlayers,
                                 color=colors[ilayer-1],
                                 interpolate=True, alpha=0.4)

    # titles and labels
    ax1.set_title('VP', fontweight='bold', fontsize=14)
    ax2.set_title('VS', fontweight='bold', fontsize=14)
    ax3.set_title('Rho', fontweight='bold', fontsize=14)
    ax4.set_title('Reflectivities', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Angles')
    ax5.set_xlabel('I')
    ax5.set_ylabel('G')

    # set axes
    ax1.set_xlim(xlims)
    ax1.set_ylim(1, 0)
    ax2.set_ylim(1, 0)
    ax3.set_ylim(1, 0)
    ax4.set_xlim([theta[0], theta[-1]])

    xlim = np.abs(ax5.get_xlim()).max()
    ylim = np.abs(ax5.get_ylim()).max()
    ax5.set_xlim([-xlim, xlim])
    ax5.set_ylim([-ylim, ylim])
    ax5.axhline(0, color='k', lw=1.5, linestyle='--')
    ax5.axvline(0, color='k', lw=1.5, linestyle='--')
    ax5.grid()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    if savefig is not None:
        fig.savefig(savefig,
                    dpi=300, bbox_inches='tight')

    return fig, (ax1, ax2, ax3, ax4, ax5)


def welllogs_ava_sensitivity(well, intervals, level, nreals, theta, colors,
                             method='shuey', figsize=(12, 5),
                             title=None, savefig=None):
    """Sensitivity study on AVA response of a stacked succession of layers in a
    well

    Perform a sensitivity study on the expected AVO response at N interfaces
    using interval averaged welllogs. The core routine
    :func:`pysubsurface.proc.uncertainty.uncertainty.ava_modelling_sensitivity` works
    with any set of curves, this routine is a thin wrapper which adds functionality
    to automatically generate those set of curves in a well based on the
    variability of elastic parameters in various intervals.

    Parameters
    ----------
    well : :obj:`pysubsurface.objects.Well`
        Well object
    intervals : :obj:`pysubsurface.objects.Interval`
        Interval object
    level : :obj:`float`
        Level of intervals to use for statistics
    theta : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
        (i.e. the second term divided by :math:`sin^2(\theta)`).
    nreals : :obj:`int`
        Number of realizations
    colors : :obj:`tuple`
        Color to be used for each layer (will use color of layer above to
        identify the same interface)
        identify the same interface)
    method : :obj:`str`
        Name of methods to be used
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
    # find out intervals and associated colors
    layers = list(well.averaged_props['LFP_VP'].keys())
    intervals_level = intervals.df[intervals.df['Level']==level]
    colors = [intervals_level[intervals_level['Name'] == layer].iloc[0]['Color'] for
              layer in layers]

    # extract basecase
    vpbase = np.array([well.averaged_props['LFP_VP'][layer]['mean'] for layer in layers])
    vsbase = np.array([well.averaged_props['LFP_VS'][layer]['mean'] for layer in layers])
    rhobase = np.array([well.averaged_props['LFP_RHOB'][layer]['mean'] for layer in layers])
    covs = [well.averaged_props['Cov'][layer] for layer in layers]

    # create realizations
    nlayers = len(layers)
    vp_reals = np.zeros((nreals, nlayers))
    vs_reals = np.zeros((nreals, nlayers))
    rho_reals = np.zeros((nreals, nlayers))

    vp_reals[0] = vpbase
    vs_reals[0] = vsbase
    rho_reals[0] = rhobase

    for ireal in range(1, nreals):
        variations = np.squeeze(np.array(
            [np.squeeze(drawsamples(np.zeros(3), covs[ilayer].values)) for ilayer in
             range(nlayers)]))
        vp_reals[ireal] = vpbase + variations[:, 0]
        vs_reals[ireal] = vsbase + variations[:, 1]
        rho_reals[ireal] = rhobase + variations[:, 2]

    fig, axs = ava_modelling_sensitivity(vp_reals, vs_reals, rho_reals,
                                         theta=theta, colors=colors,
                                         method=method, figsize=figsize,
                                         title=title, savefig=savefig)
    return fig, axs