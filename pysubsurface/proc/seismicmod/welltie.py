import numpy as np
import matplotlib.pyplot as plt
import pylops

from pysubsurface.proc.seismicmod.poststack import zerooffset_wellmod
from pysubsurface.visual.utils import _wiggletrace
from pysubsurface.objects.utils import _findclosest_well_seismicsections
from pysubsurface.utils.utils import findclosest
from scipy.stats import pearsonr


def welltie(well, seismic, ai, depth, tzlims, kind='stat', ntwav=51,
            reverse=False, seisshift=0, seismicname=None):
    r"""Well-tie

    kind = stat, inverse
    # TO DO!!
    """
    if ntwav % 2 == 0:
        ntwav += 1

    if kind == 'stat':
        wav, wavf = seismic.estimate_wavelet(
                method='stat', ntwest=(ntwav + 1) // 2, tzlims=tzlims,
                jil=10, jxl=10, plotflag=False,
                **dict(nfft=2 ** 10, nsmooth=4))
        if reverse:
            wav = -wav

        # modelled trace
        modtrace, tzmod = zerooffset_wellmod(well.welllogs,
                                             depth,
                                             seismic.dtz,
                                             wav,
                                             ai=ai)[:2]

        # real trace
        ilwell, xlwell = _findclosest_well_seismicsections(well, seismic)

        realtrace = seismic.data[findclosest(seismic.ilines, ilwell),
                                 findclosest(seismic.xlines, xlwell)]
        tzreal = seismic.tz

        # extract trace in interval of interest
        itzin_mod, itzend_mod = \
            findclosest(tzmod, tzlims[0]), findclosest(tzmod, tzlims[1])
        itzin_real, itzend_real = \
            findclosest(tzreal + seisshift, tzlims[0]), \
            findclosest(tzreal + seisshift, tzlims[1])
        tz = tzmod[itzin_mod:itzend_mod]
        modtrace = modtrace[itzin_mod:itzend_mod]
        realtrace = realtrace[itzin_real:itzend_real]

        # find wavelet scaling as energy ratio
        wavscaling = np.sqrt(np.sum(realtrace ** 2) / np.sum(modtrace ** 2))
        modtrace = wavscaling * modtrace
        wav = wavscaling * wav

    elif kind=='inverse':
        # ai profile
        tzlog = well.welllogs.logs[depth]
        tzreglog = np.arange(np.nanmin(tzlog), np.nanmax(tzlog),
                             seismic.dtz)
        aicurve, tzlog = well.welllogs.resample_curve(ai, zaxis=depth)
        aicurve = np.interp(tzreglog, tzlog, aicurve)
        aicurve[np.isnan(aicurve)] = np.nanmean(aicurve)

        # real trace
        ilwell, xlwell = _findclosest_well_seismicsections(well, seismic)

        realtrace = seismic.data[findclosest(seismic.ilines, ilwell),
                                 findclosest(seismic.xlines, xlwell)]
        tzreal = seismic.tz

        # extract trace in interval of interest
        itzin_mod, itzend_mod = \
            findclosest(tzreglog, tzlims[0]), findclosest(tzreglog, tzlims[1])
        itzin_real, itzend_real = \
            findclosest(tzreal + seisshift, tzlims[0]), findclosest(tzreal + seisshift, tzlims[1])
        tz = tzreal[itzin_real:itzend_real] + seisshift
        aicurve = aicurve[itzin_mod:itzend_mod]
        realtrace = realtrace[itzin_real:itzend_real]

        # setup inversion
        m = np.stack((np.log(aicurve),
                      np.zeros_like(aicurve),
                      np.zeros_like(aicurve)), axis=1)
        theta = np.zeros(1)

        Wavesop = \
            pylops.avo.prestack.PrestackWaveletModelling(m, theta, nwav=ntwav,
                                                         wavc=ntwav // 2,
                                                         vsvp=0.5,
                                                         linearization='fatti')

        if seisshift == 0:
            # Create regularization operator
            D2op = pylops.SecondDerivative(ntwav, dtype='float64')

            # Invert for wavelet
            wav, istop, itn, r1norm, r2norm = \
                pylops.optimization.leastsquares.RegularizedInversion(Wavesop,
                                                                      [D2op],
                                                                      realtrace,
                                                                      epsRs=[5e-1],
                                                                      returninfo=True,
                                                                      **dict(damp=np.sqrt(1e-4),
                                                                             iter_lim=200, show=0))
        else:
            # Create symmetrize operator
            Sop = pylops.Symmetrize((ntwav + 1) // 2)

            # Create smoothing operator
            Smop = pylops.Smoothing1D(5, dims=((ntwav + 1) // 2,),
                                      dtype='float64')

            # Invert for wavelet
            wav = \
                pylops.optimization.leastsquares.PreconditionedInversion(
                    Wavesop, Sop * Smop,
                    realtrace,
                    returninfo=False,
                    **dict(damp=np.sqrt(1e-4),
                           iter_lim=200,
                           show=0))

        modtrace = Wavesop * wav
    else:
        raise NotImplementedError

    # display
    t2 = np.arange((ntwav + 1) // 2) * (seismic.tz[1] - seismic.tz[0])
    t2 = np.concatenate((-t2[::-1], t2[1:]))
    fig = plt.figure(figsize=(20, 10))
    title = 'Well-tie for {}'.format(well.wellname)
    if seismicname is not None:
        title += ' and {}'.format(seismicname)
    fig.suptitle(title, y=1.05,
                 fontsize=25, fontweight='bold')
    ax0 = plt.subplot2grid((4, 6), (0, 0), rowspan=4)
    ax1 = plt.subplot2grid((4, 6), (0, 1), rowspan=4)
    ax2 = plt.subplot2grid((4, 6), (0, 2), rowspan=4)
    ax3 = plt.subplot2grid((4, 6), (0, 3), rowspan=4)
    ax4 = plt.subplot2grid((4, 6), (0, 4), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((4, 6), (2, 4), rowspan=2, colspan=2)

    _, ax0 = well.welllogs.visualize_logcurve(ai, depth, ax=ax0)
    ax1 = _wiggletrace(ax1, tz, modtrace)
    ax1.set_title('Modelled')
    ax2 = _wiggletrace(ax2, tz, realtrace)
    ax2.set_title('Real')
    ax3 = _wiggletrace(ax3, tz, realtrace - modtrace)
    ax3.set_title('Residual')
    ax4.axis('off')

    text = 'Wavelet estimation:\n\n' \
           'method: {},\n' \
           'ntwav: {},\n'.format(kind, ntwav, reverse)
    if kind == 'stat':
        text += 'reverse: {}\n' \
                'seisshift: {}\n'.format(reverse, seisshift)
    ax4.text(0.27, 0.5, 'Wavelet estimation:\n\n'
                        'method: {},\n'
                        'ntwav: {},\n'
                        'reverse: {}'.format(kind, ntwav, reverse),
             va="center", color='k', fontsize=16,
             bbox=dict(boxstyle="round", fc=(1., 1., 1.), ec='k', ))
    ax5.plot(t2, wav, 'k', lw=2)
    ax5.set_title('Statistical Wavelet')

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_ylim(tzlims[1], tzlims[0])
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-1.2 * np.abs(realtrace).max(),
                    1.2 * np.abs(realtrace).max())
    well.view_picks_and_intervals(
        np.array([ax0, ax1, ax2, ax3]), depth=depth,
        ylim=tzlims, level=[0, 1], labels=False)
    plt.tight_layout()

    return wav, modtrace, realtrace


def welltie_shift_finder(well, seismic, ai, depth, tzlims, wav,
                         iiloffset=10, ixloffset=10, flipaxis=False,
                         originlower=False, flipil=False, flipxl=False,
                         seismicname=None):
    """Identify optimal time or space shift between modelled and real seismic
    at well location.

    Create synthetic seismic trace at well location and compare it with real
    seismic. Comparison is performed by means of the Pearson correlation
    coefficient both by time/depth shifting the reference real seismic trace and
    by takinn traces in neighbour of ``iiloffset`` inlines and ``ixloffset``
    crosslines, respectively

    Parameters
    ----------
    well : :obj:`pysubsurface.objects.Well.Well`
        Well object
    seismic : :obj:`pysubsurface.objects.Seismic.Seismic`
        Seismic object
    ai : :obj:`str`
        Name of log containing acoustic impedance profile
    depth: :obj:`str`
        Name of log containing depth profile
    tzlims : :obj:`tuple`
        Start and end time/depth where correlation will be performed
    wav : :obj:`np.ndarray`
        Wavelet
    iiloffset : :obj:`float`, optional
        Half-width of the window of traces in inline direction to
        use for comparison
    ixloffset : :obj:`float`, optional
        Half-width of the window of traces in crossline direction to
        use for comparison
    flipaxis : :obj:`bool`, optional
             Flip x and y axis (``True``) or not (``False``)
        originlower : :obj:`bool`, optional
             Origin at bottom-left (``True``) or top-left (``False``)
        flipy : :obj:`bool`, optional
             flip y axis (``True``) or not (``False``)
        flipx : :obj:`bool`, optional
             flip x axis (``True``) or not (``False``)
    seismicname : :obj:`float`, optional
        Name of seismic to add to the title

    Returns
    -------
    fig : :obj:`plt.figure`
        Figure handle for trace plot
        Axes handle for trace plots
    fig_seis : :obj:`plt.figure`
        Figure handle for seismic crossections plot
    axs_seis : :obj:`plt.axes`
        Axes handles for seismic crossections plot

    """
    # modelled trace
    modtrace, tzmod = zerooffset_wellmod(well.welllogs,
                                         depth,
                                         seismic.dtz,
                                         wav,
                                         ai=ai)[:2]

    # real trace
    ilwell, xlwell = _findclosest_well_seismicsections(well, seismic)

    realtrace = seismic.data[findclosest(seismic.ilines, ilwell),
                             findclosest(seismic.xlines, xlwell)]
    tzreal = seismic.tz

    # extract trace in interval of interest
    itzin_mod, itzend_mod = findclosest(tzmod, tzlims[0]), findclosest(tzmod,
                                                                       tzlims[
                                                                           1])
    itzin_real, itzend_real = findclosest(tzreal, tzlims[0]), \
                              findclosest(tzreal, tzlims[1])
    tz = tzreal[itzin_real:itzend_real]
    modtrace = modtrace[itzin_mod:itzend_mod]
    realtrace = realtrace[itzin_real:itzend_real]

    # time xcorr
    xcorrt = np.correlate(modtrace, realtrace, mode='full') / np.sqrt(
        np.sum(modtrace ** 2) * np.sum(realtrace ** 2))
    t2 = np.arange(tz.size) * (tz[1] - tz[0])
    t2 = np.concatenate((-t2[::-1], t2[1:]))

    # spatial panel
    ilaroundwell = np.arange(ilwell - iiloffset * seismic.dil,
                             ilwell + iiloffset * seismic.dil + seismic.dil,
                             seismic.dil)
    xlaroundwell = np.arange(xlwell - iiloffset * seismic.dxl,
                             xlwell + iiloffset * seismic.dxl + seismic.dxl,
                             seismic.dxl)

    xcorr = np.zeros((2 * iiloffset + 1, 2 * ixloffset + 1))
    for iil, il in enumerate(ilaroundwell):
        for ixl, xl in enumerate(xlaroundwell):
            tmptrace = seismic.data[findclosest(seismic.ilines, il),
                                    findclosest(seismic.xlines, xl)]
            tmptrace = tmptrace[itzin_real:itzend_real]
            xcorr[iil, ixl] = pearsonr(modtrace, tmptrace)[0]
    xcorrmax = xcorr.max()
    iilmax, ixlmax = np.argwhere(xcorrmax == xcorr)[0]

    # extract real trace at corrmax location
    realtrace_max = \
        seismic.data[findclosest(seismic.ilines, ilaroundwell[iilmax]),
                     findclosest(seismic.xlines, xlaroundwell[ixlmax])]
    realtrace_max = realtrace_max[itzin_real:itzend_real]

    # display
    fig = plt.figure(figsize=(20, 13))
    title = 'Well-tie correlations for {}'.format(well.wellname)
    if seismicname is not None:
        title += ' and {}'.format(seismicname)
    fig.suptitle(title, y=1.05,
                 fontsize=25, fontweight='bold')
    ax0 = plt.subplot2grid((4, 7), (0, 0), rowspan=3, colspan=3)
    ax1 = plt.subplot2grid((4, 7), (3, 0), colspan=3)
    ax2 = plt.subplot2grid((4, 7), (0, 3), rowspan=4)
    ax3 = plt.subplot2grid((4, 7), (0, 4), rowspan=4)
    ax4 = plt.subplot2grid((4, 7), (0, 5), rowspan=4)
    ax5 = plt.subplot2grid((4, 7), (0, 6), rowspan=4)

    if flipaxis:
        im = ax0.imshow(xcorr.T, vmin=-1, vmax=1, cmap='gist_rainbow',
                        origin='lower' if originlower else None,
                        extent=(ilaroundwell[0], ilaroundwell[-1],
                                xlaroundwell[0 if originlower else -1],
                                xlaroundwell[-1 if originlower else 0]))
        ax0.contour(xcorr.T, 5, colors='k',
                    origin='lower' if originlower else None,
                    extent=(ilaroundwell[0], ilaroundwell[-1],
                            xlaroundwell[0], xlaroundwell[-1]))
        ax0.plot(ilwell, xlwell, '.k', ms=30)
        ax0.plot(ilaroundwell[iilmax], xlaroundwell[ixlmax], '.r', ms=30)
    else:
        im = ax0.imshow(xcorr, vmin=-1, vmax=1, cmap='gist_rainbow',
                        origin='lower' if originlower else None,
                        extent=(xlaroundwell[0], xlaroundwell[-1],
                                ilaroundwell[0 if originlower else -1],
                                ilaroundwell[-1 if originlower else 0]))
        ax0.contour(xcorr, 5, colors='k',
                    extent=(xlaroundwell[0], xlaroundwell[-1],
                            ilaroundwell[0], ilaroundwell[-1]))
        ax0.plot(xlwell, ilwell, '.k', ms=30)
        ax0.plot(xlaroundwell[ixlmax], ilaroundwell[iilmax], '.r', ms=30)

    plt.colorbar(im, ax=ax0)
    ax0.set_title(
        'Well-tie spatial correlation\n({} @well IL={}-XL={})\n({} '
        '@max IL={}-XL={})'.format(xcorr[iiloffset, ixloffset], ilwell,
                                   xlwell, xcorrmax, ilaroundwell[iilmax],
                                   xlaroundwell[ixlmax]))
    ax0.set_ylabel('XL' if flipaxis else 'IL')
    ax0.set_xlabel('IL' if flipaxis else 'XL')
    ax0.axis('tight')
    if (flipil and flipaxis) or (flipxl and not flipaxis):
        ax0.invert_xaxis()
    if (flipxl and flipaxis) or (flipil and not flipaxis):
        ax0.invert_axis()
    ax1.plot(t2, xcorrt, 'k', lw=2)
    ax1.set_title('Well-tie time correlation ({} @ t={})'.format(xcorrt.max(),
                                                                 t2[np.argmax(
                                                                     xcorrt)]))
    ax2 = _wiggletrace(ax2, tz, modtrace)
    ax2.set_title('Modelled')
    ax3 = _wiggletrace(ax3, tz, realtrace)
    ax3.set_title('Real @ well')
    ax4 = _wiggletrace(ax4, tz + t2[np.argmax(xcorrt)], realtrace)
    ax4.set_title('Real @ timemax')
    ax5 = _wiggletrace(ax5, tz, realtrace_max)
    ax5.set_title('Real @ spatmax')
    for ax in (ax2, ax3, ax4, ax5):
        ax.set_ylim(tzlims[1], tzlims[0] + t2[np.argmax(xcorrt)])
        ax.set_xlim(-1.2*np.abs(realtrace).max(), 1.2*np.abs(realtrace).max())
    well.view_picks_and_intervals(
        np.array([ax2, ax3, ax4, ax5]), depth=depth,
        ylim=tzlims, level=[0, 1], labels=True)
    plt.tight_layout()

    # time correlation
    fig_seis_t, axs_seis_t = \
        seismic.view(ilplot=findclosest(seismic.ilines, ilwell),
                     xlplot=findclosest(seismic.xlines, xlwell),
                     tzoom_index=False, tzoom=tzlims, cmap='seismic', clip=0.6,
                     title=(seismicname if seismicname is not None
                     else '') + ' comparision @ max time correlation',
                     figsize=(20, 17))
    _wiggletrace(axs_seis_t[0][0], tz - t2[np.argmax(xcorrt)],
                 xlwell + 5e1 * modtrace/np.abs(modtrace).max(),
                 center=xlwell)
    _wiggletrace(axs_seis_t[0][1], tz - t2[np.argmax(xcorrt)],
                 ilwell + 5e1 * modtrace / np.abs(modtrace).max(),
                 center=ilwell)

    # spatial correlation
    fig_seis_s, axs_seis_s = \
        seismic.view(ilplot=findclosest(seismic.ilines, ilaroundwell[iilmax]),
                     xlplot=findclosest(seismic.xlines, xlaroundwell[ixlmax]),
                     tzoom_index=False, tzoom=tzlims, cmap='seismic', clip=0.6,
                     title=(seismicname if seismicname is not None
                     else '') + ' comparision @ max spatial correlation',
                     figsize=(20, 17))
    _wiggletrace(axs_seis_s[0][0], tz,
                 xlaroundwell[ixlmax] + 5e1 * modtrace / np.abs(modtrace).max(),
                 center=xlaroundwell[ixlmax])
    _wiggletrace(axs_seis_s[0][1], tz,
                 ilaroundwell[iilmax] + 5e1 * modtrace / np.abs(modtrace).max(),
                 center=ilaroundwell[iilmax])

    return fig, (ax0, ax1, ax2, ax3, ax4, ax5), \
           fig_seis_t, axs_seis_t, fig_seis_s, axs_seis_s
