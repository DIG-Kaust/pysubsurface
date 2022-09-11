import pytest
import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.objects import Logs
from pysubsurface.utils.wavelets import ricker
from pysubsurface.proc.seismicmod.poststack import zerooffset_wellmod
from pysubsurface.proc.seismicmod.avo import prestack_wellmod


def test_zerooffsetmod():
    """Zero-offset modelling
    """
    logs = Logs('testdata/Well/Logs/Vertical.las')
    logs.dataframe()

    wav, twav, wcenter = ricker(np.arange(301) * 0.001, f0=30)

    trace, t, _, _ = zerooffset_wellmod(logs, 'MD', 1, wav, wavcenter=None,
                                        vp='LFP_VP', rho='LFP_RHOB',
                                        ax=None, figsize=(12, 9), title=None,
                                        savefig=None)

    trace, t, _, _ = zerooffset_wellmod(logs, 'MD', 1, wav, wavcenter=None,
                                        vp='LFP_VP', rho='LFP_RHOB',
                                        zlim=(2800, 3200),
                                        ax=None, figsize=(12, 9), title=None,
                                        savefig=None)

    # clean up
    plt.close('all')


def test_prestackmod():
    """Pre-stack modelling
    """
    logs = Logs('testdata/Well/Logs/Vertical.las')
    logs.dataframe()

    wav, twav, wcenter = ricker(np.arange(301) * 0.001, f0=30)

    gather, t, _, _ = prestack_wellmod(logs, 'MD', np.arange(0, 40),
                                       1, wav, wavcenter=None,
                                       vp='LFP_VP', vs='LFP_VS', rho='LFP_RHOB',
                                       ax=None, figsize=(12, 9), title=None,
                                       savefig=None)

    trace, t, _, _ = prestack_wellmod(logs, 'MD', np.arange(0, 40),
                                      1, wav, wavcenter=None,
                                      vp='LFP_VP', vs='LFP_VS', rho='LFP_RHOB',
                                      zlim=(2800, 3200),
                                      ax=None, figsize=(12, 9), title=None,
                                      savefig=None)

    # clean up
    plt.close('all')
