import pytest
import numpy as np

from pysubsurface.utils.wavelets import ricker
from pysubsurface.proc.seismicmod.waveletest import statistical_wavelet

par1 ={'dt':0.004, 'ntw':51, 'f0':10, 'nfft':2**10}
par2 ={'dt':0.004, 'ntw':50, 'f0':20, 'nfft':2**10}


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_statistical_wavelet(par):
    """Statistical wavelet estimation
    """
    tw = np.arange(par['ntw']) * par['dt']
    wav, tw, wav_c = ricker(tw, par['f0'])

    # create 1d, 2d, 3d data
    d = wav.copy()
    d2 = np.outer(np.ones(10), wav)
    d3 = np.outer(np.ones((10, 20)), wav).reshape(10, 20, len(tw))

    # estimate wavelet
    wavest, _, twest, fwest, wavest_c = \
        statistical_wavelet(d, ntwest=par['ntw'] if par['ntw'] % 2 \
                            else par['ntw']-1,
                            dt=par['dt'])
    wavest2, _, twest2, fwest2, wavest2_c = \
        statistical_wavelet(d2, ntwest=par['ntw'] if par['ntw'] % 2 \
                            else par['ntw']-1,
                            dt=par['dt'])
    wavest3, _, twest3, fwest3, wavest3_c = \
        statistical_wavelet(d3, ntwest=par['ntw'] if par['ntw'] % 2 \
                            else par['ntw']-1,
                            dt=par['dt'])

    np.testing.assert_almost_equal(wav, wavest, decimal=3)
    np.testing.assert_array_equal(tw, twest)
    assert wav_c == wavest_c

    np.testing.assert_almost_equal(wav, wavest2, decimal=3)
    np.testing.assert_array_equal(tw, twest)
    assert wav_c == wavest2_c

    np.testing.assert_almost_equal(wav, wavest3, decimal=3)
    np.testing.assert_array_equal(tw, twest3)
    assert wav_c == wavest3_c

