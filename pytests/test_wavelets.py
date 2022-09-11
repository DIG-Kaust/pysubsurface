import pytest
import matplotlib.pyplot as plt
from pysubsurface.utils.wavelets import *

par1 = {'nt': 21, 'dt':0.004} # odd samples
par2 = {'nt': 20, 'dt':0.004} # even samples


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ricker(par):
    """Create ricker wavelet and check size and central value
    """
    t = np.arange(par['nt'])*par['dt']
    wav, twav, wcenter = ricker(t, plotflag=True)
    plt.close('all')
    assert wav.shape[0] == (par['nt']-1 if par['nt'] % 2 == 0 else par['nt'])*2-1
    assert wav[wcenter] == 1


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_gaussian(par):
    """Create gaussian wavelet and check size and central value
    """
    t = np.arange(par['nt'])*par['dt']
    wav, twav, wcenter = gaussian(t, std=10, plotflag=True)
    plt.close('all')
    assert wav.shape[0] == (par['nt']-1 if par['nt'] % 2 == 0 else par['nt'])*2-1
    assert wav[wcenter] == 1


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_cosine(par):
    """Create cosine wavelet and check size and central value
    """
    t = np.arange(par['nt'])*par['dt']
    wav, twav, wcenter = cosine(t, extent=10)
    wav2, twav2, wcenter = cosine(t, extent=10, square=True, plotflag=True)
    plt.close('all')
    assert wav.shape[0] == (par['nt']-1 if par['nt'] % 2 == 0 else par['nt'])*2-1
    assert wav[wcenter] == 1
