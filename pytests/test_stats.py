import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from geostatsmodels.model import spherical
from pysubsurface.utils.stats import covariance, drawsamples, average_stats,\
    _estimate_variogram_range

par1 = {'mean': 1, 'cov': [[2, 1],[1, 2]], 'nreals':1}
par2 = {'mean': [1, 2], 'cov': [[2, 1],[1, 2]], 'nreals':1}
par3 = {'mean': 1, 'cov': [[2, 1],[1, 2]], 'nreals':10000000}
par4 = {'mean': [1, 2], 'cov': [[2, 1],[1, 2]], 'nreals':10000000}


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_covariance_drawsamples(par):
    """Draw samples and covariance (check it is the same as the undelying
    distribution used to draw samples)
    """
    np.random.seed(10)
    cov = np.array(par['cov'])
    features = drawsamples(par['mean'], cov, par['nreals'])
    covest = covariance(features)
    assert features.shape == (cov.shape[0], par['nreals'])

    if par['nreals'] > 1:
        print(covest, cov)
        assert np.allclose(covest, cov, atol=1e-2)


def test_average_stats():
    """Compute average of means and covariances and check if it is the same as
     estimating the covariance from the overall population
    """
    np.random.seed(10)
    nreals = [1000000, 200000, 50000]
    mean = [0, 0]
    cov1 = np.array([[2, 1], [1, 2]])
    cov2 = np.array([[3, 2], [2, 3]])
    cov3 = np.array([[6, 2], [2, 6]])

    samples1 = drawsamples(mean, cov1, nreals=nreals[0])
    samples2 = drawsamples(mean, cov2, nreals=nreals[1])
    samples3 = drawsamples(mean, cov3, nreals=nreals[2])
    samples = np.hstack((samples1, samples2, samples3))

    # mean estimation
    meanest = np.mean(samples, axis=1)
    mean1est = np.mean(samples1, axis=1)
    mean2est = np.mean(samples2, axis=1)
    mean3est = np.mean(samples3, axis=1)
    meanestave = average_stats([mean1est, mean2est, mean3est], nreals)

    # covariance estimation
    covest = covariance(samples).values
    cov1est = covariance(samples1).values
    cov2est = covariance(samples2).values
    cov3est = covariance(samples3).values
    covestave = average_stats([cov1est, cov2est, cov3est], nreals)

    assert np.allclose(meanest, meanestave, atol=1e-2)
    assert np.allclose(covest, covestave, atol=1e-2)


def test_variogram_range():
    """Estimate range of variogram and check it is the same as model
    """
    np.random.seed(10)

    h = np.arange(30)
    sv = spherical
    a, sill = 10, 5
    var = sv(h, a, sill)
    aest = _estimate_variogram_range(h, var, sill, sv)

    assert_array_almost_equal(a, aest, decimal=2)
