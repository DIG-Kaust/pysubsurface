import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import matplotlib.pyplot as plt

from geostatsmodels import model as geostatsmodel
from geostatsmodels import variograms as geostatsvariograms

import scipy.linalg as sp_lin
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut


def _estimate_variogram_range(lags, var, sill, model, verb=False):
    """Estimate range of a variogram given model definition
    and sample variogram

    """
    def fun(a, lags, variog, model):
        varmod = model(a)(lags)
        j = np.linalg.norm(variog - varmod)
        return j

    svm = lambda a: geostatsmodel.semivariance(model, [a, sill])
    f = lambda x: fun(x, lags, var, svm)
    a = 0.
    while a == 0:
        x0 = np.random.uniform(0.5, 10.)
        nl = minimize(f, x0, bounds=((0, np.inf), ),
                      tol=0., options=dict(disp=True, maxiter=200))
        if verb:
            print(nl)
        a = nl.x[0]
    return a

def confidence_to_std(confidence, interval=80):
    """Confidence to standard deviation

    Convert confidence interval into standard deviation
    for 80 (P90-P10), 90 (P5-P95), 95 (P2.5-P97.5), and 98 (P1-P99) intervals.
    See http://www.dinbelg.be/formulas.htm

    Parameters
    ----------
    confidence : :obj:`float`
        Confidence interval
    interval : :obj:`float`, optional
        Interval

    Returns
    -------
    std : :obj:`float`
        Standard deviation

    """
    if interval == 80:
        std = confidence / 1.282
    elif interval == 90:
        std = confidence / 1.65
    elif interval == 95:
        std = confidence / 1.96
    elif interval == 98:
        std = confidence / 2.33
    else:
        raise NotImplementedError('Interval provided is not allowed...')
    return std


def std_to_confidence(std, interval=80):
    """Confidence to standard deviation

    Convert confidence interval into standard deviation
    for 80 (P90-P10), 90 (P5-P95), 95 (P2.5-P97.5), and 98 (P1-P99) intervals.
    See http://www.dinbelg.be/formulas.htm

    Parameters
    ----------
    std : :obj:`float`
        Standard deviation
    interval : :obj:`float`, optional
        Interval

    Returns
    -------
    confidence : :obj:`float`
        Confidence interval

    """
    if interval == 80:
        confidence = std * 1.282
    elif interval == 90:
        confidence = std * 1.65
    elif interval == 95:
        confidence = std * 1.96
    elif interval == 98:
        confidence = std * 2.33
    else:
        raise NotImplementedError('Interval provided is not allowed...')
    return confidence


def average_stats(stats, nreals):
    """Average statistics (means, stdevs, covariance matrices)
    given the number of samples ``nsamples`` used to estimate ``stats``
    from each population

    Parameters
    ----------
    stats : :obj:`list`
        Statistics to average
    nreals : :obj:`list`
        Number of realizations for each population used to compute stats

    Returns
    -------
    stat : :obj:`np.ndarray`
        Averaged statistics'

    """
    allsamples = np.array(nreals).sum()
    weights = np.array(nreals)/allsamples
    stats = [stat*weight for stat, weight in zip(stats, weights)]
    stat = np.array(stats).sum(axis=0)
    return stat


def covariance(features, featurenames=None):
    """Compute covariance matrix from a list of np.array containing different
    features

    Parameters
    ----------
    features : :obj:`tuple` or :obj:`list`
        Arrays (:obj:`np.ndarray`) containing samples for different features
    featurenames : :obj:`tuple` or :obj:`list`, optional
        Names to give to features in output dataframe
        amples for different features

    Returns
    -------
    features : :obj:`pd.DataFrame`
        Covariance matrix

    """
    # put features in dataframe
    features = pd.DataFrame(np.vstack(features).T,
                            columns=featurenames if not None
                            else range(len(features)))
    # drop samples with nan
    features.dropna(axis=0, inplace=True)
    return features.cov()


def variogram(x, y, z, minlag=0, maxlag=1., nlags=100, lagtol=None, model=None,
              modelparams=None, plotflag=False, figsize=(12, 10), verb=False):
    """Variogram estimation

    Routine used to visualize useful information to identify the best-choice
    variogram for a set of (x,y,z) points for a z=f(x,y) function.

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Values along x axis
    y : :obj:`np.ndarray`
        Values along y axis
    z : :obj:`np.ndarray`
        Values along z axis (to be used at values of the z=f(x, y) function)
    minlag : :obj:`float`, optional
        Minimum lag to fit variogram
    maxlag : :obj:`float`, optional
        Maximum lag to fit variogram
    nlags : :obj:`float`, optional
        Number of lags to fit variogram
    lagtol : :obj:`float`, optional
        Lag tolerance to fit variogram
    model : :obj:`str`, optional
        Variogram model (``spherical``, ``exponential``, ``gaussian`` or
        function that takes lags, range and sill and returns the semivariogram).
        If ``None``, no attempt is made at fitting the variogram
    modelparams : :obj:`list`, optional
        Parameters for variogram model model (``spherical``, ``exponential``,
        ``gaussian``). If ``None``, the parameters are estimated automatically
        from the sample variogram
    plotflag : :obj:`bool`, optional
        Quickplot
    figsize : :obj:`tuple`, optional
        Size of figure
    verb : :obj:`int`, optional
        Verbosity

    Returns
    -------
    lags : :obj:`np.ndarray`
        Lags axis
    var : :obj:`np.ndarray`
        Experimental variogram
    modelparams : :obj:`list`
        Range and sill. If `modelparams` are provided, they will be simply
        returned untouched. If they are not provided, these are estimated
        from the sample variogram
    varmod : :obj:`geostatsmodels.model.semivariance`
        Semivariance model
    covmod : :obj:`geostatsmodels.model.covariance`
        Covariance model
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    axs : :obj:`plt.axes`
        Axes handles

    """
    if plotflag:
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Variogram fitting',
                     fontsize=18, fontweight='bold', y=1.04)
        ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    else:
        fig = ax1 = ax2 = ax3 = None

    # sample variogram
    xyz = np.vstack([x, y, z]).T
    lags = np.linspace(minlag, maxlag, nlags)
    if lagtol is None:
        lagtol = lags[1] - lags[0]
    try:
        lags, var = geostatsvariograms.semivariogram(xyz, lags, lagtol)
    except ValueError:
        raise ValueError('Provide larger lagtol')

    # parametric variogram and covariance
    sill = np.var(xyz[:, 2])

    if model == 'spherical':
        model = geostatsmodel.spherical
    elif model == 'exponential':
        model = geostatsmodel.exponential
    elif model == 'gaussian':
        model = geostatsmodel.gaussian
    if not callable(model):
        raise ValueError('Provide one of the avaible options for model...')

    if model and not modelparams:
        a = _estimate_variogram_range(lags, var, sill, model, verb=verb)
        modelparams = [a, sill]
    if model:
        # define covariance based on model and parameters
        varmod = geostatsmodel.semivariance(model, modelparams)
        covmod = geostatsmodel.covariance(model, modelparams)

    if plotflag:
        # histogram
        mu, std = norm.fit(z)
        ax1.hist(z, bins=15, density=True, alpha=0.6, color='#ffb3b3')
        xmin, xmax = ax1.get_xlim()
        p = norm.pdf(np.linspace(xmin, xmax, 100), mu, std)
        ax1.plot(np.linspace(xmin, xmax, 100), p, 'k', linewidth=2)
        ax1.set_title("Gaussian fit: mu = %.2f,  std = %.2f" % (mu, std))
        ax1.set_ylabel('Density')

        # quantile-quantile
        (osm, osr), (slope, intercept, r) = sp_stats.probplot(z, dist="norm",
                                                              fit=True)
        ax2.plot(osm, slope * osm + intercept, 'k', osm, osr,
                 '#ffb3b3', marker='.')
        ax2.set_xlabel('Standard Normal Quantiles')
        ax2.set_ylabel('Sorted Values')
        ax2.set_title('Normal quantile-quantile plot')
        ax3.plot(lags, var, 'ko-')
        if model:
            ax3.plot(lags, varmod(lags), '#ffb3b3', lw=5)
        ax3.set_xlabel('Lag Distance')
        ax3.set_title('Semivariogram')
        ax3.text(minlag, sill,
                 'range={}, sill={}'.format(np.round(modelparams[0], decimals=2),
                                            np.round(modelparams[1], decimals=2)),
                 fontsize=12, bbox=dict(boxstyle="round", fc=(1., 1., 1.)))
        ax3.axhline(sill, ls='--', color='k')
        plt.tight_layout()

    return lags, var, modelparams, varmod, covmod, fig, (ax1, ax2, ax3)


def correlation_parametric(nh, dh, model='exponential', range=0.1, var=None,
                           plotflag=False):
    """Parametric correlation matrix

    Create correlation matrix from a parametric definition of the correlation
    model.

    Parameters
    ----------
    nh : :obj:`int`
        Number of lags of correlation function
    dh : :obj:`int`
        Sampling of lag axis
    model : :obj:`str`, optional
        Correlation model (``spherical``, ``exponential``, ``gaussian`` or
        geostatsmodels function that takes lags and returns the covariance
        function).
    var : :obj:`float`, optional
        Variance (to be provided if model is a geostatsmodels function)
    range : :obj:`float`, optional
        Correlation range.
    plotflag : :obj:`bool`, optional
            Quickplot

    Returns
    -------
    C : :obj:`np.ndarray`
        Correlation matrix

    """
    # Lag axis
    h = np.arange(nh) * dh

    # Correlation function
    if model == 'exponential':
        corrz = np.exp(-3 * h / range)
    elif model == 'gaussian':
        corrz = np.exp(-3 * (h) ** 2 / range ** 2)
    elif model == 'spherical':
        corrz = (1 - 1.5 * (h / range) + 0.5 * (h / range) ** 3)
        corrz[h > range] = 0
    else:
        corrz = model(h) / var
    # Correlation matrix
    C = sp_lin.toeplitz(corrz)

    if plotflag:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        ax1.plot(corrz)
        ax1.set_title('Correlation function')
        ax2.imshow(C, interpolation='nearest')
        ax2.set_title('Correlation matrix')
        ax2.axis('tight')
    return C


def correlation_distances(hh, model='exponential', range=0.1, var=None,
                          plotflag=False):
    r"""Parametric correlation matrix

    Create correlation matrix based on distances:

    .. math::
        C(hh) = [c(hh_{0,0}, c(hh_{0,1}, ..., c(hh_{0,N};
                 c(hh_{1,0}, c(hh_{1,1}, ..., c(hh_{1,N};
                 ...
                 c(hh_{N,0}, c(hh_{N,1}, ..., c(hh_{N,N}]

    where ``hh`` can be created using np.meshgrid and ``c`` is the parametric
    correlation function defined by ``model``.



    Parameters
    ----------
    hh : :obj:`int`
        Matrix containing distances
    model : :obj:`str`, optional
        Correlation model (``spherical``, ``exponential``, ``gaussian`` or
        geostatsmodels function that takes lags and returns the covariance
        function).
    var : :obj:`float`, optional
        Variance (to be provided if model is a geostatsmodels function)
    range : :obj:`float`, optional
        Correlation range.
    plotflag : :obj:`bool`, optional
            Quickplot

    Returns
    -------
    C : :obj:`np.ndarray`
        Correlation matrix

    """
    # Correlation function
    if model == 'exponential':
        C = np.exp(-3 * hh / range)
    elif model == 'gaussian':
        C = np.exp(-3 * (hh) ** 2 / range ** 2)
    elif model == 'spherical':
        C = (1 - 1.5 * (hh / range) + 0.5 * (hh  / range) ** 3)
        C[hh > range] = 0
    else:
        C = (model(hh.ravel()) / var).reshape(hh.shape)

    if plotflag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
        im = ax1.imshow(C, interpolation='nearest')
        ax1.set_title('Correlation matrix')
        ax1.axis('tight')
        plt.colorbar(im, ax=ax1)
    return C


def covariance_parametric(Corr, std=100, plotflag=False):
    """Parametric covariance matrix

    Create covariance matrix given matrix of standard deviations and
    correlation matrix as in http://blogs.sas.com/content/iml/2010/12/10/
    converting-between-correlation-and-covariance-matrices.html.

    Parameters
    ----------
    Corr : :obj:`np.ndarray`
        Correlation matrix
    std : :obj:`np.ndarray` or :obj:`int`
        Standard deviation of each sample [Nsamples] or single value
    plotflag : :obj:`bool`, optional
            Quickplot

    Returns
    -------
    Cov : :obj:`np.ndarray`
        Covariance matrix

    """
    if isinstance(std, (int, float)):
        Std = np.diag(std * np.ones(Corr.shape[0]))
    else:
        Std = np.diag(std)
    Cov = np.dot(Std, np.dot(Corr, Std))

    if plotflag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
        im = ax1.imshow(Cov, interpolation='nearest')
        ax1.axis('tight')
        ax1.set_title('Covariance matrix')
        plt.colorbar(im, ax=ax1)
    return Cov


def drawsamples(mean, cov, nreals=1):
    r"""Draw samples given mean and covariance matrix

    Parameters
    ----------
    mean : :obj:`np.ndarray` or :obj:`float`
        Mean (of size :math:`Nsamples` or single value will be replicated)
    cov : :obj:`np.ndarray` or :obj:`float`
        Covariance matrix of size :math:`Nsamples \times Nsamples`
    nreals : :obj:`int`, optional
        Number of realizations

    Returns
    -------
    real : :obj:`np.ndarray`
        Realizations of size :math:`Nsamples \times nreals`

    """
    if isinstance(mean, (int, float)):
        mean = mean * np.ones(cov.shape[0])
    real = np.random.multivariate_normal(mean, cov, nreals).T
    return real


def cross_validation_regression(x, y, nreals=10):
    """Estimate an ensemble of regression parameters based on statistics
    collected from a leave-one-out approach

    Parameters
    ----------
    x : :obj:`np.ndarray`
        features for :class:`sklearn.linear_model.LinearRegression` regressor
    y : :obj:`np.ndarray`
        target for :class:`sklearn.linear_model.LinearRegression` regressor
    nreals : :obj:`int`, optional
        Number of realizations

    Returns
    -------
    reg_preds : :obj:`np.ndarray`
        Ensemble of regression parameters of size
        :math:`[nreals \times 2]`

    """
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    cv = cross_validate(reg, x.reshape(-1, 1),
                        y.reshape(-1, 1), cv=LeaveOneOut(),
                        return_estimator=True)
    reg_preds = np.array(
        [[est.intercept_[0], est.coef_[0][0]] for est in cv['estimator']])
    reg_mean = np.mean(reg_preds, axis=0)
    reg_cov = np.cov(reg_preds.T)
    reg_preds = \
        (reg_mean[:, np.newaxis] + \
        np.linalg.cholesky(reg_cov) @ np.random.normal(0, 1, (2, nreals))).T
    return reg_preds


def maximum_likelihood_regression(x, y, C, nreals=10):
    """Maximum likelihood linear regression with noisy data points.

    Estimate the mean and covariance of the Maximum likelihood linear regression
    problem. The forward problem is here defined as:

    .. math::
        \mathbf{y} = \mathbf{H} \mathbf{x} + \mathbf{e}

    where :math:`\mathbf{e} = N(\mathbf{0}, \mathbf{C})` is a normally
    distributed error and :math:`\mathbf{H}` is the linear regression operator.

    The general maximum likelihood estimator is equivalent to:

    .. math::
        \mathbf{x}_{ML} = (\mathbf{H}^T \mathbf{C}^{-1}  \mathbf{H})^{-1}
        \mathbf{H}^T \mathbf{C}^{-1} \mathbf{y}

    and

    .. math::
        \mathbf{C}_{ML} = (\mathbf{H}^T \mathbf{C}^{-1} \mathbf{H})^{-1}

    For the special case of scaled identity covariance
    (:math:`\mathbf{C} = C*\mathbf{I}`) we have:

    .. math::
        \mathbf{x}_{ML} = (\mathbf{H}^T \mathbf{H})^{-1}
        \mathbf{H}^T \mathbf{y}

    and

    .. math::
        \mathbf{C}_{ML} = C * (\mathbf{H}^T \mathbf{H})^{-1}


    Parameters
    ----------
    x : :obj:`np.ndarray`
        features for :class:`sklearn.linear_model.LinearRegression` regressor
    y : :obj:`np.ndarray`
        target for :class:`sklearn.linear_model.LinearRegression` regressor
    C : :obj:`float` or :obj:`list` or :obj:`np.ndarray`, optional
        Covariance of target. If scalar, the covariance is just an
        identity matrix scaled by C (where C is the variance, common to every
        element of the vector ``y``).
        If list, the covariance is a diagonal matrix with the elements of
        the list along the main diagonal (where each element of C is the
        variance of its corresponding element of the vector ``y``).
        If numpy array, C is the covariance matrix and must be of size
        math:`n_y \times n_y`
    nreals : :obj:`int`, optional
        Number of realizations

    Returns
    -------
    reg_pars : :obj:`np.ndarray`
        Estimated regression parameters
    Cpost : :obj:`np.ndarray`
        Posterior covariance
    reg_pars : :obj:`np.ndarray`
        Ensemble of regression parameters of size
        :math:`[nreals \times 2]`

    """
    nx = 2 if x.ndim == 1 else 1 + x.shape[1]

    if isinstance(C, list):
        C = np.diag(np.array(C))
    H = np.vstack((np.ones(len(x)), x)).T

    if isinstance(C, (int, float)):
        reg_pars = np.linalg.lstsq(H, y)[0]
        Cpost = C * np.linalg.pinv(H.T @ H)
    else:
        C1 = np.linalg.pinv(C)
        reg_pars = np.linalg.lstsq(H.T @ C1 @ H, np.dot(H.T @ C1, y))[0]
        Cpost = np.linalg.pinv(H.T @ C1 @ H)
    if nreals > 0:
        reg_reals = np.linalg.cholesky(Cpost) @ np.random.normal(0, 1, (nx, nreals))
        reg_reals = reg_pars[0] + reg_reals[0], reg_pars[1] + reg_reals[1]
    else:
        reg_reals = None
    return reg_pars, Cpost, reg_reals


def maximum_posterior_regression(x, y, C, mux, Cx, nreals=10, Cpost_eps=1e-10):
    """Maximum a posterior linear regression with uncertain data points and features

    Estimate the mean and covariance of the MAP linear regression
    problem. The forward problem is here defined as:

    .. math::
        \mathbf{y} = \mathbf{H} \mathbf{x} + \mathbf{e}

    where :math:`\mathbf{e} = N(\mathbf{0}, \mathbf{C})` is a normally
    distributed error, :math:`\mathbf{x} = N(\mu_x, \mathbf{C}_x)`
    and :math:`\mathbf{H}` is the linear regression operator.

    The general MAP estimator is equivalent to:

    .. math::
        \mathbf{x}_{MAP} = \mu_x + \mathbf{C}_x \mathbf{H}^T (\mathbf{H}
        \mathbf{C}_x \mathbf{H}^T + \mathbf{C})^{-1} (\mathbf{y} - \mathbf{H}\mu_x)

    and

    .. math::
        \mathbf{C}_{MAP} = \mathbf{C}_x - \mathbf{C}_x \mathbf{H}^T (\mathbf{H}
        \mathbf{C}_x \mathbf{H}^T + \mathbf{C})^{-1} \mathbf{H} \mathbf{C}_x

    Parameters
    ----------
    x : :obj:`np.ndarray`
        features for :class:`sklearn.linear_model.LinearRegression` regressor
    y : :obj:`np.ndarray`
        target for :class:`sklearn.linear_model.LinearRegression` regressor
    C : :obj:`float` or :obj:`list` or :obj:`np.ndarray`, optional
        Covariance of target. If scalar, the covariance is just an
        identity matrix scaled by C (where C is the variance, common to every
        element of the vector ``y``).
        If list, the covariance is a diagonal matrix with the elements of
        the list along the main diagonal (where each element of C is the
        variance of its corresponding element of the vector ``y``).
        If numpy array, C is the covariance matrix and must be of size
        math:`n_y \times n_y`
    mux : :obj:`np.ndarray`, optional
        Model (features) mean
    Cx : :obj:`np.ndarray`, optional
        Model (features) covariance
    nreals : :obj:`int`, optional
        Number of realizations
    Cpost_eps : :obj:`float`, optional
        Regularization factor to add to Cpost when applying Cholesky
        factorization

    Returns
    -------
    reg_pars : :obj:`np.ndarray`
        Estimated regression parameters
    Cpost : :obj:`np.ndarray`
        Posterior covariance
    reg_pars : :obj:`np.ndarray`
        Ensemble of regression parameters of size
        :math:`[nreals \times 2]`

    """
    nx = 2 if x.ndim == 1 else 1 + x.shape[1]

    if isinstance(C, (int, float)):
        C = C * np.eye(y.size)
    elif isinstance(C, list):
        C = np.diag(np.array(C))

    H = np.vstack((np.ones(len(x)), x)).T
    K = Cx @ H.T @ np.linalg.pinv(H @ Cx @ H.T + C)

    reg_pars = mux + K @ (y - np.dot(H, mux))
    Cpost = Cx - K @ H @ Cx

    if nreals > 0:
        reg_reals = np.linalg.cholesky(Cpost + Cpost_eps*np.eye(nx)) \
                    @ np.random.normal(0, 1, (nx, nreals))
        reg_reals = reg_pars[0] + reg_reals[0], reg_pars[1] + reg_reals[1]
    else:
        reg_reals = None
    return reg_pars, Cpost, reg_reals