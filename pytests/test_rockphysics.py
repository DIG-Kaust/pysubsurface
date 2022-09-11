import pytest
import numpy as np

import pysubsurface.proc.rockphysics.elastic as el
import pysubsurface.proc.rockphysics.bounds as bd
import pysubsurface.proc.rockphysics.fluid as fd
import pysubsurface.proc.rockphysics.solid as sd

from pysubsurface.utils.units import *
from pysubsurface.proc.rockphysics.gassmann import Gassmann
from pysubsurface.proc.rockphysics.readers import rockphysics_from_csv


def test_elastic_properties():
    """Compute elastic properties from VP, VP, RHO triplet and go back to
    original triplet
    """
    vp = 2000
    vs = 1200
    rho = 1000

    lam = el._compute_lambda(vp, vs, rho)
    k = el._compute_bulk(vp, vs, rho)
    mu = el._compute_mu(vs, rho)
    vp = el._compute_vp(k, mu, rho)
    vs = el._compute_vs(mu, rho)

    assert lam == 1120000000
    assert k == 2080000000
    assert mu == 1440000000
    assert vp == 2000
    assert vs == 1200


def test_voight_reuss_bounds():
    """Apply Voight and Reuss bounds and check that results are as
    expected for extreme cases
    """
    # single consitutent
    k = [1e9]
    f = [1.]

    kvoight = bd.voigt_bound(f, k)
    kreuss = bd.reuss_bound(f, k)

    assert float(kvoight) == k[0]
    assert np.isclose(kreuss, k[0]) # allow for small error in division

    # single consitutent (one set to 0)
    k = [1e9, 1e6]
    f = [1., 0.]

    kvoight = bd.voigt_bound(f, k)
    kreuss = bd.reuss_bound(f, k)
    assert float(kvoight) == k[0]
    assert np.isclose(kreuss, k[0])  # allow for small error in division

    # k and f are the same, same value as input if
    k = [1e9, 1e9]
    f = [0.5, 0.5]

    kvoight = bd.voigt_bound(f, k)
    kreuss = bd.reuss_bound(f, k)
    assert float(kvoight) == k[0]
    assert np.isclose(kreuss, k[0]) # allow for small error in division


def test_hashin_shtrikman_bounds():
    """Apply Hashin-Shtrikman bounds and check that results are as
    expected for extreme cases
    """
    # single consitutent
    k = [1e9]
    mu = [1e3]
    f = [1.]

    klower, kupper = bd.hashin_shtrikman(f, k, mu)
    assert float(klower) == k[0]
    assert float(kupper) == k[0]

    # single consitutent (one set to 0)
    k = [1e9, 1e6]
    mu = [1e3, 0]
    f = [1., 0.]

    klower, kupper = bd.hashin_shtrikman(f, k, mu)
    assert np.isclose(klower, k[0]) # allow for small error in division
    assert float(kupper) == k[0]

    # k and f are the same, same value as input if
    k = [1e9, 1e9]
    mu = [1e3, 1e3]
    f = [0.5, 0.5]

    klower, kupper = bd.hashin_shtrikman(f, k, mu)
    assert np.isclose(klower, k[0])  # allow for small error in division
    assert float(kupper) == k[0]


def test_fluids():
    """Apply Fluids property computation and check results
    """
    oilgrav = 42
    gasgrav = 0.9
    pres = psi_to_Pa(3200) * 1e-3
    temp = 150
    sal = 3800
    gor = 160

    bri = fd.Brine(temp, pres, sal)
    oil = fd.Oil(temp, pres, oilgrav, gasgrav, gor)
    gas = fd.Gas(temp, pres, gasgrav)

    assert bri.k == 2176049652.312428
    assert bri.rho == 933.1956506670521 
    assert oil.k == 219913468.1481167
    assert oil.rho == 552.6214860297605
    assert gas.k == 47098007.21574441
    assert gas.rho == 181.89858127346812

    assert bri.vs == 0
    assert oil.vs == 0
    assert gas.vs == 0


    # mixing fluids
    fluid = fd.Fluid({'water': (bri, 0.4),
                      'oil': (oil, 0.6)})
    assert fluid.k == 343387113.58576465
    assert fluid.rho == 704.8511518846772


def test_dry():
    pass


def test_gassmann():
    """Apply Gassmann fluid substitution
    """
    mat = sd.Matrix(
        {'sand': {'k': 36.6e9, 'rho': g_cm3_to_kg_m3(2.65), 'frac': 0.86},
         'shale': {'k': 20.9e9, 'rho': g_cm3_to_kg_m3(2.58), 'frac': 0.14}})

    oilgrav = 42
    gasgrav = 0.9
    pres = psi_to_Pa(3200) * 1e-3
    temp = 150
    sal = 3800
    gor = 160
    wat = fd.Brine(temp, pres, sal)
    oil = fd.Oil(temp, pres, oilgrav, gasgrav, gor)
    fluid = fd.Fluid({'water': (wat, 0.4),
                   'oil': (oil, 0.6)})
    fluid1 = fd.Fluid({'water': (wat, 1.),
                    'oil': (oil, 0.)})

    vp = ft_to_m(11000)
    vs = ft_to_m(6500)
    rho = g_cm3_to_kg_m3(2.2)
    poro = 0.2
    rock = sd.Rock(vp, vs, rho, mat, fluid, poro=poro)

    fluidsub = Gassmann(rock, fluid1)
    assert fluidsub.medium1.vp == 3481.7721555590456
    assert fluidsub.medium1.vs == 1939.4296210608754
    assert fluidsub.medium1.rho == 2298.7991301334105


def test_reader():
    """Use reader routines to read mineral and fluid inputs from csv files and
    check expected results
    """
    minerals, fluids = \
        rockphysics_from_csv(
            'testdata/RockPhysics/minerals.csv',
            'testdata/RockPhysics/fluids.csv',
            'testdata/RockPhysics/pressures.csv',
            'Samplefield')

    assert minerals['Shale']['k'] == 22000000000.0
    assert minerals['Shale']['rho'] == 2600.0
    assert minerals['Sandstone']['k'] == 36800000000.0
    assert minerals['Sandstone']['rho'] == 2650.0

    assert fluids['Brine'].vp == 1627.9143916749163
    assert fluids['Brine'].rho == 992.0330767785131
    assert fluids['Gas'].vp == 571.003756977636
    assert fluids['Gas'].rho == 229.7041198787938
    assert fluids['Oil'].vp == 817.5092033862684
    assert fluids['Oil'].rho == 602.892205180804
