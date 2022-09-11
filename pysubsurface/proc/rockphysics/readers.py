import logging
import numpy as np
import pandas as pd

from pysubsurface.utils.units import g_cm3_to_kg_m3
from pysubsurface.proc.rockphysics.fluid import Brine, Oil, Gas


def _minerals_from_csv(filename, column):
    """Read mineral compositions from file

    Parameters
    ----------
    filename : :obj:`str`
        Name of file containing mineral compositions.
        See csv file in ``testdata\RockPhysics`` for an example.
    column : :obj:`str`
        Column to use (generally refer to a field or segment if more than one
        is stored in the save .csv file)

    Returns
    -------
    minerals : :obj:`dict`
        Mineral compositions ready for rock physics work

    """
    mindf = pd.read_csv(filename)
    mindf['Mineral'] = mindf['Mineral'].fillna(method='ffill')
    mindf = mindf.set_index(['Mineral', 'Property', 'Units'], drop=True)

    min_names = list(mindf.index.get_level_values(0).unique())
    minerals = {}
    for min_name in min_names:
        minerals[min_name] = {'k': float(mindf.loc[min_name][column].loc['Bulk Modulus'].values[0])*1e9,
                              'rho': g_cm3_to_kg_m3(float(mindf.loc[min_name][column].loc['Density'].values[0]))}

    return minerals


def _fluids_from_csv(filenamefluid, filenamepressure, column):
    """Read fluid compositions from file

    Parameters
    ----------
    filenamefluid : :obj:`str`
        Name of file containing fluid compositions.
        See csv file in ``testdata\RockPhysics`` for an example.
    filenamepressure : :obj:`str`
        Name of file containing pressure and temperature information.
        See csv file in ``testdata\RockPhysics`` for an example.
    column : :obj:`str`
        Column to use (generally refer to a field or segment if more than one
        is stored in the save .csv file)

    Returns
    -------
    fluids : :obj:`dict`
        Fluid compositions ready for rock physics work

    """
    fldf = pd.read_csv(filenamefluid)
    fldf['Fluid'] = fldf['Fluid'].fillna(method='ffill')
    fldf = fldf.set_index(['Fluid', 'Property', 'Units'], drop=True)

    prdf = pd.read_csv(filenamepressure)
    prdf = prdf.set_index(['Property', 'Units'], drop=True)
    
    temp = prdf.loc['Temperature @ res depth'][column].values[0]
    pres = prdf.loc['Reservoir Pressure'][column].values[0]
    sali = float(fldf.loc['Brine'][column].loc['Salinity'].values[0])*1e3
    oilgrav = float(fldf.loc['Oil'][column].loc['Oil API'].values[0])
    gasgrav = float(fldf.loc['Gas'][column].loc['Gas Gravity'].values[0])
    gor = float(fldf.loc['Gas'][column].loc['GOR'].values[0])

    fluids = {'Brine': Brine(temp, pres, sali),
              'Oil': Oil(temp, pres, oilgrav, gasgrav, gor),
              'Gas': Gas(temp, pres, gasgrav)}
    return fluids


def rockphysics_from_csv(filenamemin, filenamefluid, filenamepressure, column):
    """rock physics input parameters from csv files

    Read rock physics input parameters from files and return them in format
    suitable for rock physics analysis in pysubsurface

    Parameters
    ----------
    filenamemin : :obj:`str`
        Name of file containing mineral compositions.
        See csv file in ``testdata\RockPhysics`` for an example.
    filenamefluid : :obj:`str`
        Name of file containing fluid compositions.
        See csv file in ``testdata\RockPhysics`` for an example.
    filenamepressure : :obj:`str`
        Name of file containing pressure and temperature information.
        See csv file in ``testdata\RockPhysics`` for an example.
    column : :obj:`str`
        Column to use (generally refer to a field or segment if more than one
        is stored in the save .csv file)

    Returns
    -------
    minerals : :obj:`dict`
        Mineral compositions ready for rock physics work
    fluids : :obj:`dict`
        Fluid compositions ready for rock physics work

    """
    minerals = _minerals_from_csv(filenamemin, column)
    fluids = _fluids_from_csv(filenamefluid, filenamepressure, column)
    return minerals, fluids
