import numpy as np


def extract_surface_around_wells(wells, surface, level, interval,
                                 property='Thickness (meters)',
                                 addthickness=False, extent=(0, 0)):
    """Extract surface values at well markers

    Extract property from well intervals dataframe and ``surface`` in windows
    around well locations for given interval. Note that the user is responsible
    for choosing a sensible ``surface`` which matches the chosen ``property``.

    Parameters
    ----------
    wells : :obj:`dict`
        Suite of :obj:`pysubsurface.objects.Well` objects
    surface : :obj:`pysubsurface.objects.Surface` objects
        Surface to display
    level : :obj:`int`
        Interval level to display (used in case the same ``interval`` is
        present in different levels)
    interval : :obj:`str`
        Name of interval to display
    property : :obj:`tuple`, optional
        Property to visualize (must be present in interval dataframe)
    addthickness : :obj:`bool`, optional
        Find properties in the middle of an interval by adding thickness
        (``True``) or  not (``False``)
    extent : :obj:`tuple`, optional
        half-size of extraction window in x and y directions


    Returns
    -------
    props : :obj:`np.ndarray`
       Property from well intervals dataframe
    props_surface : :obj:`np.ndarray`
       Averaged values from surface

    """
    wellnames = list(wells.keys())

    # extract formation property and its location in x-y coordinates
    xcoords = np.full(len(wellnames), np.nan)
    ycoords = np.full(len(wellnames), np.nan)
    props = np.full(len(wellnames), np.nan)
    for iwell, wellname in enumerate(wellnames):
        if wells[wellname].intervals is not None:
            xcoord, ycoord, prop = \
                wells[wellname].extrac_prop_in_interval(interval, level,
                                                        property,
                                                        addthickness=addthickness)
            if prop is not None:
                props[iwell] = prop
                xcoords[iwell] = xcoord
                ycoords[iwell] = ycoord

    # extract surface averages around those points
    props_surface = \
        surface.extract_around_points(np.vstack((xcoords, ycoords)).T,
                                      extent=extent)
    return props, props_surface
