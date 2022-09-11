import numpy as np

from pysubsurface.objects.Surface import Surface
from pysubsurface.objects.Seismic import Seismic
from pysubsurface.objects.SeismicIrregular import SeismicIrregular


def _chi_rotation_arrays(intercept, gradient, chi):
    """Chi rotation of np.ndarrays

    """
    chi = np.deg2rad(chi)
    attr = intercept * np.cos(chi) + gradient * np.sin(chi)
    return attr


def _chi_rotation_surface(intercept, gradient, chi):
    """Chi rotation of Surface objects
    """
    attr = intercept.copy()
    print(attr._regsurface)
    if attr._regsurface:
        attr.data = _chi_rotation_arrays(intercept.data, gradient.data, chi)
    else:
        attr.data.data[:] = _chi_rotation_arrays(intercept.data.data,
                                                 gradient.data.data, chi)
    return attr


def _chi_rotation_seismic(intercept, gradient, chi):
    """Chi rotation of Seismic or SeismicIrregular objects
    """
    attr = intercept.copy()
    attr.data = _chi_rotation_arrays(intercept.data,
                             gradient.data, chi)
    return attr


def chi_rotation(intercept, gradient, chi):
    """Chi rotation

    Apply chi rotation given intercept :math:`I`, gradient :math:`G`,
    and angle :math:`\chi`:

     .. math::
        Attr(\chi)= I*cos(\chi) + G*sin(\chi)

    Note that is operation is peformed element wise.

    Parameters
    ----------
    intercept : :obj:`pysubsurface.objects.Surface` or :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Intercept
    gradient : :obj:`pysubsurface.objects.Surface` or :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Gradient
    chi : :obj:`float`
         Rotation angle in degrees

    Returns
    -------
    chiattr :  :obj:`pysubsurface.objects.Surface` or :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
        Rotated attribute

    """
    if isinstance(intercept, Surface):
        attr = _chi_rotation_surface(intercept, gradient, chi)
    elif isinstance(intercept, Seismic) or isinstance(intercept, SeismicIrregular):
        attr = _chi_rotation_seismic(intercept, gradient, chi)
    return attr
