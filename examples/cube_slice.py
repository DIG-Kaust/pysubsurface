"""
Cubes and Slices
================

Two auxiliary objects used within most of the algorithms implemented in PySubsurface
are :class:`pysubsurface.objects.Slice` and  :class:`pysubsurface.objects.Cube`. Such objects
make the handling of 2- and 3-dimensional data easy by separating the actual
data from its auxiliary information (e.g., axes) which are not always needed to
carry out some of the tasks and simply replacing them by unitary, centered axes
(which turn out to be very handly in the implementation several algorithms).
"""
import numpy as np
import matplotlib.pyplot as plt
from pysubsurface.objects import Slice, Cube

plt.close('all')

###############################################################################
# We first create and display a :class:`pysubsurface.objects.Slice` object.

nx, nt = 101, 50
slice = Slice(np.random.normal(0, 1, (nx, nt)))
print(slice)


###############################################################################
# We can generate a quick plot as well as a more customized one
slice.view()
slice.view(figsize=(4, 5), cmap='gray', clip=0.4, cbar=True)

###############################################################################
# Similarly for a :class:`pysubsurface.objects.Cube` object.
ny, nx, nt = 33, 101, 50
cube = Cube(np.random.normal(0, 1, (ny, nx, nt)))
print(cube)

###############################################################################
# Again, we can generate a quick plot and a more customized one
cube.view()
cube.view(figsize=(12, 10), cmap='gray', clip=0.4, cbar=True)
