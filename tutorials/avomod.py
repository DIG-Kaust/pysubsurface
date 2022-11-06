r"""
AVA modelling
=============

This example shows how to simply create amplitude-variation-with-angle (AVA)
curves given an interface with two infinite half-space layers
"""
import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.visual.combinedviews import ava_modelling
from pysubsurface.proc.uncertainty.uncertainty import ava_modelling_sensitivity


plt.close('all')

###############################################################################
# Let's start defining a single medium.

vp1 = 1200
vs1 = 600 + vp1/2
rho1 = 1000 + vp1

vp2 = 1500
vs2 = 600 + vp2/2
rho2 = 1000 + vp2

theta = np.arange(0, 50)

###############################################################################
# We can now use then convinience function :func:`pysubsurface.visual.combinedviews.ava_modelling`
# to compute AVA curves with different methods and display them.
ava_modelling(vp1, vs1, rho1, vp2, vs2, rho2, theta,
              ('zoeppritz', 'akirichards', 'fatti'), ('r', 'b', 'g'),
              figsize=(8, 7))



###############################################################################
# Even better, we can feed it with several models and study the variability of
# our AVA response and intercept and gradient

# create baseline model
vp = np.array([2400, 2500, 2450, 2600])
vs = np.array([1000, 1200, 1300, 1600])
rho = np.array([1400, 1500, 1450, 1600])
theta = np.arange(0, 50)
nreals = 100

# create realizations
vp_reals = np.zeros((nreals, len(vp)))
vs_reals = np.zeros_like(vp_reals)
rho_reals = np.zeros_like(vp_reals)
vp_reals[0] = vp
vs_reals[0] = vs
rho_reals[0] = rho

for ireal in range(1, nreals):
    vp_reals[ireal] = vp+np.random.normal(0,100)
    vs_reals[ireal] = vs+np.random.normal(0,100)
    rho_reals[ireal] = rho+np.random.normal(0,100)

# sphinx_gallery_thumbnail_number = 2
_, _ = ava_modelling_sensitivity(vp_reals, vs_reals, rho_reals, theta,
                                 colors=['r', 'g', 'k', 'y'])
