.. _api:


PySubsurface API
================


End user
--------

Objects
~~~~~~~

.. currentmodule:: pysubsurface.objects

.. autosummary::
   :toctree: generated/

    Project
    Ensemble
    Facies
    Fault
    Interpretation
    Intervals
    Logs
    Picks
    Polygon
    PolygonSet
    Seismic
    SeismicIrregular
    RawSeismic
    Surface
    SurfacePair
    TDcurve
    Trajectory
    Well


Data Processing (Enrichment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Seismic modelling**

.. currentmodule:: pysubsurface.proc.seismicmod

.. autosummary::
   :toctree: generated/

    poststack.zerooffset_mod
    poststack.zerooffset_wellmod
    poststack.zerooffset_geomod
    poststack.timeshift
    avo.prestack_mod
    avo.prestack_wellmod
    waveletest.statistical_wavelet
    welltie.welltie
    welltie.welltie_shift_finder

.. currentmodule:: pysubsurface.proc.uncertainty.uncertainty

.. autosummary::
   :toctree: generated/

   ava_modelling_sensitivity
   welllogs_ava_sensitivity
   
**Rock physics**

.. currentmodule:: pysubsurface.proc.rockphysics

.. autosummary::
   :toctree: generated/

   bounds.voigt_bound
   bounds.reuss_bound
   bounds.voigt_reuss_hill_average
   bounds.hashin_shtrikman
   elastic.backus
   fluid.Brine
   fluid.Gas
   fluid.Oil
   fluid.Fluid
   solid.Matrix
   solid.Rock
   gassmann.Gassmann

**Geomodelling**

.. currentmodule:: pysubsurface.proc.geomod

.. autosummary::
   :toctree: generated/

   geomod.surface_from_wells
   geomod.create_geomodel


Visualizations
~~~~~~~~~~~~~~

.. currentmodule:: pysubsurface.visual

.. autosummary::
   :toctree: generated/

    combinedviews.ava_modelling
    combinedviews.categorical_statistics
    combinedviews.correlation_panel
    combinedviews.intervals_on_map
    combinedviews.scatter_well
    combinedviews.seismic_and_map
    combinedviews.seismic_through_wells


Developer
---------

Objects
~~~~~~~

.. currentmodule:: pysubsurface.objects

.. autosummary::
   :toctree: generated/

    Cube
    Slice


Convenience functions
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pysubsurface

.. currentmodule:: pysubsurface.utils.utils

.. autosummary::
   :toctree: generated/

    line_prepender
    findclosest
    findclosest_2d
    findindeces

Statistics
~~~~~~~~~~

.. currentmodule:: pysubsurface.utils.stats

.. autosummary::
   :toctree: generated/

    covariance
    drawsamples
    cross_validation_regression
    maximum_likelihood_regression
    maximum_posterior_regression


Wavelets
~~~~~~~~

.. currentmodule:: pysubsurface.utils.wavelets

.. autosummary::
   :toctree: generated/

    ricker
    gaussian
    cosine


Seismic modelling
~~~~~~~~~~~~~~~~~

.. currentmodule:: pysubsurface.proc.seismicmod.avo

.. autosummary::
   :toctree: generated/

    angle_reflectivity
    critical_angles
    zoeppritz
    akirichards
    akirichards_alt
    fatti
    shuey
    bortfeld