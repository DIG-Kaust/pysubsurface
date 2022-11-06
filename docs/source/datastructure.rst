.. _datastructure:

Data structure
==============
In order to work with the PySubsurface library, one must store data in a consistent manner. More specifically,
data are stored locally in the following data tree

.. code-block:: bash

    datatree/
    ├── config.yaml
    ├── Figs
    │   └──
    ├── Polygon
    │   ├── Poly1.dat
    │   └── Poly2.dat
    ├── Seismic
    │   ├── Post
    │   │   ├── Seis1.sgy
    │   │   ├── Seis2.sgy
    │   │   └── Seis3.sgy
    │   └── Pre
    │       └── Pre.sgy
    ├── Surface
    │   ├── Horizon
    │   │   ├── Depth
    │   │   │   ├── Hor1.dat
    │   │   │   ├── Hor2.dat
    │   │   │   ├── Hor3.dat
    │   │   │   ├── Hor4.dat
    │   │   └── Time
    │   │   │   ├── Hor1.dat
    │   │   │   ├── Hor2.dat
    │   │   │   ├── Hor3.dat
    │   │   │   ├── Hor4.dat
    │   └── Maps
    ├── Fault
    │   ├── Depth
    │   │   ├── Fault1.dat
    │   │   ├── Fault2.dat
    │   └── Time
    │       ├── Fault1.dat
    │       └── Fault2.dat
    ├── RockPhysics
    │   ├── fluids.csv
    │   ├── minerals.dat
    │   └── pressures.dat
    └── Well
        ├── Checkshots
        │   ├── Check1.csv
        │   └── Check2.csv
        ├── Logs
        │   ├── Logs1.las
        │   ├── Logs2.las
        │   ├── Logs3.las
        │   └── Logs4.las
        ├── Picks
        │   ├── PICKS1.md
        │   ├── PICKS2.md
        │   └── PICKS3.md
        ├── TDCurve
        │   ├── TD1.md
        │   ├── TD2.md
        └── Trajectories
            ├── Traj1.csv
            ├── Traj2.csv
            ├── Traj3.csv
            └── Traj4.csv

Formats for various data type are based on standard formats
where a standard exists (e.g., SEG-Y for seismic, las for logs) and DSG export
file format for other data types. This is just done to get data into the Python
enviroment but it is made in a scalable fashion such that changing data formats
in a second moment would only requiring modifying the I/O routines.

Finally, one can also create ``yaml`` file containing
information for accessing data and saving figures. An example of yaml file is:

.. code-block:: yaml

    name: Projectname
    local:
      datadir: full_path_to_local_temporary_data_directory
      figdir: full_path_to_local_temporary_figure_directory