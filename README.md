# PySubsurface
A universal Python library containing a bag of useful object to load, manipulate, and visualize subsurface data

## Objective
This library aims at creating a high level, easy to use API which spans the entire suite of data types used in subsurface projects (i.e., well logs and picks, seismic data,
  surfaces, grids etc.) and provide standardized loading, manipulation and visualization functionalities.

## Project structure
This repository is organized as follows:
* **pysubsurface**:    python library containing various classes and functions for handling and visualization of PETEC data
* **pytests**:         set of pytests for main functions of the pysubsurface library
* **testdata**:        data used in various tests and notebooks
* **notebooks**:       set of notebooks showcasing the functionalities of the library

## Getting started

Clone the repository:
```
git clone https://github.com/DIG-Kaust/pysubsurface.git
```

Create an environment using the ``environment.yml`` file: 
```
conda env create -f environment. yml
```

Install pysuburface:

```
pip install -e .
```

To ensure that everything has been setup correctly, run tests: 
```
make test
```

## Contributors
* Matteo Ravasi (matteo.ravasi@kaust.edu.sa)
