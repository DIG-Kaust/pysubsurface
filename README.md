# PySubsurface
A light-weight Python library containing a bag of useful objects to load, manipulate, and visualize subsurface data. 

[![PySubsurface-tests](https://github.com/DIG-Kaust/pysubsurface/actions/workflows/build.yaml/badge.svg)](https://github.com/DIG-Kaust/pysubsurface/actions/workflows/build.yaml)

## Objective
This library aims at creating a high level, easy to use API which spans the entire suite of data types used in 
subsurface projects (i.e., well logs and picks, seismic data, surfaces, grids etc.) and provide standardized loading, 
manipulation and visualization functionalities.

:warning: Disclaimer :warning: : this library is mostly developed to ease the use and analysis of field data within the 
Deep Imaging Group. However, we always welcome feedback and external contributions!

## Project structure
This repository is organized as follows:
* :open_file_folder: **pysubsurface**:    python library containing various classes and functions for handling and visualization of subsurface data
* :open_file_folder: **pytests**:         set of pytests for main functions of the pysubsurface library
* :open_file_folder: **testdata**:        data used in various tests and notebooks
* :open_file_folder: **docs**:            files containing sphinx documentation
* :open_file_folder: **notebooks**:       set of notebooks showcasing the functionalities of the library

## Getting started :space_invader: :robot:

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

## Contributors :baby:

* Matteo Ravasi (matteo.ravasi@kaust.edu.sa)
