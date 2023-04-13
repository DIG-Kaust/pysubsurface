PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)

.PHONY: install install_dev test doc docupdate

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

install:
	conda env create -f environment.yml && conda activate pysubsurface && pip install .

install_dev:
	conda env create -f environment.yml && conda activate pysubsurface && pip install -e .

tests:
	make pythoncheck
	$(PYTHON) setup.py test

doc:
	cd docs  && rm -rf source/api/generated && rm -rf source/examples &&\
	rm -rf source/tutorials && rm -rf build && make html && cd ..

docupdate:
	cd docs && make html && cd ..

servedoc:
	$(PYTHON) -m http.server --directory docs/build/html/
