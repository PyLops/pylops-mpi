PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
NUM_PROCESSES = 3

.PHONY: install dev-install install_conda dev-install_conda tests doc docupdate run_examples run_tutorials

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
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt && $(PIP) install -e .

install_conda:
	conda env create -f environment.yml && conda activate pylops_mpi && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && conda activate pylops_mpi && pip install -e .

lint:
	flake8 pylops_mpi/ tests/ examples/ tutorials/

tests:
	mpiexec -n $(NUM_PROCESSES) pytest tests/ --with-mpi

doc:
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf build &&\
	cd .. && sphinx-build -b html docs/source docs/build

docupdate:
	cd docs && make html && cd ..

servedoc:
	$(PYTHON) -m http.server --directory docs/build/

# Run examples using mpi
run_examples:
	sh mpi_examples.sh examples $(NUM_PROCESSES)

# Run tutorials using mpi
run_tutorials:
	sh mpi_examples.sh tutorials $(NUM_PROCESSES)
