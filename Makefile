PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
NUM_PROCESSES = 10

.PHONY: install dev-install lint tests

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
	flake8 pylops_mpi/ tests/ examples/

tests:
	mpiexec -n $(NUM_PROCESSES) pytest tests/ --with-mpi
