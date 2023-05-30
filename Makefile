PIP := "D:\main_pylops_mpi\pylops-mpi\venv\Scripts\pip.exe"
PYTHON := "D:\main_pylops_mpi\pylops-mpi\venv\Scripts\python.exe"
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

lint:
	flake8 pylops_mpi/ tests/ examples/

tests:
	mpiexec -n $(NUM_PROCESSES) pytest tests/ --with-mpi
