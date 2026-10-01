# Shortcuts. `make all` re-runs the whole project from scratch.
PY ?= python

.PHONY: setup experiments evaluate predict test all

setup:
	pip install -r requirements.txt

experiments:
	PYTHON=$(PY) ./run_experiments.sh

evaluate:
	$(PY) -m src.evaluate

predict:
	$(PY) -m src.predict samples/*.png

test:
	$(PY) -m pytest -q

all: experiments evaluate test
