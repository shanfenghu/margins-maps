.PHONY: install test check-data evaluate audit

install:
	python -m pip install -e '.[dev]'

test:
	pytest

check-data:
	PYTHONPATH=src python -m margins_maps.inputs

evaluate:
	MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=src python -m margins_maps.workflow

audit:
	black --check src tests experiments
	flake8 src tests experiments
	pytest
	python -m compileall -q src experiments tests
