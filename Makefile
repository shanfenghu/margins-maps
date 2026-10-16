.PHONY: install test check-data evaluate verify-results reproduce audit

install:
	python -m pip install -e '.[dev]'

test:
	pytest

check-data:
	PYTHONPATH=src python -m margins_maps.inputs

evaluate:
	MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=src python -m margins_maps.workflow

verify-results:
	PYTHONPATH=src python -m margins_maps.verification

reproduce: evaluate
	$(MAKE) verify-results

audit:
	black --check src tests experiments
	flake8 src tests experiments
	pytest
	python -m compileall -q src experiments tests
	$(MAKE) verify-results
