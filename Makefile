.PHONY: install test evaluate audit

install:
	python -m pip install -e '.[dev]'

test:
	pytest

evaluate:
	python -m margins_maps.workflow

audit:
	pytest
	python -m compileall -q src experiments tests
