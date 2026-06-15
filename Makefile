.PHONY: install test lint fmt compile check index optimize evaluate

install:
	python -m pip install -e .[test]

test:
	python -m pytest

lint:
	ruff check .

fmt:
	ruff format .

compile:
	python -m compileall src

check: lint compile test

index:
	python -m evo_rag_hpo.index --config configs/default.yaml

optimize:
	python -m evo_rag_hpo.optimize --config configs/default.yaml

evaluate:
	python -m evo_rag_hpo.evaluate 1 2 6 0 3 --config configs/default.yaml
