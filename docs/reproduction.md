# Reproduction Guide

## 1. Environment

```bash
conda env create -f environment.yml
conda activate evo-rag
```

For pip-only environments:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e .[analysis,test]
```

## 2. Ollama

Install Ollama from the official project website and pull all models configured in `configs/default.yaml` before running the full optimization. This avoids long blocking downloads during experiment execution.

## 3. Data

Place licensed technical documents under `data/raw/` and the evaluation CSV under `data/evaluation/`. The default evaluation CSV path is:

```text
data/evaluation/evaluation_testset_50_short_query_length.csv
```

## 4. Run

Build indexes:

```bash
python -m evo_rag_hpo.index --config configs/default.yaml
```

Run optimization:

```bash
python -m evo_rag_hpo.optimize --config configs/default.yaml
```

Evaluate one candidate:

```bash
python -m evo_rag_hpo.evaluate 1 2 6 0 3 --config configs/default.yaml
```

## 5. Validate

```bash
python -m compileall src
pytest
```

