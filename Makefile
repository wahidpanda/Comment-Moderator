.PHONY: install download train test eval lint typecheck

PYTHON ?= python
UV ?= uv

install:
	$(UV) pip install -e ".[dev]" 2>/dev/null || $(PYTHON) -m pip install -e ".[dev]"

download:
	$(PYTHON) -m research.download --out data/raw

train:
	$(PYTHON) -m research.train --data-dir data/raw --out models/lr/v1

train-spam:
	$(PYTHON) -m research.train_spam --out models/spam/v1

train-smoke:
	$(PYTHON) -m research.train --data-dir data/raw --out models/lr/v1 --sample 2000 --skip-test --smoke

test:
	$(PYTHON) -m pytest tests/

eval:
	$(PYTHON) -m research.eval_bundle --data-dir data/raw --out docs/eval_metrics.json
	$(PYTHON) -c "import pathlib, shutil; pathlib.Path('web/public').mkdir(parents=True, exist_ok=True); shutil.copy('docs/eval_metrics.json','web/public/eval_metrics.json')"

eval-full:
	$(PYTHON) -m research.eval --data-dir data/raw --out docs/eval_full_metrics.json

download-300k:
	$(PYTHON) -m research.download --out data/raw --force --sample 300000

api:
	$(PYTHON) -m api.main

creator-eval:
	$(PYTHON) -m research.generate_creator_eval

lint:
	ruff check engine api research tests

typecheck:
	mypy engine research