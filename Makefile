UV ?= uv
PYTHON ?= $(UV) run python
RUFF_TARGETS := src tests scripts

.PHONY: uv-sync history-contract stage-web fmt fmt-check lint test e2e visual-qa serve verify

uv-sync:
	$(UV) sync --locked

history-contract:
	$(PYTHON) scripts/check_public_history.py

stage-web:
	$(PYTHON) scripts/stage_web_python.py

fmt:
	$(UV) run ruff format $(RUFF_TARGETS)

fmt-check:
	$(UV) run ruff format --check $(RUFF_TARGETS)

lint:
	$(UV) run ruff check $(RUFF_TARGETS)

test:
	$(UV) run pytest -q --ignore=tests/e2e

e2e: stage-web
	$(UV) run pytest -q tests/e2e

visual-qa:
	$(PYTHON) scripts/visual_qa.py

serve: stage-web
	cd web && python3 -m http.server 8000

verify: history-contract stage-web fmt-check lint test e2e
