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
	$(UV) run pytest -q tests/e2e --junitxml=.pytest_tmp/e2e/results.xml --tracing=on --output=.pytest_tmp/e2e/browser

visual-qa:
	$(PYTHON) scripts/visual_qa.py

serve: stage-web
	cd web && python3 -m http.server 8000

verify: history-contract manuscript-check stage-web fmt-check lint conditional-reporting-check test e2e

.PHONY: future-setting-fit future-setting-recovery
future-setting-fit:
	$(UV) run --locked --offline --group fitting python scripts/future_setting.py public $(FUTURE_SETTING_ARGS)

future-setting-recovery:
	$(UV) run --locked --offline --group fitting python scripts/future_setting.py recovery $(FUTURE_SETTING_ARGS)

.PHONY: manuscript-wip
manuscript-wip:
	$(PYTHON) scripts/manuscript/wip.py $(WIP_ARGS)

.PHONY: manuscript-check
manuscript-check:
	$(PYTHON) scripts/manuscript/check_package.py

.PHONY: conditional-reporting-check
conditional-reporting-check:
	$(PYTHON) scripts/check_conditional_reporting.py
