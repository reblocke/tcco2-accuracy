"""Workflow compatibility wrapper for manuscript-ready reporting outputs."""

from __future__ import annotations

from ..reporting.manuscript import (
    ManuscriptParametersResult,
    ManuscriptWorkflowResult,
    run_manuscript_outputs,
    run_manuscript_parameters,
)

__all__ = [
    "ManuscriptParametersResult",
    "ManuscriptWorkflowResult",
    "run_manuscript_outputs",
    "run_manuscript_parameters",
]
