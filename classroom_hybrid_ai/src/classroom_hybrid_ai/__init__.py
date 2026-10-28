"""Offline, auditable small-model/large-model classroom video analysis."""

__version__ = "0.1.0"

# Public offline paper-scoring entry point.  The module is dependency-light and
# does not import a model runtime until a caller explicitly loads a profile.
from .paper_evaluation import audit_splits, evaluate_research_run, validate_split_audit

__all__ = ["audit_splits", "evaluate_research_run", "validate_split_audit"]
