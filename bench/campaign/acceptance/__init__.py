"""Scientific acceptance policies for article replications."""

from .diagnostics import evaluate_diagnostic_runs
from .historical import evaluate_historical_runs

__all__ = ["evaluate_diagnostic_runs", "evaluate_historical_runs"]
