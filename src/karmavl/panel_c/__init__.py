"""Frozen, outcome-gated implementation for Panel C.

The package separates fold-local training and control selection from target
scoring.  Importing it never opens an answer, correctness, or outcome ledger.
"""

from .costs import deployment_utility, total_gpu_seconds
from .nested import select_fold_b_star, select_fold_margin, verify_nested_loso

__all__ = [
    "deployment_utility",
    "select_fold_b_star",
    "select_fold_margin",
    "total_gpu_seconds",
    "verify_nested_loso",
]

