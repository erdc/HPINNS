"""! @file swe_pinn/cli.py
@brief Thin CLI wrapper around the solver.
"""
from __future__ import annotations
import argparse
from user_input import USER_PINN_INPUTS
from swe_pinn.core.solver import run_solver

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SWE PINN, or one parallel search worker.")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run Optuna trials only. Requires --worker-id. Does not train the final model.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Parallel worker index. Sampler seed is 42 + this id.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Train the best completed trial already stored in the study. Adds no trials.",
    )
    args = parser.parse_args()
    if args.finalize and args.search_only:
        parser.error("Use only one of --finalize and --search-only.")
    if args.search_only and args.worker_id is None:
        parser.error("--search-only requires --worker-id.")
    run_solver(
        USER_PINN_INPUTS,
        worker_id=args.worker_id if args.search_only else None,
        finalize_only=args.finalize,
    )
