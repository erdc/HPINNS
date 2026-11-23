"""! @file swe_pinn/problems/__init__.py
@brief Problem registry for built-in test cases.
"""
from __future__ import annotations
from user_input import (
    TIDAL_TEST_PARAMETERS,
    DAM_BREAK_PARAMETERS,
    TRANSCRITICAL_BUMP_PARAMETERS,
    STEADY_ROTATION_PARAMETERS,
    STEADY_VORTEX_PARAMETERS,
    STEADY_EDDIES_PARAMETERS,
)
from swe_pinn.problems.tidal_test import build_tidal_test_problem
from swe_pinn.problems.dam_break import build_dam_break_problem
from swe_pinn.problems.transcritical_bump import build_transcritical_bump_problem
from swe_pinn.problems.steady_rotation import build_steady_rotation_problem
from swe_pinn.problems.steady_vortex import build_steady_vortex_problem
from swe_pinn.problems.steady_eddies import build_steady_eddies_problem

def build_problem(cfg):
    if cfg.problem_name == "tidal_test":
        return build_tidal_test_problem(TIDAL_TEST_PARAMETERS)
    if cfg.problem_name == "dam_break":
        return build_dam_break_problem(DAM_BREAK_PARAMETERS)
    if cfg.problem_name == "transcritical_bump":
        return build_transcritical_bump_problem(TRANSCRITICAL_BUMP_PARAMETERS)
    if cfg.problem_name == "steady_rotation":
        return build_steady_rotation_problem(STEADY_ROTATION_PARAMETERS)
    if cfg.problem_name == "steady_vortex":
        return build_steady_vortex_problem(STEADY_VORTEX_PARAMETERS)
    if cfg.problem_name == "steady_eddies":
        return build_steady_eddies_problem(STEADY_EDDIES_PARAMETERS)
    raise ValueError(f"Unknown problem_name: {cfg.problem_name}")
