"""Pipeline-v5 local visual-MI action geometry utilities.

These utilities are intentionally separate from the v3 teacher, Qwen reward,
and planner interfaces.  They only construct and validate a local action-space
field from frozen visual tokens and simulator finite differences.
"""

from mi_reward.control.mi_action_field import (
    FixedNormalization,
    LocalActionField,
    action_gradient_hessian_from_function,
    damped_trust_region_step,
    estimate_jacobian,
    fit_fixed_normalization,
    mi_and_gradient,
    normalize_tokens,
    pullback_gradient,
    pullback_hessian,
)

__all__ = [
    "FixedNormalization",
    "LocalActionField",
    "action_gradient_hessian_from_function",
    "damped_trust_region_step",
    "estimate_jacobian",
    "fit_fixed_normalization",
    "mi_and_gradient",
    "normalize_tokens",
    "pullback_gradient",
    "pullback_hessian",
]
