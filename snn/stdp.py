"""
Spike-Timing-Dependent Plasticity (STDP) Learning Rule
=======================================================
STDP updates synaptic weights based on the relative timing of
pre- and post-synaptic spikes.

In our flashcard model:
    Pre-synaptic spike  = the card being shown (presentation event)
    Post-synaptic spike = the learner successfully recalling the answer

    Causal pairing (pre → post, short delay) = potentiation (LTP): weight ↑
    Anti-causal (post → pre, or long delay)  = depression (LTD):   weight ↓

Rating scale (1–4, like Anki):
    4 — Easy:   Instant recall. Strong potentiation.
    3 — Good:   Normal recall with brief effort. Moderate potentiation.
    2 — Hard:   Recalled but with difficulty. Weak potentiation.
    1 — Again:  Failed recall. Depression.

The STDP window function used here is the standard asymmetric exponential:

    ΔW = A_plus  * exp(-|Δt| / tau_plus)   if Δt > 0  (LTP)
    ΔW = -A_minus * exp(-|Δt| / tau_minus)  if Δt ≤ 0  (LTD)

where Δt is the simulated spike-timing difference derived from rating.
"""

import math
from dataclasses import dataclass


@dataclass
class STDPConfig:
    """Hyperparameters for the STDP rule."""
    A_plus: float = 0.15    # Max potentiation amplitude
    A_minus: float = 0.10   # Max depression amplitude
    tau_plus: float = 1.0   # LTP time constant (in rating-units)
    tau_minus: float = 1.0  # LTD time constant (in rating-units)
    w_min: float = 0.0      # Minimum weight
    w_max: float = 1.0      # Maximum weight


# Rating → simulated Δt (spike timing difference)
# Positive Δt = pre before post = causal = LTP
# Negative Δt = failed or anti-causal = LTD
RATING_TO_DT = {
    4: +0.2,   # Easy:  very short positive Δt → strong potentiation
    3: +0.6,   # Good:  moderate positive Δt → normal potentiation
    2: +1.4,   # Hard:  long positive Δt → weak potentiation
    1: -0.5,   # Again: anti-causal → depression
}


def stdp_delta_w(rating: int, config: STDPConfig | None = None) -> float:
    """
    Compute the synaptic weight change ΔW for a given recall rating.

    Args:
        rating: Integer 1–4 representing recall quality.
        config: STDP hyperparameters. Uses defaults if None.

    Returns:
        ΔW: The weight change to apply (can be positive or negative).
    """
    if config is None:
        config = STDPConfig()

    if rating not in RATING_TO_DT:
        raise ValueError(f"Rating must be 1–4, got {rating}")

    dt = RATING_TO_DT[rating]

    if dt > 0:
        # Long-Term Potentiation (LTP)
        dw = config.A_plus * math.exp(-dt / config.tau_plus)
    else:
        # Long-Term Depression (LTD)
        dw = -config.A_minus * math.exp(abs(dt) / config.tau_minus)

    return dw


def apply_stdp(w_current: float, rating: int, config: STDPConfig | None = None) -> float:
    """
    Apply STDP update and return the new clamped weight.

    Uses a soft weight-dependence to prevent saturation:
        LTP  scales by (w_max - w)  — harder to potentiate already-strong synapses
        LTD  scales by (w - w_min)  — harder to depress already-weak synapses

    Args:
        w_current: Current synaptic weight.
        rating:    Recall rating (1–4).
        config:    STDP hyperparameters.

    Returns:
        New synaptic weight, clamped to [w_min, w_max].
    """
    if config is None:
        config = STDPConfig()

    dw = stdp_delta_w(rating, config)

    if dw > 0:
        # Weight-dependent potentiation
        dw_effective = dw * (config.w_max - w_current)
    else:
        # Weight-dependent depression
        dw_effective = dw * (w_current - config.w_min)

    new_w = w_current + dw_effective
    return max(config.w_min, min(config.w_max, new_w))


def rating_label(rating: int) -> str:
    """Human-readable label for a rating."""
    return {4: "Easy", 3: "Good", 2: "Hard", 1: "Again"}.get(rating, "Unknown")


def explain_update(w_before: float, w_after: float, rating: int) -> str:
    """Generate a human-readable explanation of the STDP update."""
    delta = w_after - w_before
    sign = "+" if delta >= 0 else ""
    next_type = "LTP (potentiation)" if delta >= 0 else "LTD (depression)"
    return (
        f"Rating: {rating_label(rating)} ({rating}/4) | "
        f"{next_type} | "
        f"Δw = {sign}{delta:.4f} | "
        f"Weight: {w_before:.4f} → {w_after:.4f}"
    )
