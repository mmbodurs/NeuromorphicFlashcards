"""
SNN-Based Review Scheduler
===========================
Coordinates the LIF neuron model and STDP rule to produce a complete
flashcard scheduling system. This replaces SM-2 with a biologically
grounded alternative.

SM-2 recap (for comparison):
    - Uses an "ease factor" (EF) per card, starting at 2.5
    - Interval multiplied by EF on each correct review
    - EF drops on hard/wrong answers

Our SNN approach:
    - Each card has a LIF neuron with synaptic weight w ∈ [0, 1]
    - w is the "memory strength" — equivalent to but more principled than EF
    - STDP updates w based on recall quality (timing-dependent rule)
    - Next interval is derived from w via an exponential function
    - Forgetting is modelled explicitly via membrane potential leak
"""

import time
from typing import Tuple

from .neuron import LIFNeuron
from .stdp import apply_stdp, STDPConfig, explain_update

# Fixed lateral synapse strength W_{B→A}.
# A value of 0.3 means: an Easy recall of B (spike_strength=1.0) injects
# 0.3 units of current into A. Since the firing threshold is 1.0, A needs
# to already be at V≥0.7 for a single Easy review of B to trigger spontaneous fire.
LATERAL_WEIGHT = 0.3


class CardScheduler:
    """
    Manages the SNN state for a single flashcard and computes its
    review schedule.
    """

    def __init__(
        self,
        weight: float = 0.05,
        v_mem: float = 0.0,
        t_last: float | None = None,
        stdp_config: STDPConfig | None = None,
    ):
        self.neuron = LIFNeuron(weight=weight, v_mem=v_mem, t_last=t_last)
        self.stdp_config = stdp_config or STDPConfig()

    # ------------------------------------------------------------------
    # Core scheduling API
    # ------------------------------------------------------------------

    def review(self, rating: int, t_now: float | None = None) -> Tuple[float, str]:
        """
        Process a review event and update the neuron state via STDP.

        Args:
            rating:  Recall quality, 1 (Again) to 4 (Easy).
            t_now:   Review timestamp. Defaults to now.

        Returns:
            (new_weight, explanation_string)
        """
        if t_now is None:
            t_now = time.time()

        # 1. Apply membrane leak (time since last review)
        self.neuron.leak(t_now)

        # 2. Integrate the review stimulus (pre-synaptic spike)
        self.neuron.integrate(input_current=1.0)

        # 3. Apply STDP weight update based on recall rating
        w_before = self.neuron.w
        self.neuron.w = apply_stdp(w_before, rating, self.stdp_config)
        w_after = self.neuron.w

        # 4. Record spike time and reset membrane potential
        self.neuron.t_last = t_now
        self.neuron.reset()

        explanation = explain_update(w_before, w_after, rating)
        return w_after, explanation

    def receive_lateral_spike(
        self,
        spike_strength: float,
        link_weight: float = LATERAL_WEIGHT,
        t_now: float | None = None,
    ) -> bool:
        """
        Apply a lateral excitatory spike arriving from a linked (prerequisite) card.

        This is called on Card A after Card B (its prerequisite) was reviewed.

        Math:
            lateral_current = W_{B→A} · S_B
            V_A_new         = V_A_decayed + lateral_current

        where:
            W_{B→A}       = link_weight (default 0.3)
            S_B           = spike_strength = rating / 4  (0.25 → 1.0)
            V_A_decayed   = A's membrane potential after exponential leak

        Two outcomes:
        1. V_A_new >= 1.0  → spontaneous fire → reset t_last and V.
                             The review interval effectively resets as if A was recalled.
                             IMPORTANT: STDP weight is NOT changed — the user didn't
                             consciously review this card.

        2. V_A_new < 1.0   → sub-threshold priming → V is stored, slowing future
                             decay (A's next due date is pushed back slightly).

        Returns:
            True if the card fired spontaneously (its scheduled review can be skipped).
        """
        if t_now is None:
            t_now = time.time()

        # Step 1: apply exponential membrane leak since last review
        self.neuron.leak(t_now)

        # Step 2: compute lateral current and inject it into the membrane
        lateral_current = link_weight * spike_strength
        fired = self.neuron.inject_lateral(lateral_current)

        if fired:
            # Spontaneous fire: update the review clock but leave weight unchanged.
            # The neuron "remembered" without the user actively recalling —
            # the synapse strength should only change through conscious recall.
            self.neuron.t_last = t_now
            self.neuron.reset()

        return fired

    def is_due(self, t_now: float | None = None) -> bool:
        """True if this card is ready for review."""
        return self.neuron.is_due(t_now)

    def next_review_in_hours(self) -> float:
        """Hours until the next scheduled review."""
        now = time.time()
        delta = self.neuron.next_review_timestamp() - now
        return max(0.0, delta / 3600.0)

    def next_review_in_days(self) -> float:
        """Days until the next scheduled review."""
        return self.next_review_in_hours() / 24.0

    def memory_strength(self) -> float:
        """Synaptic weight, 0.0 (unknown) to 1.0 (mastered)."""
        return self.neuron.w

    def stability_label(self) -> str:
        """Human-readable memory strength label."""
        w = self.neuron.w
        if w < 0.15:
            return "New"
        elif w < 0.35:
            return "Learning"
        elif w < 0.60:
            return "Familiar"
        elif w < 0.80:
            return "Strong"
        else:
            return "Mastered"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return self.neuron.to_dict()

    @classmethod
    def from_dict(cls, d: dict, stdp_config: STDPConfig | None = None) -> "CardScheduler":
        return cls(
            weight=d["w"],
            v_mem=d["V"],
            t_last=d["t_last"],
            stdp_config=stdp_config,
        )
