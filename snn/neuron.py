"""
Leaky Integrate-and-Fire (LIF) Neuron Model
============================================
Each flashcard is represented by an LIF neuron. The neuron's synaptic
weight encodes memory strength. The membrane potential accumulates
"evidence of recall" during a study session and leaks over time
(forgetting). When a review event fires, STDP updates the weight.

Key variables per card-neuron:
    w       : synaptic weight  [0.0, 1.0]  — memory strength
    V       : membrane potential            — current activation level
    t_last  : timestamp of last review spike (Unix seconds)
    theta   : firing threshold (fixed at 1.0 for simplicity)
    tau_m   : membrane time constant (hours) — controls forgetting rate
"""

import math
import time


class LIFNeuron:
    """
    A single Leaky Integrate-and-Fire neuron representing one flashcard's
    memory trace.
    """

    THETA = 1.0          # Firing threshold (normalized)
    TAU_M_HOURS = 24.0   # Membrane time constant: potential halves every ~24h

    def __init__(self, weight: float = 0.1, v_mem: float = 0.0, t_last: float | None = None):
        """
        Args:
            weight:  Initial synaptic weight (memory strength). New cards start weak.
            v_mem:   Initial membrane potential.
            t_last:  Timestamp of last review spike (seconds since epoch).
                     None means the card has never been reviewed.
        """
        self.w = float(weight)
        self.V = float(v_mem)
        self.t_last = t_last if t_last is not None else time.time()

    # ------------------------------------------------------------------
    # Membrane dynamics
    # ------------------------------------------------------------------

    def leak(self, t_now: float | None = None) -> float:
        """
        Apply exponential leak to the membrane potential based on elapsed
        time since the last update. Returns the decayed potential.

        V(t) = V(t_last) * exp(-(t - t_last) / tau)

        Args:
            t_now: Current timestamp (seconds). Defaults to time.time().
        """
        if t_now is None:
            t_now = time.time()

        elapsed_hours = (t_now - self.t_last) / 3600.0
        decay = math.exp(-elapsed_hours / self.TAU_M_HOURS)
        self.V *= decay
        return self.V

    def integrate(self, input_current: float) -> float:
        """
        Add a synaptic input current to the membrane potential.
        The current is weighted by the neuron's synaptic weight.

        Returns the new membrane potential.
        """
        self.V += self.w * input_current
        return self.V

    def inject_lateral(self, lateral_current: float) -> bool:
        """
        Inject excitatory current arriving from a lateral synapse
        (i.e., a linked card's spike train).

        This implements the W_{B→A} · S_B term from the expanded LIF equation:
            τ_m · dV_A/dt = -(V_A - V_rest) + R(I_direct + W_{B→A} · S_B)

        The link weight and spike strength are already multiplied together
        before this call — we just add the result directly to V.

        Returns True if the membrane potential crossed the firing threshold,
        meaning this card "fired spontaneously" due to its neighbour.
        """
        self.V += lateral_current
        # Check whether V has crossed theta — spontaneous fire
        return self.V >= self.THETA

    def did_fire(self) -> bool:
        """Returns True if V has crossed the threshold."""
        return self.V >= self.THETA

    def reset(self):
        """Reset membrane potential after a spike (hard reset)."""
        self.V = 0.0

    # ------------------------------------------------------------------
    # Interval prediction
    # ------------------------------------------------------------------

    def next_interval_hours(self) -> float:
        """
        Compute the recommended review interval (in hours) based on the
        current synaptic weight.

        Biologically inspired: stronger synapses → longer-lasting memories.
        The relationship is exponential (like Ebbinghaus forgetting curve):

            interval = min_h * exp(w * scale)

        where scale is tuned so that:
            w=0.0  → ~4 h   (brand new card)
            w=0.5  → ~48 h  (2 days — solid recall)
            w=1.0  → ~720 h (30 days — long-term memory)
        """
        min_h = 4.0
        scale = math.log(720.0 / min_h)   # ≈ 5.19
        return min_h * math.exp(self.w * scale)

    def next_review_timestamp(self) -> float:
        """Returns the Unix timestamp when this card should next be reviewed."""
        return self.t_last + self.next_interval_hours() * 3600.0

    def is_due(self, t_now: float | None = None) -> bool:
        """Returns True if the card is due for review right now."""
        if t_now is None:
            t_now = time.time()
        return t_now >= self.next_review_timestamp()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"w": self.w, "V": self.V, "t_last": self.t_last}

    @classmethod
    def from_dict(cls, d: dict) -> "LIFNeuron":
        return cls(weight=d["w"], v_mem=d["V"], t_last=d["t_last"])

    def __repr__(self) -> str:
        interval_h = self.next_interval_hours()
        if interval_h < 48:
            interval_str = f"{interval_h:.1f}h"
        else:
            interval_str = f"{interval_h/24:.1f}d"
        return f"LIFNeuron(w={self.w:.3f}, V={self.V:.3f}, next={interval_str})"
