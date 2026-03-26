"""
Neuromorphic Flashcard Engine
==============================
A spaced-repetition flashcard app where the review scheduler is powered
by a Leaky Integrate-and-Fire (LIF) spiking neural network and updated
via Spike-Timing-Dependent Plasticity (STDP) — replacing the classical
SM-2 algorithm used by Anki.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Neuromorphic Flashcards",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
PAGES = {
    "🧠 Study": "study",
    "📚 Manage Cards": "cards",
    "📊 Statistics": "stats",
    "ℹ️ About": "about",
}

with st.sidebar:
    st.markdown("## 🧠 Neuromorphic Flashcards")
    st.caption("Powered by LIF neurons & STDP")
    st.markdown("---")
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        """
        **How it works:**

        Each card has a **LIF neuron**.
        Its *synaptic weight w* encodes memory strength.

        When you review a card, **STDP** updates *w*:
        - ✅ Correct recall → LTP (weight ↑)
        - ❌ Failed recall → LTD (weight ↓)

        The next review interval is:
        > `interval = 4h × exp(w × 5.19)`

        Strong memory = longer intervals, naturally.
        """,
        unsafe_allow_html=False
    )

page = PAGES[selection]

if page == "study":
    from ui.study import render
    render()
elif page == "cards":
    from ui.cards import render
    render()
elif page == "stats":
    from ui.stats import render
    render()
elif page == "about":
    st.header("ℹ️ About Neuromorphic Flashcards")

    st.markdown("""
    ### The Core Idea

    Traditional spaced repetition (Anki's **SM-2**) uses a hand-crafted
    *ease factor* multiplied against intervals. It works, but it's a heuristic
    with no biological basis.

    This app replaces SM-2 with a proper **spiking neural network** model:

    | Component | Role |
    |---|---|
    | **LIF neuron** | Represents each card's memory trace |
    | **Synaptic weight *w*** | Memory strength ∈ [0, 1] |
    | **Membrane potential *V*** | Accumulated activation (leaks over time = forgetting) |
    | **STDP rule** | Updates *w* based on recall quality |

    ### The STDP Rule

    Spike-Timing-Dependent Plasticity (STDP) is a biological learning rule
    discovered in cortical neurons. In our model:

    - **Pre-synaptic spike** = the card is shown
    - **Post-synaptic spike** = successful recall

    Causal pairing (card shown → recalled quickly) causes
    **Long-Term Potentiation (LTP)**: the synapse strengthens.

    Failed recall causes **Long-Term Depression (LTD)**: the synapse weakens.

    The weight update is:

    ```
    ΔW = A₊ × exp(-Δt / τ₊)   [LTP, causal]
    ΔW = -A₋ × exp(-Δt / τ₋)  [LTD, anti-causal]
    ```

    Weight-dependence prevents saturation — stronger synapses are harder
    to further potentiate, mirroring biological homeostasis.

    ### Interval Formula

    ```
    interval = 4h × exp(w × ln(720/4))
    ```

    This gives:
    - w = 0.0 → 4 hours   (brand new card)
    - w = 0.5 → ~48 hours  (2 days)
    - w = 1.0 → 720 hours  (30 days)

    ### References

    - Bi & Poo (1998) — Original STDP discovery in hippocampal neurons
    - Maass (1997) — Networks of spiking neurons
    - Wozniak (1990) — SM-2 algorithm (Anki)
    - snnTorch (Eshraghian et al., 2021) — SNN training framework
    """)
