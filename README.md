# Neuromorphic Flashcard Engine 🧠

A spaced-repetition flashcard app powered by a **Leaky Integrate-and-Fire (LIF) spiking neural network** and **Spike-Timing-Dependent Plasticity (STDP)** — replacing Anki's SM-2 algorithm with a biologically grounded memory model.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## How It Works

Each flashcard is represented by a **LIF neuron** with a synaptic weight `w ∈ [0, 1]` encoding memory strength.

When you review a card and rate your recall (1–4), **STDP** updates the weight:
- ✅ Correct recall → Long-Term Potentiation (LTP): `w` increases
- ❌ Failed recall → Long-Term Depression (LTD): `w` decreases

The next review interval is derived from `w` using an exponential curve inspired by the **Ebbinghaus forgetting curve**:

```
interval = 4h × exp(w × ln(720/4))
```

| Weight | Interval |
|--------|----------|
| 0.0    | 4 hours  |
| 0.5    | ~2 days  |
| 1.0    | 30 days  |

---

## Features

### Study Sessions
Review cards due today. After flipping a card, rate your recall:
- **1 – Again**: completely forgot it
- **2 – Hard**: recalled with difficulty
- **3 – Good**: recalled correctly
- **4 – Easy**: recalled instantly

### Card & Deck Management
Create decks and add cards individually or via **bulk import** (`front | back`, one per line).

### Card Linking (Knowledge Graph)
Cards can be linked to each other to express prerequisite relationships. For example, if a card about *action potentials* depends on understanding *membrane potential*, you can link them.

**How to link cards:**
- When adding or editing a card, use the **🔗 Prerequisite links** picker to select cards this one builds on
- Or type `[[Card front text]]` directly in the back field

**What happens under the hood:**
When a prerequisite card (B) is reviewed successfully, every card that links to it (A) receives a **lateral excitatory current**:

```
I_lateral = W_{B→A} × S_B     where S_B = rating / 4
```

If the injected current pushes card A's membrane potential past the firing threshold, it fires spontaneously — resetting its decay timer without changing its weight (a "free reminder"). Otherwise it receives sub-threshold priming, slowing its forgetting.

This embeds a **knowledge dependency graph** directly into the SNN topology, mimicking how cortical circuits use lateral excitation.

### Statistics Dashboard
The analytics page has four tabs:

**Memory Distribution** — histogram and pie chart showing how your cards are distributed across stability stages (New → Learning → Familiar → Strong → Mastered).

**Review History** — reviews-per-day bar chart, rating distribution, and a table of your last 20 reviews.

**Weight Trajectories** — line chart tracking how memory strength (`w`) has evolved over time for your most-reviewed cards.

**Algorithm Comparison** — side-by-side simulation of three scheduling algorithms across identical rating sequences:

| Algorithm | Core model | Used by |
|-----------|-----------|---------|
| **SNN** | Synaptic weight via STDP | This app |
| **SM-2** | Ease-factor multiplier | Anki (classic) |
| **FSRS-4.5** | Power-law decay; stability + difficulty | Anki (since 2023) |

FSRS models forgetting with a power-law rather than an exponential:
```
R(t) = (1 + 19/81 × t/S)^(-1)
```
The next interval is set so retrievability stays at 90%: `I ≈ 0.473 × S`

---

## Project Structure

```
neuromorphic-flashcards/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── snn/
│   ├── neuron.py           # LIF neuron — membrane potential, leak, firing
│   ├── stdp.py             # STDP learning rule — weight updates from recall
│   └── scheduler.py        # Review scheduling + lateral priming logic
├── db/
│   └── models.py           # SQLite database layer
├── ui/
│   ├── study.py            # Study session page (shows prereq pills, priming log)
│   ├── cards.py            # Card/deck management + link picker
│   └── stats.py            # Analytics dashboard (incl. FSRS comparison)
└── docs/
    └── concepts/           # Concept explainers for the science behind the app
```

---

## Data Storage

Cards and review history are stored locally in a SQLite database at:
```
~/.neuromorphic_flashcards/cards.db
```

No data leaves your machine.

---

## References

- Bi & Poo (1998) — STDP discovery in hippocampal neurons
- Maass (1997) — Networks of spiking neurons
- Wozniak (1990) — SM-2 algorithm
- Eshraghian et al. (2021) — snnTorch framework
- Ebbinghaus (1885) — The forgetting curve
- Ye et al. (2022) — [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki)
