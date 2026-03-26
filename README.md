# Neuromorphic Flashcard Engine 🧠

A spaced-repetition flashcard app powered by a **Leaky Integrate-and-Fire (LIF) spiking neural network** and **Spike-Timing-Dependent Plasticity (STDP)** — replacing Anki's SM-2 algorithm with a biologically grounded memory model.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How It Works

Each flashcard is represented by a **LIF neuron** with a synaptic weight `w ∈ [0, 1]` encoding memory strength.

When you review a card and rate your recall (1–4), **STDP** updates the weight:
- ✅ Correct recall → Long-Term Potentiation (LTP): `w` increases
- ❌ Failed recall → Long-Term Depression (LTD): `w` decreases

The next review interval is derived from `w`:
```
interval = 4h × exp(w × ln(720/4))
```
| Weight | Interval |
|--------|----------|
| 0.0    | 4 hours  |
| 0.5    | ~2 days  |
| 1.0    | 30 days  |

## Project Structure

```
neuromorphic-flashcards/
├── app.py              # Streamlit entry point
├── requirements.txt
├── snn/
│   ├── neuron.py       # LIF neuron model
│   ├── stdp.py         # STDP learning rule
│   └── scheduler.py    # Review scheduling engine
├── db/
│   └── models.py       # SQLite database layer
└── ui/
    ├── study.py        # Study session page
    ├── cards.py        # Card/deck management page
    └── stats.py        # Analytics dashboard
```

## Data Storage

Cards and review history are stored in a local SQLite database at:
`~/.neuromorphic_flashcards/cards.db`

## References

- Bi & Poo (1998) — STDP discovery in hippocampal neurons
- Maass (1997) — Networks of spiking neurons
- Wozniak (1990) — SM-2 algorithm
- Eshraghian et al. (2021) — snnTorch framework
