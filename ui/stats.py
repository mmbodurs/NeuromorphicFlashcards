"""
Statistics & Analytics Dashboard
==================================
Visualizes memory strength distributions, review history,
STDP weight trajectories, and a three-way interval comparison:
SNN (our model) vs SM-2 (Anki) vs FSRS-4.5 (state-of-the-art).
"""

import math
import time

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from db.models import (
    get_connection, initialize_db, list_decks,
    get_cards_in_deck, get_review_history, get_deck_stats,
)
from snn.scheduler import CardScheduler


def render():
    st.header("📊 Statistics")

    conn = get_connection()
    initialize_db(conn)

    decks = list_decks(conn)
    if not decks:
        st.info("No decks yet. Create one in **Manage Cards**.")
        return

    deck_options = {d["name"]: d["id"] for d in decks}
    chosen = st.selectbox("Select deck", list(deck_options.keys()))
    deck_id = deck_options[chosen]

    stats = get_deck_stats(conn, deck_id)
    cards = get_cards_in_deck(conn, deck_id)

    # -- Summary metrics --
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cards", stats["total"])
    col2.metric("Reviewed", stats["reviewed"])
    col3.metric("New (unseen)", stats["new"])
    col4.metric("Avg Memory Strength", f"{stats['avg_weight']:.3f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Memory Distribution", "Review History", "Weight Trajectories", "Algorithm Comparison"
    ])

    with tab1:
        _memory_distribution(cards)

    with tab2:
        _review_history(conn)

    with tab3:
        _weight_trajectories(conn, deck_id)

    with tab4:
        _algorithm_comparison()


def _memory_distribution(cards):
    st.subheader("Memory Strength Distribution")
    if not cards:
        st.info("No cards yet.")
        return

    weights = [c["weight"] for c in cards]
    labels = []
    for w in weights:
        if w < 0.15:
            labels.append("New")
        elif w < 0.35:
            labels.append("Learning")
        elif w < 0.60:
            labels.append("Familiar")
        elif w < 0.80:
            labels.append("Strong")
        else:
            labels.append("Mastered")

    df = pd.DataFrame({"weight": weights, "stability": labels})

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="weight", nbins=20,
            title="Weight Distribution",
            color_discrete_sequence=["#3a86ff"],
            labels={"weight": "Memory Strength (w)", "count": "Cards"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        count_df = df["stability"].value_counts().reset_index()
        count_df.columns = ["Stability", "Count"]
        color_map = {
            "New": "#888888", "Learning": "#e07b39",
            "Familiar": "#3a86ff", "Strong": "#06a77d", "Mastered": "#8338ec"
        }
        fig2 = px.pie(
            count_df, names="Stability", values="Count",
            title="Cards by Stability",
            color="Stability", color_discrete_map=color_map
        )
        st.plotly_chart(fig2, use_container_width=True)


def _review_history(conn):
    st.subheader("Recent Review History")
    history = get_review_history(conn, limit=100)
    if not history:
        st.info("No reviews yet. Start studying!")
        return

    df = pd.DataFrame([dict(r) for r in history])
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], unit="s")
    df["rating_label"] = df["rating"].map({1: "Again", 2: "Hard", 3: "Good", 4: "Easy"})

    col1, col2 = st.columns(2)

    with col1:
        daily = df.set_index("reviewed_at").resample("D").size().reset_index()
        daily.columns = ["date", "reviews"]
        fig = px.bar(daily, x="date", y="reviews", title="Reviews per Day",
                     color_discrete_sequence=["#3a86ff"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dist = df["rating_label"].value_counts().reset_index()
        dist.columns = ["Rating", "Count"]
        color_map = {"Again": "#c0392b", "Hard": "#e07b39", "Good": "#06a77d", "Easy": "#3a86ff"}
        fig2 = px.bar(dist, x="Rating", y="Count", title="Rating Distribution",
                      color="Rating", color_discrete_map=color_map)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Last 20 Reviews")
    display_df = df[["reviewed_at", "deck_name", "front", "rating_label", "weight_before", "weight_after"]].head(20)
    display_df.columns = ["Time", "Deck", "Card", "Rating", "w Before", "w After"]
    display_df["Card"] = display_df["Card"].str[:50] + "..."
    st.dataframe(display_df, use_container_width=True)


def _weight_trajectories(conn, deck_id: int):
    st.subheader("Memory Strength Over Time")
    st.caption("Weight trajectories for recently reviewed cards.")

    history = get_review_history(conn, limit=500)
    if not history:
        st.info("No review history yet.")
        return

    df = pd.DataFrame([dict(r) for r in history if r["deck_id"] == deck_id])
    if df.empty:
        st.info("No review history for this deck yet.")
        return

    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], unit="s")
    df = df.sort_values("reviewed_at")

    top_cards = df["card_id"].value_counts().head(10).index.tolist()
    df_top = df[df["card_id"].isin(top_cards)]

    fig = go.Figure()
    for card_id in top_cards:
        cdf = df_top[df_top["card_id"] == card_id].sort_values("reviewed_at")
        label = cdf["front"].iloc[0][:30] + "..."
        fig.add_trace(go.Scatter(
            x=cdf["reviewed_at"], y=cdf["weight_after"],
            mode="lines+markers", name=label
        ))

    fig.update_layout(
        title="Weight Trajectories (Top 10 Cards)",
        xaxis_title="Date", yaxis_title="Memory Strength (w)",
        yaxis=dict(range=[0, 1]),
        legend=dict(font=dict(size=10))
    )
    st.plotly_chart(fig, use_container_width=True)


def _simulate_fsrs(ratings: list[int]) -> list[float]:
    """
    Simulate the FSRS-4.5 scheduler for a sequence of ratings.

    FSRS (Free Spaced Repetition Scheduler) is the state-of-the-art algorithm
    used by Anki since 2023.  Unlike SM-2, it is grounded in a power-law memory
    model and learns per-user weights.

    Core concepts
    -------------
    Stability (S): the number of days at which retrievability equals 90%.
        Higher S → memory is strong, next review can be scheduled far away.
    Difficulty (D): how intrinsically hard the card is (1 = easiest, 10 = hardest).
    Retrievability R(t): the probability of recalling the card t days after review.
        R(t) = (1 + FACTOR × t / S)^(-1)   where FACTOR = 19/81 ≈ 0.2346
        This is a power-law decay — slower than pure exponential at long delays.

    Scheduling rule
    ---------------
    Set R(interval) = desired_retention (0.9).  Solving for interval:
        0.9 = (1 + FACTOR × I / S)^(-1)
        I = S × (1/0.9 - 1) / FACTOR = S × (1/9) / (19/81) = S × 9/19 ≈ 0.473 × S

    This means the next review is scheduled at roughly half the stability value —
    so a card with S=10 days gets reviewed in ~4.7 days.

    Weight index mapping (0-based, FSRS-4.5 default weights)
    --------------------------------------------------------
    w[0..3]  : initial stability per grade (Again/Hard/Good/Easy)
    w[4], w[5]: initial difficulty formula parameters
    w[10..13]: stability formula when card is FORGOTTEN
    w[16..18]: stability formula when card is RECALLED
    """

    # FSRS-4.5 default weights w[0] through w[18] (0-indexed)
    # Source: https://github.com/open-spaced-repetition/fsrs4anki
    W = [
        0.4072,   # w[0]  initial S for Again
        1.1829,   # w[1]  initial S for Hard
        3.1262,   # w[2]  initial S for Good
        15.4722,  # w[3]  initial S for Easy
        7.2102,   # w[4]  initial difficulty base
        0.5316,   # w[5]  initial difficulty scaling
        1.0651,   # w[6]
        0.0589,   # w[7]
        1.5330,   # w[8]
        0.1544,   # w[9]
        1.0071,   # w[10] stability-after-forgetting coefficient
        1.9395,   # w[11] difficulty exponent in forgetting formula
        0.1100,   # w[12] stability growth exponent in forgetting formula
        0.2900,   # w[13] retrievability weight in forgetting formula
        2.2700,   # w[14]
        0.2500,   # w[15]
        2.9898,   # w[16] stability growth scaling (recall formula)
        0.5100,   # w[17] stability decay exponent (recall formula)
        0.3400,   # w[18] retrievability weight (recall formula)
    ]

    # Power-law retrievability factor: makes R(t) decay as a power law, not exponential.
    # Chosen so that R(S) = 0.9 exactly when FSRS_FACTOR = (0.9^-1 - 1) × (S/S) = 19/81.
    FSRS_FACTOR = 19 / 81

    # Desired retention: we want to review just before recall drops below 90%.
    DESIRED_RETENTION = 0.9

    # From R(I) = DESIRED_RETENTION, solving for I:
    #   DESIRED_RETENTION = (1 + FSRS_FACTOR × I / S)^(-1)
    #   I = S × (1/DESIRED_RETENTION - 1) / FSRS_FACTOR
    INTERVAL_FACTOR = (1.0 / DESIRED_RETENTION - 1.0) / FSRS_FACTOR  # ≈ 0.473

    intervals = []
    S = None   # stability (days)
    D = None   # difficulty [1, 10]
    t = 0.0    # days elapsed since last review

    for rating in ratings:
        # Map app ratings (1=Again, 2=Hard, 3=Good, 4=Easy)
        # to FSRS grade    (0=Again, 1=Hard, 2=Good, 3=Easy)
        grade = rating - 1

        if S is None:
            # ── First review: initialise stability and difficulty ──────────────
            # Initial stability is just the weight for this grade.
            # A "Good" first answer → S = 3.1 days; "Easy" → S = 15.5 days.
            S = W[grade]

            # Initial difficulty: higher grade (easier answer) → lower D.
            # Formula: D₀ = w[4] - e^(w[5] × (grade - 1)) + 1
            # grade=0 (Again) gives highest D; grade=3 (Easy) gives lowest D.
            D = W[4] - math.exp(W[5] * (grade - 1)) + 1
            D = max(1.0, min(10.0, D))   # clamp to valid range

        else:
            # ── Subsequent reviews: update S based on recall outcome ───────────
            # Compute retrievability at the time of this review.
            # This tells us how much the memory had decayed before the review.
            R = (1 + FSRS_FACTOR * t / S) ** (-1)

            if grade == 0:
                # Card was FORGOTTEN — stability collapses then partially recovers.
                # Short stability means we need to review again soon.
                # Formula (w[10..13]):
                #   S' = w[10] × D^(-w[11]) × ((S+1)^w[12] - 1) × e^(w[13]×(1-R))
                S = (
                    W[10]
                    * D ** (-W[11])
                    * ((S + 1) ** W[12] - 1)
                    * math.exp(W[13] * (1 - R))
                )
            else:
                # Card was RECALLED — stability grows, more so when:
                #   • difficulty D is low (easy card)
                #   • current stability S is low (early in learning)
                #   • retrievability R was low (review came late = spacing bonus)
                # Formula (w[16..18]):
                #   S' = S × e^(w[16] × (11-D) × S^(-w[17]) × (e^(w[18]×(1-R)) - 1))
                S = S * math.exp(
                    W[16]
                    * (11 - D)
                    * S ** (-W[17])
                    * (math.exp(W[18] * (1 - R)) - 1)
                )

            # Safety clamp — stability can never be less than 0.1 days
            S = max(0.1, S)

        # ── Compute next review interval ──────────────────────────────────────
        # Schedule the card so retrievability at review time equals 90%.
        # Derived from R(I) = DESIRED_RETENTION:
        #   I = S × (1/R_target - 1) / FSRS_FACTOR  ≈  0.473 × S
        interval = max(1.0, S * INTERVAL_FACTOR)
        intervals.append(interval)

        # Move the clock forward to the next review
        t = interval

    return intervals


def _algorithm_comparison():
    st.subheader("SNN vs SM-2 vs FSRS: Interval Comparison")
    st.markdown("""
    This chart compares the review intervals produced by three scheduling algorithms
    for identical rating sequences, so you can see how each model "thinks" about memory.

    | Algorithm | Core idea | Used by |
    |-----------|-----------|---------|
    | **SNN** | Synaptic weight (biologically grounded STDP) | This app |
    | **SM-2** | Ease-factor multiplier on a fixed interval | Anki (classic) |
    | **FSRS-4.5** | Power-law decay; stability + difficulty parameters | Anki (since 2023) |

    **SM-2 interval rule:**
    - Rating 1 (Again): reset to 1 day, ease factor decreases
    - Rating 2 (Hard): repeat current interval
    - Rating 3 (Good): `interval × EF` (EF starts at 2.5)
    - Rating 4 (Easy): `interval × EF × 1.3`

    **SNN interval rule:**
    - `interval = 4h × exp(weight × 5.19)`
    - Weight updated via STDP (spike-timing dependent plasticity)

    **FSRS interval rule:**
    - Interval scheduled so retrievability `R(t) = 90%`
    - `R(t) = (1 + 19/81 × t/S)⁻¹`  (power-law forgetting)
    - Stability S grows after each successful recall; collapses after forgetting
    """)

    scenario = st.selectbox("Rating scenario", [
        "All Good (3333...)",
        "All Easy (4444...)",
        "Mixed Good/Hard (3232...)",
        "Forget and recover (3341...)",
    ])

    rating_patterns = {
        "All Good (3333...)": [3] * 15,
        "All Easy (4444...)": [4] * 15,
        "Mixed Good/Hard (3232...)": ([3, 2] * 8)[:15],
        "Forget and recover (3341...)": ([3, 3, 4, 1] * 4)[:15],
    }
    ratings = rating_patterns[scenario]

    # ── SNN simulation ────────────────────────────────────────────────────────
    # Start from a nearly-new card (weight = 0.05) and simulate forward.
    scheduler = CardScheduler(weight=0.05)
    snn_intervals = []
    t = time.time()
    for r in ratings:
        scheduler.review(r, t_now=t)
        interval = scheduler.next_review_in_hours() / 24   # convert hours → days
        snn_intervals.append(interval)
        t += interval * 86400                               # advance clock

    # ── SM-2 simulation ───────────────────────────────────────────────────────
    # Classic Anki algorithm: ease-factor multiplied onto a growing interval.
    sm2_intervals = []
    interval_sm2 = 1.0   # starts at 1 day
    ef = 2.5             # ease factor; decreases on hard/fail, increases on easy
    for r in ratings:
        if r == 1:        # Again: reset and penalise ease factor
            interval_sm2 = 1.0
            ef = max(1.3, ef - 0.2)
        elif r == 2:      # Hard: keep same interval (ease factor unchanged here)
            pass
        elif r == 3:      # Good: multiply by ease factor
            interval_sm2 = interval_sm2 * ef
        elif r == 4:      # Easy: multiply by ease factor plus a bonus
            interval_sm2 = interval_sm2 * ef * 1.3
        sm2_intervals.append(interval_sm2)

    # ── FSRS-4.5 simulation ───────────────────────────────────────────────────
    # Power-law memory model with stability + difficulty parameters.
    fsrs_intervals = _simulate_fsrs(ratings)

    # ── Build dataframe and chart ─────────────────────────────────────────────
    df = pd.DataFrame({
        "Review":        list(range(1, len(ratings) + 1)),
        "Rating":        ratings,
        "SNN (days)":    [round(v, 2) for v in snn_intervals],
        "SM-2 (days)":   [round(v, 2) for v in sm2_intervals],
        "FSRS (days)":   [round(v, 2) for v in fsrs_intervals],
    })

    fig = go.Figure()

    # SNN — solid blue
    fig.add_trace(go.Scatter(
        x=df["Review"], y=df["SNN (days)"],
        mode="lines+markers", name="SNN (this app)",
        line=dict(color="#3a86ff", width=2),
    ))

    # SM-2 — dashed orange (classic Anki)
    fig.add_trace(go.Scatter(
        x=df["Review"], y=df["SM-2 (days)"],
        mode="lines+markers", name="SM-2 (Anki classic)",
        line=dict(color="#e07b39", width=2, dash="dash"),
    ))

    # FSRS — dotted green (modern Anki / state-of-the-art)
    fig.add_trace(go.Scatter(
        x=df["Review"], y=df["FSRS (days)"],
        mode="lines+markers", name="FSRS-4.5 (Anki modern)",
        line=dict(color="#06a77d", width=2, dash="dot"),
    ))

    fig.update_layout(
        xaxis_title="Review Number",
        yaxis_title="Interval until next review (days)",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)
