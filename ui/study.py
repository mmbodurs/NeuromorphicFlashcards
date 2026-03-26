"""
Study Session UI
================
Shows due flashcards one at a time. The user rates recall quality (1–4),
which triggers the STDP weight update and reschedules the card.

Lateral excitatory synapses:
    If a card's text contains [[Other Card]], reviewing it will inject
    lateral current into "Other Card"'s neuron. Strong recall can
    spontaneously fire the linked card, skipping its next review.
"""

import time
import streamlit as st
from db.models import (
    get_connection, initialize_db, list_decks, get_due_cards,
    get_snn_state, update_snn_state, log_review,
    get_cards_primed_by, get_link_refs,
)
from snn.scheduler import CardScheduler, LATERAL_WEIGHT


def render():
    st.header("🧠 Study Session")

    conn = get_connection()
    initialize_db(conn)

    decks = list_decks(conn)
    if not decks:
        st.info("No decks yet. Go to **Manage Cards** to create one.")
        return

    deck_options = {d["name"]: d["id"] for d in decks}
    chosen_deck_name = st.selectbox("Choose a deck", list(deck_options.keys()))
    deck_id = deck_options[chosen_deck_name]

    due_cards = get_due_cards(conn, deck_id)

    if not due_cards:
        st.success("🎉 All caught up! No cards due in this deck right now.")
        _show_next_due(conn, deck_id)
        return

    st.info(f"**{len(due_cards)}** card(s) due")

    # -- Session state initialization --
    if "study_queue" not in st.session_state or st.session_state.get("study_deck") != deck_id:
        st.session_state.study_queue = [dict(c) for c in due_cards]
        st.session_state.study_deck = deck_id
        st.session_state.study_idx = 0
        st.session_state.show_back = False
        st.session_state.session_done = False
        # lateral_log stores priming events to display after each rating
        st.session_state.lateral_log = []

    if st.session_state.session_done:
        st.balloons()
        st.success("Session complete! All due cards reviewed.")
        if st.button("Start again"):
            del st.session_state["study_queue"]
            del st.session_state["session_done"]
            st.rerun()
        return

    queue = st.session_state.study_queue
    idx = st.session_state.study_idx

    if idx >= len(queue):
        st.session_state.session_done = True
        st.rerun()
        return

    card = queue[idx]
    progress = idx / len(queue)
    st.progress(progress, text=f"Card {idx + 1} of {len(queue)}")

    # -- Card display --
    st.markdown("---")

    # Stability badge
    snn_row = get_snn_state(conn, card["id"])
    scheduler = CardScheduler(
        weight=snn_row["weight"],
        v_mem=snn_row["v_mem"],
        t_last=snn_row["t_last"],
    )
    stability = scheduler.stability_label()
    badge_color = {
        "New": "#888888", "Learning": "#E07B39",
        "Familiar": "#3A86FF", "Strong": "#06A77D", "Mastered": "#8338EC"
    }.get(stability, "#888888")

    st.markdown(
        f'<span style="background:{badge_color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:bold">{stability}</span>',
        unsafe_allow_html=True
    )

    # Tags
    if card.get("tags"):
        st.caption(f"Tags: {card['tags']}")

    # ----------------------------------------------------------------
    # Prerequisite link display — show what this card depends on
    # ----------------------------------------------------------------
    all_text = (card["front"] or "") + " " + (card["back"] or "")
    prereq_refs = get_link_refs(all_text)
    if prereq_refs:
        _render_prereq_pills(prereq_refs)

    st.markdown("### Front")
    # Strip [[link]] tokens from the displayed card text — they're internal metadata
    display_front = _strip_links(card["front"])
    st.markdown(
        f'<div style="background:#1e1e2e;color:#cdd6f4;border-radius:12px;'
        f'padding:24px 28px;font-size:1.2rem;line-height:1.6;margin-bottom:12px">'
        f'{display_front}</div>',
        unsafe_allow_html=True
    )

    # Show lateral priming result from the PREVIOUS card's rating
    if st.session_state.lateral_log:
        _render_lateral_log(st.session_state.lateral_log)
        st.session_state.lateral_log = []

    if not st.session_state.show_back:
        if st.button("Show Answer", use_container_width=True, type="primary"):
            st.session_state.show_back = True
            st.rerun()
    else:
        st.markdown("### Back")
        display_back = _strip_links(card["back"])
        st.markdown(
            f'<div style="background:#1e2e1e;color:#a6e3a1;border-radius:12px;'
            f'padding:24px 28px;font-size:1.2rem;line-height:1.6;margin-bottom:20px">'
            f'{display_back}</div>',
            unsafe_allow_html=True
        )

        st.markdown("**How well did you recall this?**")
        col1, col2, col3, col4 = st.columns(4)

        rating_map = {
            "🔴 Again": (1, col1, "#c0392b"),
            "🟠 Hard":  (2, col2, "#e07b39"),
            "🟢 Good":  (3, col3, "#06a77d"),
            "🔵 Easy":  (4, col4, "#3a86ff"),
        }

        chosen_rating = None
        for label, (rating, col, color) in rating_map.items():
            with col:
                if st.button(label, use_container_width=True, key=f"rate_{rating}"):
                    chosen_rating = rating

        if chosen_rating is not None:
            _process_rating(conn, card, scheduler, chosen_rating)

            # After the STDP update, propagate a lateral spike to all cards
            # that reference this card with [[card front]] in their text.
            lateral_events = _apply_lateral_priming(conn, card, chosen_rating, deck_id)
            st.session_state.lateral_log = lateral_events

            st.session_state.study_idx += 1
            st.session_state.show_back = False
            st.rerun()

        # Show what STDP will do for each option
        st.markdown("---")
        _show_stdp_preview(scheduler)


# ------------------------------------------------------------------
# Lateral priming — propagate spike to linked cards
# ------------------------------------------------------------------

def _apply_lateral_priming(conn, card: dict, rating: int, deck_id: int) -> list[dict]:
    """
    After Card B is reviewed with `rating`, find all cards that depend on B
    (i.e., contain [[B front]] in their text) and inject lateral current.

    spike_strength = rating / 4  →  Again=0.25, Hard=0.5, Good=0.75, Easy=1.0

    Returns a list of dicts describing each priming event, for display.
    """
    # S_B: intensity of the spike train from this recall
    spike_strength = rating / 4.0

    primed_cards = get_cards_primed_by(conn, card["id"], deck_id)
    events = []
    t_now = time.time()

    for target in primed_cards:
        target_snn = get_snn_state(conn, target["id"])
        target_scheduler = CardScheduler(
            weight=target_snn["weight"],
            v_mem=target_snn["v_mem"],
            t_last=target_snn["t_last"],
        )

        # Compute the lateral current before injection for display purposes
        lateral_current = LATERAL_WEIGHT * spike_strength
        v_before = target_scheduler.neuron.V

        # Apply the lateral spike — may or may not trigger spontaneous fire
        fired = target_scheduler.receive_lateral_spike(
            spike_strength=spike_strength,
            link_weight=LATERAL_WEIGHT,
            t_now=t_now,
        )

        # Persist the updated neuron state (V and t_last may have changed)
        update_snn_state(
            conn,
            card_id=target["id"],
            weight=target_scheduler.neuron.w,   # weight unchanged by design
            v_mem=target_scheduler.neuron.V,
            t_last=target_scheduler.neuron.t_last,
        )

        events.append({
            "front": target["front"],
            "lateral_current": lateral_current,
            "v_before": v_before,
            "v_after": target_scheduler.neuron.V,
            "fired": fired,
            "new_interval_h": target_scheduler.next_review_in_hours(),
        })

    return events


# ------------------------------------------------------------------
# Aesthetic rendering helpers
# ------------------------------------------------------------------

def _render_prereq_pills(refs: list[str]):
    """
    Render a subtle 'built on' row showing which cards this card depends on.
    Positioned just above the card front — gives a neural network topology feel.
    """
    pills = " ".join(
        f'<span style="background:#181825;color:#89b4fa;border:1px solid #313244;'
        f'border-radius:20px;padding:3px 11px;font-size:0.72rem;'
        f'letter-spacing:0.03em;white-space:nowrap">'
        f'<span style="opacity:0.6">⬡</span> {ref}</span>'
        for ref in refs
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;'
        f'flex-wrap:wrap">'
        f'<span style="color:#585b70;font-size:0.72rem;letter-spacing:0.05em;'
        f'text-transform:uppercase">Built on</span>'
        f'{pills}</div>',
        unsafe_allow_html=True,
    )


def _render_lateral_log(events: list[dict]):
    """
    Render a styled notification block showing which cards were primed
    by the lateral spike from the card that was just rated.
    """
    fired_cards   = [e for e in events if e["fired"]]
    primed_cards  = [e for e in events if not e["fired"]]

    if not events:
        return

    # Build rows for each event
    rows_html = ""
    for e in events:
        front_short = e["front"][:48] + ("…" if len(e["front"]) > 48 else "")
        current_str = f"+{e['lateral_current']:.2f} V"

        if e["fired"]:
            # Spontaneous fire — show a more prominent indicator
            status_html = (
                f'<span style="background:#1e3a2e;color:#a6e3a1;border-radius:6px;'
                f'padding:1px 8px;font-size:0.72rem;font-weight:bold">⚡ spontaneous fire</span>'
            )
            interval_str = f"review skipped — resets to {e['new_interval_h']:.1f}h"
        else:
            bar_width = min(int(e["v_after"] * 80), 80)
            status_html = (
                f'<span style="background:#1a1a2e;border-radius:3px;'
                f'display:inline-block;width:80px;height:6px;vertical-align:middle;'
                f'overflow:hidden">'
                f'<span style="background:#45475a;display:block;width:{bar_width}px;'
                f'height:6px"></span></span>'
                f'<span style="color:#585b70;font-size:0.7rem;margin-left:6px">'
                f'V={e["v_after"]:.2f}</span>'
            )
            interval_str = f"next review in {e['new_interval_h']:.1f}h"

        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid #1e1e2e">'
            f'<span style="color:#cdd6f4;font-size:0.82rem">{front_short}</span>'
            f'<span style="display:flex;align-items:center;gap:10px">'
            f'<span style="color:#f38ba8;font-size:0.75rem">{current_str}</span>'
            f'{status_html}'
            f'<span style="color:#585b70;font-size:0.72rem">{interval_str}</span>'
            f'</span></div>'
        )

    # Title line
    n = len(events)
    fire_count = len(fired_cards)
    title = f"⟶  Lateral priming: {n} card{'s' if n > 1 else ''} received excitatory current"
    if fire_count:
        title += f" · {fire_count} spontaneous fire{'s' if fire_count > 1 else ''}"

    st.markdown(
        f'<div style="background:#12121e;border:1px solid #313244;border-radius:10px;'
        f'padding:12px 16px;margin:12px 0">'
        f'<div style="color:#89b4fa;font-size:0.78rem;font-weight:600;'
        f'letter-spacing:0.04em;margin-bottom:8px">{title}</div>'
        f'{rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _strip_links(text: str) -> str:
    """
    Remove [[link]] tokens from displayed card text.
    They are internal wiring metadata — the user shouldn't see them mid-study.
    """
    import re
    return re.sub(r'\s*\[\[[^\]]+\]\]', '', text).strip()


# ------------------------------------------------------------------
# Original helpers (unchanged)
# ------------------------------------------------------------------

def _process_rating(conn, card, scheduler: CardScheduler, rating: int):
    """Apply STDP update and persist to DB."""
    t_now = time.time()
    w_before = scheduler.memory_strength()
    w_after, explanation = scheduler.review(rating, t_now)

    update_snn_state(
        conn,
        card_id=card["id"],
        weight=w_after,
        v_mem=scheduler.neuron.V,
        t_last=t_now,
    )
    log_review(
        conn,
        card_id=card["id"],
        rating=rating,
        weight_before=w_before,
        weight_after=w_after,
        interval_hours=scheduler.next_review_in_hours(),
    )


def _show_stdp_preview(scheduler: CardScheduler):
    """Show what each rating would do to the weight."""
    from snn.stdp import apply_stdp
    w = scheduler.memory_strength()
    st.caption("**STDP preview** — how each rating would change your memory strength:")
    cols = st.columns(4)
    for i, (label, rating) in enumerate(zip(["Again", "Hard", "Good", "Easy"], [1, 2, 3, 4])):
        new_w = apply_stdp(w, rating)
        delta = new_w - w
        sign = "+" if delta >= 0 else ""
        cols[i].metric(label, f"{new_w:.3f}", f"{sign}{delta:.3f}")


def _show_next_due(conn, deck_id: int):
    """Show when the next card is due."""
    row = conn.execute("""
        SELECT MIN(s.t_last + (4.0 * exp(s.weight * 5.19)) * 3600) as next_ts
        FROM snn_state s JOIN cards c ON c.id = s.card_id
        WHERE c.deck_id = ? AND c.is_active = 1
    """, (deck_id,)).fetchone()
    if row and row["next_ts"]:
        delta_h = (row["next_ts"] - time.time()) / 3600
        if delta_h < 1:
            st.caption(f"Next card due in {int(delta_h * 60)} minutes.")
        elif delta_h < 24:
            st.caption(f"Next card due in {delta_h:.1f} hours.")
        else:
            st.caption(f"Next card due in {delta_h / 24:.1f} days.")
