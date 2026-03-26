"""
Card & Deck Management UI
==========================
Create/edit/delete decks and cards. Supports bulk card input via
a simple separator syntax.
"""

import streamlit as st
from db.models import (
    get_connection, initialize_db, list_decks, create_deck, delete_deck,
    create_card, get_cards_in_deck, update_card, delete_card, get_link_refs,
)


def render():
    st.header("📚 Manage Cards")

    conn = get_connection()
    initialize_db(conn)

    tab_decks, tab_cards = st.tabs(["Decks", "Cards"])

    with tab_decks:
        _render_decks(conn)

    with tab_cards:
        _render_cards(conn)


def _render_decks(conn):
    st.subheader("Your Decks")

    decks = list_decks(conn)
    if decks:
        for deck in decks:
            col1, col2, col3 = st.columns([4, 1, 1])
            col1.markdown(f"**{deck['name']}** — {deck['card_count']} card(s)")
            if deck["description"]:
                col1.caption(deck["description"])
            if col3.button("🗑️", key=f"del_deck_{deck['id']}", help="Delete deck"):
                delete_deck(conn, deck["id"])
                st.rerun()
    else:
        st.info("No decks yet. Create one below.")

    st.markdown("---")
    st.subheader("Create New Deck")

    with st.form("new_deck"):
        name = st.text_input("Deck name", placeholder="e.g. Chinese HSK2 Vocab")
        desc = st.text_area("Description (optional)", height=80)
        if st.form_submit_button("Create Deck", type="primary"):
            if name.strip():
                create_deck(conn, name.strip(), desc.strip())
                st.success(f"Deck '{name}' created!")
                st.rerun()
            else:
                st.error("Please enter a deck name.")


def _render_cards(conn):
    decks = list_decks(conn)
    if not decks:
        st.info("Create a deck first.")
        return

    deck_options = {d["name"]: d["id"] for d in decks}
    chosen_deck = st.selectbox("Select deck", list(deck_options.keys()), key="cards_deck_select")
    deck_id = deck_options[chosen_deck]

    st.markdown("---")

    add_tab, view_tab, bulk_tab = st.tabs(["Add Card", "View & Edit Cards", "Bulk Import"])

    with add_tab:
        _add_card_form(conn, deck_id)

    with view_tab:
        _view_cards(conn, deck_id)

    with bulk_tab:
        _bulk_import(conn, deck_id)


def _add_card_form(conn, deck_id: int):
    st.subheader("Add a New Card")

    # Collect existing cards so the user can pick prerequisites from a dropdown
    existing_cards = get_cards_in_deck(conn, deck_id)
    existing_fronts = [c["front"] for c in existing_cards]

    with st.form("new_card"):
        front = st.text_area("Front (question / prompt)", height=100,
                             placeholder="What is the membrane potential equation for a LIF neuron?")
        back = st.text_area("Back (answer)", height=100,
                            placeholder="τ dV/dt = -(V - V_rest) + R·I(t)")
        tags = st.text_input("Tags (comma-separated, optional)", placeholder="neuroscience, snn, equations")

        # Prerequisite link picker
        # Selecting Card B here means "this new card DEPENDS ON Card B"
        # → [[Card B front]] will be appended to the back text
        prereqs = st.multiselect(
            "🔗 Prerequisite links (optional)",
            options=existing_fronts,
            help=(
                "Select cards that this card builds on. "
                "[[link]] tokens will be added to the back text. "
                "You can also type [[Card name]] directly."
            ),
        )

        if st.form_submit_button("Add Card", type="primary"):
            if front.strip() and back.strip():
                # Append [[prereq]] tokens that aren't already in the text
                back_final = back.strip()
                for p in prereqs:
                    token = f"[[{p}]]"
                    if token not in back_final:
                        back_final += f"\n{token}"
                create_card(conn, deck_id, front, back_final, tags)
                st.success("Card added!")
                st.rerun()
            else:
                st.error("Both front and back are required.")


def _view_cards(conn, deck_id: int):
    cards = get_cards_in_deck(conn, deck_id)
    if not cards:
        st.info("No cards in this deck yet.")
        return

    st.subheader(f"{len(cards)} card(s)")
    search = st.text_input("Search cards", placeholder="Filter by content or tag...")

    # Build a lookup of front → card for the link picker in each edit form
    all_fronts = [c["front"] for c in cards]

    for card in cards:
        front_text = card["front"]
        if search and search.lower() not in front_text.lower() and search.lower() not in card["back"].lower():
            continue

        with st.expander(f"**{front_text[:80]}{'...' if len(front_text) > 80 else ''}**"):
            # SNN state display
            w = card["weight"]
            review_count = card["review_count"]
            col1, col2 = st.columns(2)
            col1.metric("Memory Strength", f"{w:.3f}")
            col2.metric("Reviews", review_count)

            # Show existing [[link]] references as styled pills
            existing_refs = get_link_refs(card["front"] + " " + card["back"])
            if existing_refs:
                pills_html = " ".join(
                    f'<span style="background:#2a2a3e;color:#89b4fa;border:1px solid #45475a;'
                    f'border-radius:8px;padding:2px 9px;font-size:0.75rem;margin-right:4px">'
                    f'🔗 {ref}</span>'
                    for ref in existing_refs
                )
                st.markdown(
                    f'<div style="margin-bottom:8px">{pills_html}</div>',
                    unsafe_allow_html=True,
                )

            # Edit form
            with st.form(f"edit_{card['id']}"):
                new_front = st.text_area("Front", value=card["front"], height=80)
                new_back = st.text_area("Back", value=card["back"], height=80)
                new_tags = st.text_input("Tags", value=card["tags"])

                # Show link picker — options exclude the card itself
                other_fronts = [f for f in all_fronts if f != card["front"]]
                add_prereqs = st.multiselect(
                    "🔗 Add prerequisite links",
                    options=other_fronts,
                    help="Selected cards will be appended as [[link]] tokens to the back text.",
                )

                col_save, col_del = st.columns([3, 1])
                if col_save.form_submit_button("Save"):
                    back_final = new_back.strip()
                    for p in add_prereqs:
                        token = f"[[{p}]]"
                        if token not in back_final:
                            back_final += f"\n{token}"
                    update_card(conn, card["id"], new_front, back_final, new_tags)
                    st.success("Updated!")
                    st.rerun()
                if col_del.form_submit_button("🗑️ Delete", type="secondary"):
                    delete_card(conn, card["id"])
                    st.rerun()


def _bulk_import(conn, deck_id: int):
    st.subheader("Bulk Import")
    st.markdown("""
    Paste multiple cards at once. Each card is one line in the format:

    ```
    Front text | Back text
    ```

    Lines starting with `#` are treated as comments and skipped.
    """)

    with st.form("bulk_import"):
        raw = st.text_area(
            "Cards (one per line, front | back)",
            height=250,
            placeholder=(
                "# Chinese HSK vocabulary\n"
                "你好 | Hello / How are you\n"
                "谢谢 | Thank you\n"
                "# Neuroscience\n"
                "What does STDP stand for? | Spike-Timing-Dependent Plasticity"
            )
        )
        tags = st.text_input("Apply tags to all imported cards (optional)")
        if st.form_submit_button("Import Cards", type="primary"):
            lines = raw.strip().splitlines()
            added, skipped = 0, 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" not in line:
                    skipped += 1
                    continue
                parts = line.split("|", 1)
                front, back = parts[0].strip(), parts[1].strip()
                if front and back:
                    create_card(conn, deck_id, front, back, tags)
                    added += 1
                else:
                    skipped += 1

            st.success(f"Imported {added} card(s). Skipped {skipped} malformed line(s).")
            st.rerun()
