# pricing_helpers.py

from app import lookup_price, lookup_price_rows, gather_booking_choice, STORE_INFO, openai_client

def handle_price_intent(user_input: str, lead_state: dict):
    """
    Unified price intent handler:
    1. Try Google Sheet lookup.
    2. If found → quote + booking choice.
    3. If not found → AI fallback with sheet data.
    """
    lower = (user_input or "").lower()

    # 1) Try sheet lookup
    match = lookup_price(lower)
    if match:
        return gather_booking_choice(
            f"The current price for {match['Device']} {match['RepairType']} is ${match['Price']}. "
            "Would you like to book that?",
            lead_state
        )

    # 2) AI fallback with sheet data
    price_rows = lookup_price_rows() or []
    if not price_rows:
        return gather_booking_choice(
            "I couldn't access our price list right now. Would you like me to still book you in?",
            lead_state
        )

    sheet_text = "\n".join(f"{r['Device']} {r['RepairType']}: ${r['Price']}" for r in price_rows)
    system_prompt = (
        f"You are the receptionist for {STORE_INFO['name']} in {STORE_INFO['city']}. "
        f"Here is the current price list:\n{sheet_text}\n"
        "If you find a match, quote it. If not, say you'll check with a tech."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    out = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    text = out.choices[0].message.content.strip()

    return gather_booking_choice(
        text + " Would you like to book that?",
        lead_state
    )