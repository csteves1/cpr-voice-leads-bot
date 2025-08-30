# pricing_helpers.py

def handle_price_intent(
    user_input,
    lead_state,
    lookup_price_fn,
    lookup_price_rows_fn,
    gather_booking_choice_fn,
    store_info,
    openai_client
):
    lower = (user_input or "").lower()

    match = lookup_price_fn(lower)
    if match:
        return gather_booking_choice_fn(
            f"The current price for {match['Device']} {match['RepairType']} is ${match['Price']}. "
            "Would you like to book that?",
            lead_state
        )

    price_rows = lookup_price_rows_fn() or []
    if not price_rows:
        return gather_booking_choice_fn(
            "I couldn't access our price list right now. Would you like me to still book you in?",
            lead_state
        )

    sheet_text = "\n".join(f"{r['Device']} {r['RepairType']}: ${r['Price']}" for r in price_rows)
    system_prompt = (
        f"You are the receptionist for {store_info['name']} in {store_info['city']}. "
        f"Here is the current price list:\n{sheet_text}\n"
        "If you find a match, quote it. If not, say you'll check with a tech."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    out = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    text = out.choices[0].message.content.strip()

    return gather_booking_choice_fn(
        text + " Would you like to book that?",
        lead_state
    )