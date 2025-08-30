# shared.py

# === Store Info ===
STORE_INFO = {
    "name": "CPR Cell Phone Repair",
    "city": "Myrtle Beach",
    "address": "1000 South Commons Drive, Unit 103, Myrtle Beach, SC",
    "hours": "Mon–Sat 9am–6pm",
    "phone": "843-555-1234"
}

# === Price Lookup ===
def lookup_price(spoken: str):
    from difflib import get_close_matches
    import os, json, gspread

    creds_json = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        return None

    creds = gspread.service_account_from_dict(json.loads(creds_json))
    sheet = creds.open_by_key(os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")).sheet1
    rows = sheet.get_all_records()

    spoken_lower = spoken.lower()
    for row in rows:
        device = row.get("Device", "").lower()
        repair = row.get("RepairType", "").lower()
        if device in spoken_lower and repair in spoken_lower:
            return row

    # Fuzzy match fallback
    all_phrases = [f"{r['Device']} {r['RepairType']}" for r in rows]
    match = get_close_matches(spoken_lower, all_phrases, n=1, cutoff=0.7)
    if match:
        for row in rows:
            if f"{row['Device']} {row['RepairType']}".lower() == match[0]:
                return row
    return None

def lookup_price_rows():
    import os, json, gspread

    creds_json = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        return []

    creds = gspread.service_account_from_dict(json.loads(creds_json))
    sheet = creds.open_by_key(os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")).sheet1
    return sheet.get_all_records()

# === Booking Response ===
def gather_booking_choice(text: str, lead_state: dict):
    from twilio.twiml.voice_response import VoiceResponse, Gather
    from urllib.parse import urlencode

    vr = VoiceResponse()
    g = Gather(
        input="speech",
        action=f"/voice/inbound/post_booking_choice?{urlencode(lead_state)}",
        method="POST",
        timeout=25,
        speech_timeout="auto",
        speech_model="phone_call"
    )
    g.say(text)
    vr.append(g)
    return vr

# === OpenAI Client ===
from openai import OpenAI
import os

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))