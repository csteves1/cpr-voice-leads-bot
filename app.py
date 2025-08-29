import os, re, json, base64
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from intents import is_hours, is_location, is_phone, is_landmarks, is_directions
from pricing import lookup_price, lookup_price_rows
from repairq import get_inventory_by_sku, create_appointment
from calls import start_outbound_call
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime as dt, timedelta
import dateparser
from urllib.parse import quote_plus
from urllib.parse import urlencode

import logging
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.INFO)

def repeat_q(prompt, day="", time_slot="", **kwargs):
    # Preserve booking day/time slot unless overridden
    kwargs.setdefault("day", day)
    kwargs.setdefault("time_slot", time_slot)
    print(f"[Prompting] {prompt} | next_params={kwargs}")

    params = urlencode(kwargs)

    vrq = VoiceResponse()
    vrq.say(prompt)
    g = Gather(
        input="speech",
        action=f"/voice/outbound/lead/intake?{params}",
        method="POST",
        timeout=30,
        speech_timeout="auto",
        speech_model="phone_call"
    )
    vrq.append(g)
    return Response(str(vrq), media_type="application/xml")
def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

STORE_INFO = {
    "name": "CPR Cell Phone Repair",
    "city": "Myrtle Beach",
    "address": "1000 South Commons Drive, Myrtle Beach, SC 29588",
    "phone": "(843) 750-0449",
    "hours": "Monday to Saturday 9am-6pm, Sunday we are closed",
}

# Popular device models for validation and suggestions
DEVICE_MODELS = {
    "apple": [
        # iPhone
        "iPhone 6", "iPhone 6 Plus", "iPhone 6s", "iPhone 6s Plus",
        "iPhone 7", "iPhone 7 Plus", "iPhone 8", "iPhone 8 Plus",
        "iPhone X", "iPhone XR", "iPhone XS", "iPhone XS Max",
        "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max",
        "iPhone SE 2nd Gen", "iPhone SE 3rd Gen",
        "iPhone 12 Mini", "iPhone 12", "iPhone 12 Pro", "iPhone 12 Pro Max",
        "iPhone 13 Mini", "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max",
        "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
        "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
        # iPad
        "iPad 6th Gen", "iPad 7th Gen", "iPad 8th Gen", "iPad 9th Gen", "iPad 10th Gen",
        "iPad Air 3", "iPad Air 4", "iPad Air 5",
        "iPad Mini 5", "iPad Mini 6",
        "iPad Pro 9.7", "iPad Pro 10.5",
        "iPad Pro 11 1st Gen", "iPad Pro 11 2nd Gen", "iPad Pro 11 3rd Gen", "iPad Pro 11 4th Gen",
        "iPad Pro 12.9 2nd Gen", "iPad Pro 12.9 3rd Gen", "iPad Pro 12.9 4th Gen", "iPad Pro 12.9 5th Gen", "iPad Pro 12.9 6th Gen"
    ],
    "samsung": [
        # S series
        "Galaxy S8", "Galaxy S8 Plus", "Galaxy S9", "Galaxy S9 Plus",
        "Galaxy S10e", "Galaxy S10", "Galaxy S10 Plus",
        "Galaxy S20", "Galaxy S20 Plus", "Galaxy S20 Ultra",
        "Galaxy S21", "Galaxy S21 Plus", "Galaxy S21 Ultra",
        "Galaxy S22", "Galaxy S22 Plus", "Galaxy S22 Ultra",
        "Galaxy S23", "Galaxy S23 Plus", "Galaxy S23 Ultra",
        "Galaxy S24", "Galaxy S24 Plus", "Galaxy S24 Ultra",
        # Note
        "Galaxy Note 8", "Galaxy Note 9", "Galaxy Note 10", "Galaxy Note 10 Plus",
        "Galaxy Note 20", "Galaxy Note 20 Ultra",
        # A series
        "Galaxy A10e", "Galaxy A12", "Galaxy A13", "Galaxy A32", "Galaxy A42", "Galaxy A51", "Galaxy A52", "Galaxy A53", "Galaxy A54",
        # Foldables
        "Galaxy Z Flip 3", "Galaxy Z Flip 4", "Galaxy Z Flip 5",
        "Galaxy Z Fold 3", "Galaxy Z Fold 4", "Galaxy Z Fold 5"
    ],
    "lg": [
        "LG G4", "LG G5", "LG G6", "LG G7 ThinQ", "LG G8 ThinQ",
        "LG V20", "LG V30", "LG V35", "LG V40", "LG V50", "LG V60",
        "LG Stylo 4", "LG Stylo 5", "LG Stylo 6",
        "LG K40", "LG K51", "LG K92"
    ],
    "motorola": [
        "Moto G5", "Moto G6", "Moto G7",
        "Moto G Power 2020", "Moto G Power 2021", "Moto G Power 2022", "Moto G Power 2023",
        "Moto G Stylus 2020", "Moto G Stylus 2021", "Moto G Stylus 2022", "Moto G Stylus 2023",
        "Moto G Pure", "Moto G Play",
        "Moto Edge 2021", "Moto Edge 2022", "Moto Edge 2023", "Moto Edge Plus 2023",
        "Moto One 5G", "Moto One 5G Ace"
    ]
}

# ---------------------------
# Dynamic intake configuration
# ---------------------------

REQUIRED_FIELDS = [
    "day",
    "time_slot",
    "first_name",
    "last_name",
    "phone",
    "alt_phone",
    "email",
    "zip_code",
    "referral",
    "imei",
    "device_model",
    "passcode",
    "diagnostic"
]

FIELD_PROMPTS = {
    "day": "What day works best for your appointment?",
    "time_slot": "What time of day works best? Morning, early afternoon, or late afternoon?",
    "first_name": "What's your first name?",
    "last_name": "What's your last name?",
    "phone": "What's the best phone number to reach you?",
    "alt_phone": "Do you have an alternate phone number? If not, just say no.",
    "email": "What's your email address?",
    "zip_code": "What's your zip code?",
    "referral": "How did you hear about us?",
    "imei": "Please read your fifteen-digit I M E I number, or say skip if not available.",
    "device_model": "What device model is it?",
    "passcode": "Do you have a passcode for the device? If not, say no.",
    "diagnostic": "Briefly describe the issue with the device."
}

# ---------------------------
# Shared utilities
# ---------------------------

def is_skip(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in ["skip", "na", "n/a", "none"]) or t == "no"

def wants_yes(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in ["yes", "yep", "correct", "that's right", "looks good", "good to go"])

def wants_no(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in ["no", "nope", "not quite", "change", "fix", "edit"])

def build_lead_state(**kwargs) -> dict:
    return {field: kwargs.get(field, "") for field in REQUIRED_FIELDS}

def next_missing_field(lead_state: dict) -> str | None:
    for field in REQUIRED_FIELDS:
        if not lead_state.get(field):
            return field
    return None

def repeat_q(prompt, *, action_path="/voice/outbound/lead/intake", timeout=15, speech_timeout="auto", day="", time_slot="", stage="", reported_model="", **kwargs):
    # Preserve shared context across turns
    kwargs.setdefault("day", day)
    kwargs.setdefault("time_slot", time_slot)
    kwargs.setdefault("stage", stage)
    kwargs.setdefault("reported_model", reported_model)

    params = urlencode(kwargs)

    vrq = VoiceResponse()
    vrq.say(prompt)
    g = Gather(
        input="speech",
        action=f"{action_path}?{params}",
        method="POST",
        timeout=timeout,
        speech_timeout=speech_timeout,
        speech_model="phone_call"
    )
    vrq.append(g)
    return Response(str(vrq), media_type="application/xml")

def read_back_summary(lead_state: dict) -> str:
    def safe(v): return v if v and v != "NA" else "not provided"
    lines = [
        f"Day: {safe(lead_state.get('day'))}",
        f"Time: {safe(lead_state.get('time_slot'))}",
        f"First name: {safe(lead_state.get('first_name'))}",
        f"Last name: {safe(lead_state.get('last_name'))}",
        f"Primary phone: {safe(lead_state.get('phone'))}",
        f"Alternate phone: {safe(lead_state.get('alt_phone'))}",
        f"Email: {safe(lead_state.get('email'))}",
        f"Zip code: {safe(lead_state.get('zip_code'))}",
        f"Referral: {safe(lead_state.get('referral'))}",
        f"IMEI: {safe(lead_state.get('imei'))}",
        f"Device model: {safe(lead_state.get('device_model'))}",
        f"Passcode: {safe(lead_state.get('passcode'))}",
        f"Diagnostic: {safe(lead_state.get('diagnostic'))}",
    ]
    return "Here’s what I have: " + ". ".join(lines) + ". Does this look correct?"


def imei_stage_logic(answer: str, lead_state: dict, reported_model: str = ""):
    # Skip or missing → mark NA and continue
    if not answer or is_skip(answer):
        lead_state["imei"] = "NA"
        return None, lead_state, ""

    digits_only = extract_value(answer, "imei")  # should return 15-digit string or ""
    if len(digits_only) != 15:
        return "imei_reprompt", lead_state, ""

    imei_info = imei_lookup(digits_only)
    lead_state["imei"] = digits_only
    api_model = (imei_info.get("model") or "").strip()

    if api_model:
        existing = (lead_state.get("device_model") or "").strip()
        if existing and api_model.lower() != existing.lower():
            return "confirm_device_model", lead_state, api_model
        if not existing:
            lead_state["device_model"] = api_model

    return None, lead_state, ""


def handle_field_progression(answer: str, stage: str, reported_model: str, **current_values):
    lead_state = build_lead_state(**current_values)

    # Special confirmation for IMEI-reported model
    if stage == "confirm_device_model":
        if wants_yes(answer):
            lead_state["device_model"] = reported_model
            stage = ""
        elif wants_no(answer):
            return {"prompt": "Got it — what's the correct device model?", "stage": "device_model", "lead_state": lead_state}
        else:
            return {"prompt": f"I checked the IMEI and it shows {reported_model}. Is that correct?", "stage": "confirm_device_model", "lead_state": lead_state}

    # IMEI field
    if stage == "imei":
        imei_branch, lead_state, api_model = imei_stage_logic(answer, lead_state, reported_model)
        if imei_branch == "imei_reprompt":
            return {"prompt": "I only heard part of the number. Please read your complete fifteen-digit I M E I number, digit by digit.", "stage": "imei", "lead_state": lead_state, "timeout": 12}
        if imei_branch == "confirm_device_model":
            return {"prompt": f"I checked the IMEI and it shows {api_model}. Is that correct?", "stage": "confirm_device_model", "lead_state": lead_state, "reported_model": api_model}
        stage = ""

    # Normal field capture (including day/time)
    if stage and stage in REQUIRED_FIELDS:
        if not answer:
            return {"prompt": FIELD_PROMPTS[stage], "stage": stage, "lead_state": lead_state}
        if is_skip(answer):
            lead_state[stage] = "NA"
        else:
            parsed = extract_value(answer, stage) or answer
            lead_state[stage] = parsed
        stage = ""

    # Advance to next missing
    next_field = next_missing_field(lead_state)
    if next_field:
        kwargs = {"timeout": 12, "speech_timeout": "auto"} if next_field == "imei" else {}
        return {"prompt": FIELD_PROMPTS[next_field], "stage": next_field, "lead_state": lead_state, **kwargs}

    # Summary confirmation
    return {"prompt": read_back_summary(lead_state), "stage": "confirm_summary", "lead_state": lead_state}


async def proceed_to_quote_and_confirm(*, action_path="/voice/outbound/lead/intake", **lead_state):
    device = lead_state.get("device_model") or ""
    diag = lead_state.get("diagnostic") or ""
    price_row = lookup_price(device, diag)  # optional

    if price_row and price_row.get("price"):
        say = f"The current price for that repair is {price_row['price']}."
        summary = read_back_summary(lead_state)
        prompt = f"{say} {summary}"
        return repeat_q(prompt, action_path=action_path, stage="confirm_summary", **lead_state)

    try:
        create_appointment(lead_state)
        return repeat_q("Great — I’ve created your ticket. Would you like a text with directions to the store?",
                        action_path=action_path, stage="post_ticket", **lead_state)
    except Exception:
        return repeat_q("Thanks. I’m having trouble creating the ticket right now. I’ll save your details and a technician will follow up shortly.",
                        action_path=action_path, **lead_state)



# Flattened list for quick search
ALL_DEVICE_MODELS = [m for brand in DEVICE_MODELS.values() for m in brand]

app = FastAPI()

def say_and_listen(vr: VoiceResponse, text: str, action="/voice/inbound/process"):
    vr.say(text)
    g = Gather(
        input="speech",
        action=action,
        method="POST",
        timeout=15,
        speech_timeout="auto",
        speech_model="phone_call",
        hints="repair, screen, battery, iPhone, price, directions, appointment, hours, address, yes, no, morning, afternoon"
    )
    vr.append(g)
    return vr

import re
from urllib.parse import urlencode

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

def normalize_digits_spoken(s: str) -> str:
    # Handle "eight four three ..." and trailing punctuation
    w = re.sub(r"[^\w\s@.+-]", "", (s or "").lower())
    tokens = w.split()
    out = []
    for t in tokens:
        out.append(DIGIT_WORDS.get(t, t))
    return "".join(out)

def normalize_phone(s: str) -> str:
    digits = re.sub(r"\D", "", normalize_digits_spoken(s))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits

def is_valid_phone(s: str) -> bool:
    return len(normalize_phone(s)) == 10

def format_phone_usa(s: str) -> str:
    d = normalize_phone(s)
    return f"({d[0:3]}) {d[3:6]}-{d[6:10]}" if len(d) == 10 else s

def normalize_email(s: str) -> str:
    e = (s or "").strip().rstrip(".").lower()
    return re.sub(r"\s+", "", e)

def is_valid_email(e: str) -> bool:
    e = normalize_email(e)
    return re.match(r"^[^@]+@[^@]+\.[a-z]{2,}$", e, re.I) is not None

def luhn_checksum(number: str) -> int:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10

def is_valid_imei(s: str) -> bool:
    digits = re.sub(r"\D", "", s or "")
    return len(digits) == 15 and luhn_checksum(digits) == 0

def suggest_device_models(user_text: str, limit: int = 5):
    """Lightweight fuzzy match by token overlap and substring presence."""
    q = (user_text or "").lower().strip()
    if not q:
        return []
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for m in ALL_DEVICE_MODELS:
        m_lower = m.lower()
        m_tokens = set(re.findall(r"[a-z0-9]+", m_lower))
        overlap = len(q_tokens & m_tokens)
        score = overlap + (1 if q in m_lower or m_lower in q else 0)
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [m for _, m in scored[:limit]]

import os
import requests

def imei_lookup(imei: str) -> dict:
    """
    Look up IMEI details using IMEI.info API.
    Returns a dict with at least 'brand' and 'model' if found.
    """
    api_key = os.getenv("IMEI_INFO_API_KEY")
    if not api_key:
        raise RuntimeError("IMEI_INFO_API_KEY not set in environment")

    url = "https://api.imei.info/v1/imei"  # Example endpoint — check docs for exact path
    params = {"apikey": api_key, "imei": imei}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[IMEI Lookup ERROR] {e}")
        return {}

    # Adjust parsing based on actual API response structure
    return {
        "brand": data.get("brand", ""),
        "model": data.get("model", ""),
        "raw": data
    }
import re

def extract_value(answer: str, field: str) -> str:
    """Extracts the relevant value from a natural sentence based on field type."""
    answer = answer.strip()

    # If it's a name, just grab the last word (e.g., "my name is Chris" → "Chris")
    if field in ["first_name", "last_name"]:
        words = answer.split()
        return words[-1] if words else ""

    # If it's an email, use regex
    if field == "email":
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", answer)
        return match.group(0) if match else ""

    # If it's a phone number, grab digits
    if field in ["phone", "alt_phone"]:
        digits = re.sub(r"\D", "", answer)
        return digits if len(digits) >= 7 else ""

    # If it's a zip code, grab 5-digit number
    if field == "zip_code":
        match = re.search(r"\b\d{5}\b", answer)
        return match.group(0) if match else ""

    # If it's IMEI, grab 15-digit number
    if field == "imei":
        digits = re.sub(r"\D", "", answer)
        return digits if len(digits) == 15 else ""

    # Default fallback: return full answer
    return answer

# --- Secure Google Sheets helpers ---
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_sheets_credentials():
    # Prefer base64 env var for security (no file needed on Render)
    b64_creds = os.getenv("GOOGLE_SHEETS_CREDS_B64")
    if b64_creds:
        info = json.loads(base64.b64decode(b64_creds).decode("utf-8"))
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    # Fallback: file path (local dev)
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        return service_account.Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    raise RuntimeError("No Google Sheets credentials found.")

def save_mock_ticket(data: dict):
    try:
        creds = get_sheets_credentials()
        sheet_id = os.getenv("MOCK_TICKETS_SHEET_ID")
        if not sheet_id:
            raise RuntimeError("MOCK_TICKETS_SHEET_ID not set")

        service = build("sheets", "v4", credentials=creds)
        values = [[
            data.get("day", ""),
            data.get("time_slot", ""),
            data.get("first_name", ""),
            data.get("last_name", ""),
            data.get("phone", ""),
            data.get("alt_phone", ""),
            data.get("email", ""),
            data.get("zip_code", ""),
            data.get("referral", ""),
            data.get("imei", ""),
            data.get("device_model", ""),
            data.get("passcode", ""),
            data.get("diagnostic", ""),
            datetime.now().isoformat()
        ]]
        body = {"values": values}
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range="Tickets!A:M",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body
        ).execute()
        print("[Sheets] Ticket saved successfully")
    except Exception as e:
        print(f"[Sheets ERROR] {e}")

@app.post("/voice/inbound/process")
async def voice_process(req: Request):
    form = await req.form()
    user_input = (form.get("SpeechResult") or "").strip()
    lower = user_input.lower()
    vr = VoiceResponse()

    # Quick fixed intents
    if is_directions(lower):
        return Response(str(say_and_listen(vr, "Sure. What's your starting address or location?")), media_type="application/xml")
    if is_hours(lower):
        return Response(str(say_and_listen(vr, f"Our hours are {STORE_INFO['hours']}. Anything else I can help with?")), media_type="application/xml")
    if is_location(lower):
        return Response(str(say_and_listen(vr, f"We're at {STORE_INFO['address']}. Do you need directions?")), media_type="application/xml")
    if is_phone(lower):
        return Response(str(say_and_listen(vr, f"Our phone number is {STORE_INFO['phone']}. What else can I help with?")), media_type="application/xml")
    if is_landmarks(lower):
        return Response(str(say_and_listen(vr, "We’re near Goodwill and Lowe’s, in the strip with Chipotle, McAlister’s, Sport Clips, and UPS.")), media_type="application/xml")

    repair_keywords = [
        "repair","screen","battery","cracked","broken","device","phone","tablet",
        "hours","address","location","directions","price","quote","appointment"
    ]
    if not any(t in lower for t in repair_keywords):
        return Response(
            str(say_and_listen(vr, "I can help with repairs, pricing, or booking. Is your question about a device or repair?")),
            media_type="application/xml"
        )

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # --- Price-related safe match ---
        if any(t in lower for t in repair_keywords):
            detected = None
            spoken = norm_text(lower)

            for row in lookup_price_rows():
                dev_norm = norm_text(row.get("Device", ""))
                rep_norm = norm_text(row.get("RepairType", ""))
                if not dev_norm or not rep_norm:
                    continue

                # Drop brand word for tail matching (still requires full tail match)
                parts = dev_norm.split()
                tail = " ".join(parts[1:]) if parts and parts[0] in [
                    "samsung","galaxy","apple","iphone","google","pixel"
                ] else dev_norm

                if rep_norm in spoken and (dev_norm in spoken or tail in spoken):
                    detected = row
                    break

            if detected:
                vr.say(f"The current price for {detected['Device']} {detected['RepairType']} is ${detected['Price']}.")
                return Response(str(say_and_listen(vr, "Anything else I can help with?")), media_type="application/xml")
            else:
                # No safe match → re-ask
                return Response(
                    str(say_and_listen(
                        vr,
                        "Could you tell me the exact device model, like 'Samsung Galaxy S23 Ultra' or 'iPhone 13 Pro'?",
                        action="/voice/inbound/process"
                    )),
                    media_type="application/xml"
                )

        # --- Non-price branch → LLM fallback ---
        system_prompt = f"""
        You are the receptionist for {STORE_INFO['name']} in {STORE_INFO['city']}.
        Stay strictly on store/services/repairs. Keep answers to 1–3 sentences.
        If asked for a price and you don't know it, say you'll check with a tech.
        """
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        out = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
        text = out.choices[0].message.content.strip()
        return Response(str(say_and_listen(vr, text)), media_type="application/xml")

    except Exception:
        return Response(
            str(say_and_listen(vr, "Sorry, I’m having trouble right now. Could you try again shortly?")),
            media_type="application/xml"
        )

from datetime import datetime, timedelta
import pytz
import re
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

EASTERN = pytz.timezone("US/Eastern")

# Persistent job store (SQLite file on your Render disk at /var/data)
jobstores = {
    # NOTE: Four slashes after sqlite: for absolute path
    'default': SQLAlchemyJobStore(url='sqlite:////var/data/scheduled_jobs.sqlite')
}

# Set up basic logging so you can see job activity in Render logs
logging.basicConfig(level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.INFO)

# Create and start the scheduler when the app starts
scheduler = BackgroundScheduler(jobstores=jobstores, timezone=EASTERN)
scheduler.start()
logging.info("[Scheduler] Started with job store at /var/data/scheduled_jobs.sqlite")

from urllib.parse import urlencode
from twilio.rest import Client
import os

BASE_URL = "https://cpr-voice-leads-bot.onrender.com"  # your Render URL

def start_outbound_call(lead: dict):
    # Split name into first/last
    name_parts = lead.get("name", "").strip().split(" ", 1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    # Fetch Twilio number at runtime
    from_number = os.getenv("TWILIO_NUMBER")
    print(f"[DEBUG] TWILIO_NUMBER in start_outbound_call: {from_number!r}")
    if not from_number:
        raise RuntimeError("TWILIO_NUMBER env var is not set or empty — cannot initiate outbound call.")

    # Build query params for initial stage of verification flow
    params = urlencode({
        "stage": "intro",
        "first_name": first_name,
        "last_name": last_name,
        "phone": lead.get("phone", ""),
        "email": lead.get("email", ""),
        "device_model": lead.get("device", ""),
        "referral": lead.get("source", ""),
        "diagnostic": lead.get("repair_type", ""),
        "imei": lead.get("imei", "")
    })

    # Pick the correct flow URL
    verify_url = f"{BASE_URL}/voice/outbound/lead/intake?{params}"

    print(f"[DEBUG] Placing call to {lead.get('phone')} from {from_number} with URL {verify_url}")

    # Twilio REST client
    client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

    call = client.calls.create(
        to=lead.get("phone"),
        from_=from_number,
        url=verify_url
    )

    print(f"[DEBUG] Outbound call SID: {call.sid}")
    return call

def should_call_now():
    now = datetime.now(EASTERN)
    start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    end = now.replace(hour=18, minute=0, second=0, microsecond=0)
    return start <= now <= end


def schedule_lead_call(lead):
    now = datetime.now(EASTERN)
    if should_call_now():
        print("[Lead Call] Within business hours — calling immediately.")
        start_outbound_call(lead)
    else:
        # For quick‑fire test: run 2 minutes from now
        next_call_time = now + timedelta(minutes=2)
        
        print(f"[Lead Call] Outside business hours — scheduling for {next_call_time.isoformat()}")

        # Make a safe job id so duplicates get replaced if same lead/time
        safe_phone = re.sub(r"\D", "", lead.get("phone", ""))
        job_id = f"lead_call_{safe_phone}_{int(next_call_time.timestamp())}"

        scheduler.add_job(
            start_outbound_call,
            'date',
            run_date=next_call_time,
            args=[lead],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=600  # 10‑minute grace if app is down briefly
        )

@app.post("/webhooks/repairq/lead")
async def repairq_lead(req: Request):
    # ✅ Safeguard: check Twilio number is set before scheduling
    twilio_num = os.getenv("TWILIO_NUMBER")
    if not twilio_num:
        print("[ERROR] TWILIO_NUMBER env var is missing or empty — skipping outbound call")
        return {"ok": False, "error": "Missing Twilio number in server environment"}

    payload = await req.json()
    lead = {
        "name": payload.get("name", ""),
        "phone": payload.get("phone", ""),
        "device": payload.get("device", ""),
        "repair_type": payload.get("repair_type", ""),
        "source": payload.get("source", "online_lead"),
    }
    schedule_lead_call(lead)
    return {"ok": True}

@app.post("/voice/outbound/lead")
@app.get("/voice/outbound/lead")
async def voice_outbound_lead(req: Request, **state):
    """Entry point for outbound calls — hand straight to intake AI engine."""
    # state here comes from query params (RepairQ lead info)
    return repeat_q(
        "Hi, let's get a few quick details to help with your booking.",
        action_path="/voice/outbound/lead/intake",
        stage=state.get("stage", ""), 
        **{f: state.get(f, "") for f in REQUIRED_FIELDS}
    )
@app.post("/voice/outbound/lead/process")
async def outbound_lead_process(req: Request, **state):
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()

    result = handle_field_progression(
        answer=answer,
        stage=state.get("stage", ""),
        reported_model=state.get("reported_model", ""),
        **{f: state.get(f, "") for f in REQUIRED_FIELDS}
    )
    if result and result.get("prompt"):
        return repeat_q(
            result["prompt"],
            action_path="/voice/outbound/lead/process",
            stage=result.get("stage", ""),
            reported_model=result.get("reported_model", ""),
            timeout=25,
            speech_timeout="auto",
            **result["lead_state"]
        )
    return repeat_q("Sorry, I didn’t catch that. Could you repeat?",
                    action_path="/voice/outbound/lead/process",
                    **state)


@app.post("/voice/outbound/lead/intake")
async def lead_intake(
    req: Request,
    day: str = "",
    time_slot: str = "",
    first_name: str = "",
    last_name: str = "",
    phone: str = "",
    alt_phone: str = "",
    email: str = "",
    zip_code: str = "",
    referral: str = "",
    imei: str = "",
    device_model: str = "",
    passcode: str = "",
    diagnostic: str = "",
    stage: str = "",
    reported_model: str = ""
):
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()

    # 1) Final "all fields complete" confirmation
    if stage == "confirm_summary":
        if wants_yes(answer):
            # All fields are done, so now you can save to RepairQ or Google Sheets
            ticket = build_lead_state(
                day=day, time_slot=time_slot,
                first_name=first_name, last_name=last_name,
                phone=phone, alt_phone=alt_phone, email=email,
                zip_code=zip_code, referral=referral,
                imei=imei, device_model=device_model,
                passcode=passcode, diagnostic=diagnostic
            )
            try:
                save_mock_ticket(ticket)
            except Exception as e:
                print(f"[ERROR saving ticket] {e}")

            vr_final = VoiceResponse()
            vr_final.say(f"Thanks {first_name}, you're booked for the {day} at {time_slot}. I've saved your ticket.")
            vr_final.say("I can also answer general phone repair questions if you have any.")
            # Optional: redirect into inbound Q&A
            vr_final.redirect("/voice/inbound")
            params = urlencode(ticket)
            vr_final.redirect(f"/voice/inbound/verify?{params}")
            return Response(str(vr_final), media_type="application/xml")

        if wants_no(answer):
            return repeat_q(
                "Which field do you want to change? For example: day, time, email, or device model.",
                action_path="/voice/outbound/lead/intake",
                stage="correction_target",
                **build_lead_state(
                    day=day, time_slot=time_slot,
                    first_name=first_name, last_name=last_name,
                    phone=phone, alt_phone=alt_phone, email=email,
                    zip_code=zip_code, referral=referral,
                    imei=imei, device_model=device_model,
                    passcode=passcode, diagnostic=diagnostic
                )
            )

        # Neither a clear yes nor no → re-read summary
        return repeat_q(
            read_back_summary(build_lead_state(
                day=day, time_slot=time_slot,
                first_name=first_name, last_name=last_name,
                phone=phone, alt_phone=alt_phone, email=email,
                zip_code=zip_code, referral=referral,
                imei=imei, device_model=device_model,
                passcode=passcode, diagnostic=diagnostic
            )),
            action_path="/voice/outbound/lead/intake",
            stage="confirm_summary",
            **build_lead_state(
                day=day, time_slot=time_slot,
                first_name=first_name, last_name=last_name,
                phone=phone, alt_phone=alt_phone, email=email,
                zip_code=zip_code, referral=referral,
                imei=imei, device_model=device_model,
                passcode=passcode, diagnostic=diagnostic
            )
        )

    # 1) Final "all fields complete" confirmation
    if stage == "confirm_summary":
        lead_state = build_lead_state(
            day=day, time_slot=time_slot,
            first_name=first_name, last_name=last_name,
            phone=phone, alt_phone=alt_phone, email=email,
            zip_code=zip_code, referral=referral,
            imei=imei, device_model=device_model,
            passcode=passcode, diagnostic=diagnostic
        )

        if wants_yes(answer):
            # Save the completed ticket to RepairQ or Google Sheets
            try:
                save_mock_ticket(lead_state)
            except Exception as e:
                print(f"[ERROR saving ticket] {e}")

            # Thank caller & transition into post‑booking Q&A
            vr_final = VoiceResponse()
            vr_final.say(
                f"Thanks {lead_state['first_name']}, you're booked for the "
                f"{lead_state['day']} at {lead_state['time_slot']}. I've saved your ticket."
            )
            vr_final.say("I can also answer general phone repair questions if you have any.")
            params = urlencode(lead_state)
            vr_final.redirect(f"/voice/inbound/verify?{params}")
            return Response(str(vr_final), media_type="application/xml")

        if wants_no(answer):
            return repeat_q(
                "Which field do you want to change? For example: day, time, email, or device model.",
                action_path="/voice/outbound/lead/intake",
                stage="correction_target",
                **lead_state
            )

        # Unclear answer → re‑read the summary
        return repeat_q(
            read_back_summary(lead_state),
            action_path="/voice/outbound/lead/intake",
            stage="confirm_summary",
            **lead_state
        )

    # 2) Field‑level correction target
    if stage == "correction_target":
        field_map = {
            "day": "day", "date": "day",
            "time": "time_slot", "time slot": "time_slot",
            "first": "first_name", "first name": "first_name",
            "last": "last_name", "last name": "last_name",
            "phone": "phone", "primary": "phone",
            "alternate": "alt_phone", "alt": "alt_phone",
            "email": "email",
            "zip": "zip_code", "zip code": "zip_code",
            "referral": "referral",
            "imei": "imei",
            "model": "device_model", "device": "device_model",
            "passcode": "passcode", "code": "passcode",
            "diagnostic": "diagnostic", "issue": "diagnostic"
        }
        key = (answer or "").lower().strip()
        target = field_map.get(key)
        if not target:
            for k, v in field_map.items():
                if k in key:
                    target = v
                    break

        lead_state = build_lead_state(
            day=day, time_slot=time_slot,
            first_name=first_name, last_name=last_name,
            phone=phone, alt_phone=alt_phone, email=email,
            zip_code=zip_code, referral=referral,
            imei=imei, device_model=device_model,
            passcode=passcode, diagnostic=diagnostic
        )

        if not target:
            return repeat_q(
                "Sorry, which field should I change? For example say: day, time, email, phone, or device model.",
                action_path="/voice/outbound/lead/intake",
                stage="correction_target",
                **lead_state
            )

        # Ask for the new value
        return repeat_q(
            f"Okay, what's the correct {target.replace('_', ' ')}?",
            action_path="/voice/outbound/lead/intake",
            stage=target,
            **lead_state
        )

    # 3) Core progression — ask next missing field, or reach summary if all done
    result = handle_field_progression(
        answer=answer,
        stage=stage,
        reported_model=reported_model,
        day=day, time_slot=time_slot,
        first_name=first_name, last_name=last_name,
        phone=phone, alt_phone=alt_phone, email=email,
        zip_code=zip_code, referral=referral,
        imei=imei, device_model=device_model,
        passcode=passcode, diagnostic=diagnostic
    )
    if result and result.get("prompt") and result.get("lead_state"):
        return repeat_q(
            result["prompt"],
            action_path="/voice/outbound/lead/intake",
            stage=result.get("stage", ""),
            reported_model=result.get("reported_model", reported_model),
            timeout=result.get("timeout", 15),
            speech_timeout=result.get("speech_timeout", "auto"),
            **result["lead_state"]
        )
    
# 4) Failsafe — only runs if we somehow fall through all above
    return repeat_q("Sorry, I didn’t catch that. Could you repeat?",
                    action_path="/voice/outbound/lead/intake")

# === NEW: General phone repair Q&A after booking ===
@app.post("/voice/inbound")
async def voice_inbound(req: Request):
    """Entry point after booking to handle general phone repair questions."""
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()
    print(f"[Inbound Q&A] answer={answer}")

    vr = VoiceResponse()

    # First turn – greet and invite a question
    if not answer:
        g = Gather(
            input="speech",
            action="/voice/inbound",
            method="POST",
            timeout=15,
            speech_timeout="auto",
            speech_model="phone_call"
        )
        g.say("What would you like to know? You can ask about our hours, our location, how long repairs take, or any other phone repair question.")
        vr.append(g)
        return Response(str(vr), media_type="application/xml")

    lower = answer.lower()

    # Quick fixed answers for common topics
    if "hour" in lower:
        vr.say(f"Our hours are {STORE_INFO['hours']}.")
    elif "location" in lower or "address" in lower or "where" in lower:
        vr.say(f"We're at {STORE_INFO['address']}.")
    elif "how long" in lower or "long" in lower or "turnaround" in lower:
        vr.say("We usually quote repairs at 1 to 2 hours once we begin work.")
    else:
        # Hand off to your AI repair Q&A flow (already implemented in /voice/inbound/process)
        vr.redirect("/voice/inbound/process")
        return Response(str(vr), media_type="application/xml")

    # After answering, loop back so they can ask more
    vr.redirect("/voice/inbound")
    return Response(str(vr), media_type="application/xml")

@app.post("/voice/inbound/verify")
async def inbound_verify(
    req: Request,
    day: str = "", time_slot: str = "",
    first_name: str = "", last_name: str = "",
    phone: str = "", alt_phone: str = "", email: str = "",
    zip_code: str = "", referral: str = "",
    imei: str = "", device_model: str = "",
    passcode: str = "", diagnostic: str = "",
    stage: str = "",
    reported_model: str = ""
):
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()

    lead_state = build_lead_state(
        day=day, time_slot=time_slot,
        first_name=first_name, last_name=last_name,
        phone=phone, alt_phone=alt_phone, email=email,
        zip_code=zip_code, referral=referral,
        imei=imei, device_model=device_model,
        passcode=passcode, diagnostic=diagnostic
    )

    # Handle correction stage (user wants to change a specific field)
    if stage == "correction_target":
        field_map = {
            "day": "day", "date": "day",
            "time": "time_slot", "time slot": "time_slot",
            "first": "first_name", "first name": "first_name",
            "last": "last_name", "last name": "last_name",
            "phone": "phone", "primary": "phone",
            "alternate": "alt_phone", "alt": "alt_phone",
            "email": "email",
            "zip": "zip_code", "zip code": "zip_code",
            "referral": "referral",
            "imei": "imei",
            "model": "device_model", "device": "device_model",
            "passcode": "passcode", "code": "passcode",
            "diagnostic": "diagnostic", "issue": "diagnostic"
        }
        key = (answer or "").lower().strip()
        target = field_map.get(key)
        if not target:
            for k, v in field_map.items():
                if k in key:
                    target = v
                    break
        if not target:
            return repeat_q(
                "Sorry, which field should I change?",
                action_path="/voice/inbound/verify",
                stage="correction_target",
                timeout=30, speech_timeout="auto",
                **lead_state
            )
        return repeat_q(
            f"Okay, what's the correct {target.replace('_', ' ')}?",
            action_path="/voice/inbound/verify",
            stage=target,
            timeout=30, speech_timeout="auto",
            **lead_state
        )

    # First time in? Read back summary
    if not stage:
        return repeat_q(
            read_back_summary(lead_state),
            action_path="/voice/inbound/verify",
            stage="confirm_summary",
            timeout=30, speech_timeout="auto",
            **lead_state
        )

    # Summary confirmation
    if stage == "confirm_summary":
        if wants_yes(answer):
            try:
                save_mock_ticket(lead_state)  # Google Sheets / RepairQ push
            except Exception as e:
                print(f"[ERROR saving ticket] {e}")
            vr_final = VoiceResponse()
            vr_final.say("Thanks, your booking is confirmed.")
            vr_final.say("Would you like to ask any other questions?")
            vr_final.redirect("/voice/inbound")
            return Response(str(vr_final), media_type="application/xml")
        if wants_no(answer):
            return repeat_q(
                "Which field would you like to change?",
                action_path="/voice/inbound/verify",
                stage="correction_target",
                timeout=30, speech_timeout="auto",
                **lead_state
            )
        return repeat_q(
            read_back_summary(lead_state),
            action_path="/voice/inbound/verify",
            stage="confirm_summary",
            timeout=30, speech_timeout="auto",
            **lead_state
        )

    # Otherwise, let the AI progression handle any remaining/missing fields
    result = handle_field_progression(
        answer=answer,
        stage=stage,
        reported_model=reported_model,
        **lead_state
    )

    # Save progress on every update
    try:
        save_mock_ticket(result["lead_state"])
    except Exception as e:
        print(f"[ERROR saving progress] {e}")

    if result and result.get("prompt"):
        return repeat_q(
            result["prompt"],
            action_path="/voice/inbound/verify",
            stage=result.get("stage", ""),
            reported_model=result.get("reported_model", reported_model),
            timeout=30, speech_timeout="auto",
            **result["lead_state"]
        )

    # Failsafe
    return repeat_q(
        "Sorry, I didn’t catch that. Could you repeat?",
        action_path="/voice/inbound/verify",
        timeout=30, speech_timeout="auto",
        **lead_state
    )

# Health check
@app.get("/health")
def health():
    return {"ok": True}

# Test‑only: get inventory for a given SKU
@app.get("/dev/inventory/{sku}")
def dev_inventory(sku: str):
    try:
        data = get_inventory_by_sku(sku)
        return {"sku": sku, "inventory_data": data}
    except Exception as e:
        return {"error": str(e)}

# Test‑only: book a dummy appointment in RepairQ
@app.post("/dev/appointment")
def dev_appointment():
    try:
        fake_customer = {"name": "Test User", "phone": "+18435551234"}
        resp = create_appointment(fake_customer, "Pixel 6", "Screen Repair", "2025-08-28T10:00:00Z")
        return {"created": resp}
    except Exception as e:
        return {"error": str(e)}