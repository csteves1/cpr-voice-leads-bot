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

TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
BASE_URL = "https://cpr-voice-leads-bot.onrender.com"  # your Render URL

def start_outbound_call(lead: dict):
    # Split name into first/last if possible
    name_parts = lead.get("name", "").strip().split(" ", 1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    # Build query params to seed verification flow
    params = urlencode({
        "stage": "intro",
        "day": "",  # fill if you have it
        "time_slot": "",  # fill if you have it
        "first_name": first_name,
        "last_name": last_name,
        "phone": lead.get("phone", ""),
        "email": lead.get("email", ""),  # include if present in lead
        "device_model": lead.get("device", ""),
        "diagnostic": lead.get("repair_type", ""),
        "imei": lead.get("imei", "")
    })

    verify_url = f"{BASE_URL}/voice/inbound/verify?{params}"

    print(f"[Outbound Call] Starting call to {lead.get('phone')} with URL: {verify_url}")

    # Twilio REST client
    client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    call = client.calls.create(
        to=lead["phone"],
        from_=TWILIO_NUMBER,
        url=verify_url
    )
    print(f"[Outbound Call] Twilio call SID: {call.sid}")


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
def lead_intro(name: str = "", device: str = "", repair: str = "", source: str = "online_lead"):
    vr = VoiceResponse()
    if source == "website_appt":
        vr.say(f"Hi {name}, this is {STORE_INFO['name']} confirming your appointment request.")
    else:
        vr.say(f"Hi {name}, this is {STORE_INFO['name']}. I saw your interest in a {repair} for your {device}.")
    p = lookup_price(device, repair)
    if p and p.get("price"):
        vr.say(f"The current price for that repair is {p['price']}.")
        if p.get("sku"):
            try:
                inv = get_inventory_by_sku(p["sku"])
                # Adjust to your schema; set a safe default message:
                vr.say("We have the part available.")
            except Exception:
                vr.say("I’m checking part availability.")
    return Response(str(say_and_listen(vr, "Would you like to book an appointment?", action="/voice/outbound/lead/process")),
                    media_type="application/xml")

# --- UPDATED: more forgiving, routes to intake flow ---
@app.post("/voice/outbound/lead/process")
async def lead_process(req: Request):
    form = await req.form()
    speech = (form.get("SpeechResult") or "").lower()
    vr = VoiceResponse()
    yes_words = ["yes","book","schedule","sure","yeah","yep","please","ok","okay","yup","why not"]
    if any(w in speech for w in yes_words):
        return Response(str(say_and_listen(vr, "Great. Which day would you like to come in?", action="/voice/outbound/lead/day")),
                        media_type="application/xml")
    else:
        vr.say("No problem. If you need anything later, just call us.")
        vr.redirect("/voice/inbound")  # back to Q&A mode
        return Response(str(vr), media_type="application/xml")


@app.post("/voice/outbound/lead/day")
async def lead_day(req: Request):
    form = await req.form()
    spoken = (form.get("SpeechResult") or "").strip()
    print(f"[Booking Day] Caller said: {spoken}")

    if not spoken:
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "Sorry, I didn’t catch that. Which day would you like to come in?",
                action="/voice/outbound/lead/day"
            )),
            media_type="application/xml"
        )

    parsed = dateparser.parse(spoken)
    if not parsed:
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "Could you say a day like Friday or a date like August 29th?",
                action="/voice/outbound/lead/day"
            )),
            media_type="application/xml"
        )

    weekday = parsed.strftime("%A").lower()
    valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    if weekday not in valid_days:
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "We’re open Monday through Saturday. What day works best for you?",
                action="/voice/outbound/lead/day"
            )),
            media_type="application/xml"
        )

    # Try extracting time directly from parsed datetime
    time_obj = parsed.time()
    open_time = dt.strptime("9:00 AM", "%I:%M %p").time()
    close_time = dt.strptime("6:00 PM", "%I:%M %p").time()

    if time_obj and open_time <= time_obj <= close_time:
        time_str = time_obj.strftime("%I:%M %p").lstrip("0")
        encoded_time = quote_plus(time_str)  # encode after creating it
        print(f"[Parsed Combined] Day={weekday}, Time={time_str}")
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                f"Got it. Let's get you booked for {weekday.capitalize()} at {time_str}. What's your first name?",
                action=f"/voice/outbound/lead/intake?day={weekday.capitalize()}&time_slot={encoded_time}"
            )),
            media_type="application/xml"
        )

    # No valid/within-hours time in phrase → ask separately
    return Response(
        str(say_and_listen(
            VoiceResponse(),
            "What time works best for you? We’re open 9 A.M. to 6 P.M.",
            action=f"/voice/outbound/lead/time?day={weekday.capitalize()}"
        )),
        media_type="application/xml"
    )


@app.post("/voice/outbound/lead/time")
async def lead_time(req: Request, day: str = ""):
    form = await req.form()
    spoken_time = (form.get("SpeechResult") or "").strip()
    print(f"[Booking Time] day={day}, spoken_time={spoken_time}")

    if not spoken_time:
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "Sorry, I didn’t catch that. What time works best for you between 9 A.M. and 6 P.M.?",
                action=f"/voice/outbound/lead/time?day={day}"
            )),
            media_type="application/xml"
        )

    # Normalize common speech-to-text quirks
    cleaned = (
        spoken_time.lower()
        .replace(".", "")
        .replace("a m", "am")
        .replace("p m", "pm")
        .replace("a.m", "am")
        .replace("p.m", "pm")
        .replace(" o clock", "")
        .strip()
    )
    print(f"[Time Normalized] Raw='{spoken_time}' → Cleaned='{cleaned}'")

    # Try parsing with dateparser
    parsed = dateparser.parse(cleaned)
    if not parsed:
        print(f"[Time Parsing Failed] Cleaned='{cleaned}' → Parsed=None")
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "Could you say a specific time like 10 A.M. or 1:30 P.M.? We’re open 9 A.M. to 6 P.M.",
                action=f"/voice/outbound/lead/time?day={day}"
            )),
            media_type="application/xml"
        )

    time_obj = parsed.time()
    open_time = dt.strptime("9:00 AM", "%I:%M %p").time()
    close_time = dt.strptime("6:00 PM", "%I:%M %p").time()

    if not (open_time <= time_obj <= close_time):
        print(f"[Time Out of Range] Parsed={time_obj}")
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                "Our hours are 9 A.M. to 6 P.M. What time within those hours works for you?",
                action=f"/voice/outbound/lead/time?day={day}"
            )),
            media_type="application/xml"
        )

    time_str = time_obj.strftime("%I:%M %p").lstrip("0")
    encoded_time = quote_plus(time_str)  # encode here, not earlier
    print(f"[Time Parsed] Final={time_str}")

    return Response(
        str(say_and_listen(
            VoiceResponse(),
            f"Got it. Let's get you booked for {day} at {time_str}. What's your first name?",
            action=f"/voice/outbound/lead/intake?day={day}&time_slot={encoded_time}"
        )),
        media_type="application/xml"
    )
    
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
    diagnostic: str = ""
):
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()
    print(f"[DEBUG] Intake start: day={day}, time_slot={time_slot}")
    print(f"[Intake] day={day}, time_slot={time_slot}, first_name={first_name}, last_name={last_name}, "
          f"phone={phone}, alt_phone={alt_phone}, email={email}, zip_code={zip_code}, referral={referral}, "
          f"imei={imei}, device_model={device_model}, passcode={passcode}, diagnostic={diagnostic}, answer={answer}")

    from urllib.parse import urlencode

    def repeat_q(prompt, **kwargs):
        kwargs.setdefault("day", day)
        kwargs.setdefault("time_slot", time_slot)
        print(f"[Prompting] {prompt} | next_params={kwargs}")
        params = urlencode(kwargs)  # ✅ encodes spaces, punctuation, etc.
        vrq = VoiceResponse()
        vrq.say(prompt)
        g = Gather(
            input="speech",
            action=f"/voice/outbound/lead/intake?{params}",
            method="POST",
            timeout=15,
            speech_timeout="auto",
            speech_model="phone_call"
        )
        vrq.append(g)
        return Response(str(vrq), media_type="application/xml")
    # Step-by-step checks
    if not first_name:
        if not answer:
            return repeat_q("What's your first name?")
        return repeat_q("Thanks. What's your last name?", first_name=answer)

    if not last_name:
        if not answer:
            return repeat_q("What's your last name?", first_name=first_name)
        return repeat_q("What's the best phone number to reach you?", first_name=first_name, last_name=answer)

    if not phone:
        if not answer:
            return repeat_q("What's the best phone number to reach you?", first_name=first_name, last_name=last_name)
        return repeat_q("Do you have an alternate phone number? If not, just say no.",
                        first_name=first_name, last_name=last_name, phone=answer)

    if not alt_phone:
        if not answer:
            return repeat_q("Do you have an alternate phone number? If not, just say no.",
                            first_name=first_name, last_name=last_name, phone=phone)
        return repeat_q("What's your email address?",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=answer)

    if not email:
        if not answer:
            return repeat_q("What's your email address?",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone)
        return repeat_q("What's your zip code?",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=answer)

    if not zip_code:
        if not answer:
            return repeat_q("What's your zip code?",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email)
        return repeat_q("How did you hear about us?",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                        email=email, zip_code=answer)

    if not referral:
        if not answer:
            return repeat_q("How did you hear about us?",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                            email=email, zip_code=zip_code)
        return repeat_q("Can you provide the device IMEI? If not, just say no.",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                        email=email, zip_code=zip_code, referral=answer)

    if not imei:
        lower = answer.lower()
        digits_only = re.sub(r"\D", "", answer)

        # Handle skip phrases
        if any(phrase in lower for phrase in ["no", "nah", "none", "don't have", "do not have", "not available", "skip"]):
            return repeat_q("No problem. Do you have a passcode for the device? If not, say no.",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                            email=email, zip_code=zip_code, referral=referral, imei="", device_model="")

        # Handle stall phrases
        if any(phrase in lower for phrase in ["hold on", "hang on", "give me a minute", "one sec", "just a second"]):
            return repeat_q("Sure, take your time. When you're ready, please read out the fifteen-digit I M E I number clearly, digit by digit.",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                            email=email, zip_code=zip_code, referral=referral, imei="", device_model="")


        # Validate IMEI
        if len(digits_only) == 15:
            return repeat_q("Got it. Do you have a passcode for the device? If not, say no.",
                            first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                            email=email, zip_code=zip_code, referral=referral, imei=digits_only, device_model="")

        # Incomplete or unclear
        return repeat_q("I only heard part of the number. Please read your complete fifteen-digit I M E I number, digit by digit.",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                        email=email, zip_code=zip_code, referral=referral, imei="", device_model="")

    if not passcode:
        passcode_value = "" if answer.lower() in ["no", "nah", "none"] else answer
        return repeat_q("Briefly describe the issue with the device.",
                        first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone,
                        email=email, zip_code=zip_code, referral=referral,
                        imei=imei, device_model=device_model, passcode=passcode_value)

    if not diagnostic:
        diagnostic_value = answer
        ticket = {
            "day": day,
            "time_slot": time_slot,
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "alt_phone": alt_phone,
            "email": email,
            "zip_code": zip_code,
            "referral": referral,
            "imei": imei,
            "device_model": device_model,
            "passcode": passcode,
            "diagnostic": diagnostic_value
        }
        print(f"[Ticket Complete] {ticket}")
        try:
            save_mock_ticket(ticket)
            print("[Ticket saved to Google Sheets]")
        except Exception as e:
            print(f"[ERROR saving ticket] {e}")

        vr_final = VoiceResponse()
        vr_final.say(f"Thanks {first_name}, you're booked for the {day} at {time_slot}. I've saved your ticket.")
        vr_final.say("I can also answer general phone repair questions if you have any.")
        vr_final.redirect("/voice/inbound")
        
        params = urlencode(ticket)
        vr_final.redirect(f"/voice/inbound/verify?{params}")
        return Response(str(vr_final), media_type="application/xml")

        # Failsafe
    return repeat_q("Sorry, I didn’t catch that. Could you repeat?")

# === NEW: General phone repair Q&A after booking ===
@app.post("/voice/inbound")
async def voice_inbound(req: Request):
    """Entry point after booking to handle general phone repair questions."""
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()
    print(f"[Inbound Q&A] answer={answer}")

    # First turn – greet and invite a question
    if not answer:
        vr = VoiceResponse()
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
    vr = VoiceResponse()

    # Quick fixed answers for common topics
    if "hour" in lower:
        vr.say(f"Our hours are {STORE_INFO['hours']}.")
    elif "location" in lower or "address" in lower or "where" in lower:
        vr.say(f"We're at {STORE_INFO['address']}.")
    elif "how long" in lower or "long" in lower or "turnaround" in lower:
        vr.say("We usually quote repairs at 1 to 2 hours once we begin work.")
    else:
        # Hand off to your existing AI repair Q&A flow
        vr.redirect("/voice/inbound/process")
        return Response(str(vr), media_type="application/xml")

    # After answering, loop back so they can ask more
    vr.redirect("/voice/inbound")
    return Response(str(vr), media_type="application/xml")

@app.post("/voice/inbound/verify")
async def inbound_verify(
    req: Request,
    # Seeded from booking
    day: str = "", time_slot: str = "",
    first_name: str = "", last_name: str = "",
    phone: str = "", email: str = "",
    device_model: str = "", diagnostic: str = "",
    imei: str = "",
    stage: str = "intro"  # intro, name, last, phone, email, device, diag_add, imei, done
):
    form = await req.form()
    answer = (form.get("SpeechResult") or "").strip()
    print(f"[Verify] stage={stage} answer={answer} | state={{'first': '{first_name}', 'last': '{last_name}', 'phone': '{phone}', 'email': '{email}', 'device': '{device_model}', 'diag': '{diagnostic}', 'imei': '{imei}'}}")

    def ask(next_stage: str, prompt: str, **state):
        # Always allow patient pauses
        params = urlencode({**state, "stage": next_stage})
        vr = VoiceResponse()
        g = Gather(
            input="speech",
            action=f"/voice/inbound/verify?{params}",
            method="POST",
            timeout=30,
            speech_timeout="auto",
            speech_model="phone_call"
        )
        g.say(prompt)
        vr.append(g)
        return Response(str(vr), media_type="application/xml")

    # Intro summary and start
    if stage == "intro":
        summary = []
        if first_name or last_name: summary.append(f"name {first_name} {last_name}".strip())
        if phone: summary.append(f"phone {format_phone_usa(phone)}")
        if email: summary.append(f"email {normalize_email(email)}")
        if device_model: summary.append(f"device {device_model}")
        if diagnostic: summary.append(f"issue {diagnostic}")
        intro_msg = "I’ll quickly verify your details. " + (". ".join(summary) + "." if summary else "")
        return ask("name", f"{intro_msg} Is your first name still {first_name or 'unknown'}? You can say yes or say your first name.", 
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # First name
    if stage == "name":
        low = answer.lower()
        if low in ("yes", "yeah", "yep", "correct") and first_name:
            return ask("last", f"Great. Is your last name still {last_name or 'unknown'}? You can say yes or say your last name.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        # Treat any other non-empty answer as replacement
        if answer:
            return ask("last", f"Thanks {answer}. Is your last name still {last_name or 'unknown'}? You can say yes or say your last name.",
                       day=day, time_slot=time_slot, first_name=answer, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        return ask("name", "Sorry, what’s your first name?",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # Last name
    if stage == "last":
        low = answer.lower()
        if low in ("yes", "yeah", "yep", "correct") and last_name:
            return ask("phone", f"I have your phone as {format_phone_usa(phone)}. Is that correct? You can say yes or give a different number.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            return ask("phone", f"Got it, {answer}. Now, I have your phone as {format_phone_usa(phone)}. Is that correct?",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=answer, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        return ask("last", "Sorry, what’s your last name?",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # Phone with 10-digit requirement
    if stage == "phone":
        low = answer.lower()
        if low in ("yes", "yeah", "yep", "correct") and is_valid_phone(phone):
            return ask("email", f"I have your email as {normalize_email(email)}. Is that correct? You can say yes or give a different email.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            new_digits = normalize_phone(answer)
            if is_valid_phone(new_digits):
                pretty = format_phone_usa(new_digits)
                return ask("email", f"Thanks. I’ll use {pretty}. Is your email still {normalize_email(email) or 'unknown'}?",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=new_digits, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
            return ask("phone", "I didn’t get a 10 digit phone number. Please say it digit by digit.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        return ask("phone", "What’s the best phone number to reach you? Please say it digit by digit.",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # Email with simple pattern
    if stage == "email":
        low = answer.lower()
        if low in ("yes", "yeah", "yep", "correct") and is_valid_email(email):
            return ask("device", f"I have your device as {device_model or 'unknown'}. Is that correct? You can say yes or say the model.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            new_email = normalize_email(answer)
            if is_valid_email(new_email):
                return ask("device", f"Thanks. I’ll use {new_email}. Is your device still {device_model or 'unknown'}?",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=new_email, device_model=device_model, diagnostic=diagnostic, imei=imei)
            return ask("email", "That email didn’t look right. Please say it like user at gmail dot com.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        return ask("email", "What’s the best email to reach you?",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # Device model using curated list and suggestion
    if stage == "device":
        low = answer.lower()
        if low in ("yes", "yeah", "yep", "correct") and device_model:
            return ask("diag_add", f"I have your issue as {diagnostic or 'not set'}. Would you like to add more detail?",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            # Exact match first
            exact = next((m for m in ALL_DEVICE_MODELS if m.lower() == answer.lower()), None)
            if exact:
                return ask("diag_add", f"Got it, {exact}. Your issue is {diagnostic or 'not set'}. Add more detail?",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=exact, diagnostic=diagnostic, imei=imei)
            # Suggest close matches
            suggestions = suggest_device_models(answer, limit=3)
            if suggestions:
                opts = "; ".join(suggestions)
                return ask("device_choice", f"I heard {answer}. Did you mean {opts}? Please say the exact one.",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
            # No good suggestion, ask again
            return ask("device", "Could you say the device model again, like iPhone 13 or Galaxy S22 Ultra?",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        return ask("device", "What device model is it?",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    if stage == "device_choice":
        # Accept any of the suggestions verbatim if they repeat it
        if answer:
            match = next((m for m in ALL_DEVICE_MODELS if m.lower() == answer.lower()), None)
            if match:
                return ask("diag_add", f"Thanks. Your device is {match}. Would you like to add more detail to the issue?",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=match, diagnostic=diagnostic, imei=imei)
        return ask("device", "No problem. What device model is it?",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # Append diagnostic if desired
    if stage == "diag_add":
        low = answer.lower()
        if low in ("no", "nah", "nope", "skip"):
            return ask("imei", "If you have the device IMEI number, you can say it now. Otherwise say skip.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            new_diag = f"{diagnostic}; {answer}" if diagnostic else answer
            return ask("imei", "Thanks. If you have the device IMEI number, say it now. Otherwise say skip.",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=new_diag, imei=imei)
        return ask("diag_add", "You can add any extra details about the issue, or say skip.",
                   day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)

    # IMEI optional with validation and soft lookup
    if stage == "imei":
        low = answer.lower()
        if low in ("skip", "no", "nah", "nope", ""):
            # Done, move to Q&A
            return ask("done", "All set. I can answer general phone repair questions now. What would you like to know?",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
        if answer:
            raw = re.sub(r"\D", "", answer)
            if not is_valid_imei(raw):
                return ask("imei", "That didn’t sound like a valid IMEI. It should be 15 digits. You can try again or say skip.",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)
            # Optional checker, non-blocking
            checked_model = None
            try:
                checked = imei_lookup(raw)  # implement or integrate as needed
                checked_model = (checked.get("brand", "") + " " + checked.get("model", "")).strip()
            except Exception as e:
                print(f"[IMEI Lookup ERROR] {e}")
            if checked_model and device_model and checked_model.lower() != device_model.lower():
                return ask("device", f"I found {checked_model} for that IMEI, which is different from {device_model}. Would you like to update the device model? Say the correct model, or say keep.",
                           day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=raw)
            # Accept and proceed
            return ask("done", "Thanks. IMEI added. I can answer general phone repair questions now. What would you like to know?",
                       day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=raw)

    if stage == "done":
        # Hand off to your AI Q&A flow
        vr = VoiceResponse()
        params = urlencode({
            "day": day, "time_slot": time_slot,
            "first_name": first_name, "last_name": last_name,
            "phone": phone, "email": email,
            "device_model": device_model, "diagnostic": diagnostic, "imei": imei
        })
        vr.redirect(f"/voice/inbound/process?{params}")
        return Response(str(vr), media_type="application/xml")

    # Unknown stage fallback
    return ask("intro", "Let’s verify your details.",
               day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, email=email, device_model=device_model, diagnostic=diagnostic, imei=imei)


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