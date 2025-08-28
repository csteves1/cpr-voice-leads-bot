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

def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

STORE_INFO = {
    "name": "CPR Cell Phone Repair",
    "city": "Myrtle Beach",
    "address": "1000 South Commons Drive, Myrtle Beach, SC 29588",
    "phone": "(843) 750-0449",
    "hours": "Monday to Saturday 9am-6pm, Sunday we are closed",
}

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

        # This is the end of the /voice/inbound/process route handler.
        # No additional code is needed here.
@app.post("/webhooks/repairq/lead")
async def repairq_lead(req: Request):
    payload = await req.json()
    lead = {
        "name": payload.get("name",""),
        "phone": payload.get("phone",""),
        "device": payload.get("device",""),
        "repair_type": payload.get("repair_type",""),
        "source": payload.get("source","online_lead"),
    }
    start_outbound_call(lead)
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

from datetime import datetime as dt, timedelta
import dateparser

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

    # Try to extract time from the same phrase
    time_obj = None
    for fmt in ("%I:%M %p", "%I %p"):
        try:
            time_obj = dt.strptime(spoken.replace(".", "").upper(), fmt).time()
            break
        except ValueError:
            pass

    # If not parsed, try assuming PM for digits
    if not time_obj and any(t in spoken for t in ["am", "pm", ":"]):
        try:
            time_obj = dateparser.parse(spoken).time()
        except Exception:
            pass

    # If time is valid and in range, skip time prompt
    if time_obj and dt.strptime("9:00 AM", "%I:%M %p").time() <= time_obj <= dt.strptime("6:00 PM", "%I:%M %p").time():
        time_str = time_obj.strftime("%I:%M %p").lstrip("0")
        print(f"[Parsed Combined] Day={weekday}, Time={time_str}")
        return Response(
            str(say_and_listen(
                VoiceResponse(),
                f"Got it. Let's get you booked for {weekday.capitalize()} at {time_str}. What's your first name?",
                action=f"/voice/outbound/lead/intake?day={weekday.capitalize()}&time_slot={time_str}"
            )),
            media_type="application/xml"
        )

    # Otherwise, ask for time separately
    return Response(
        str(say_and_listen(
            VoiceResponse(),
            "What time works best for you? We’re open 9 A.M. to 6 P.M.",
            action=f"/voice/outbound/lead/time?day={weekday.capitalize()}"
        )),
        media_type="application/xml"
    )

from datetime import datetime as dt
import dateparser

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
    print(f"[Time Parsed] Final={time_str}")

    return Response(
        str(say_and_listen(
            VoiceResponse(),
            f"Got it. Let's get you booked for {day} at {time_str}. What's your first name?",
            action=f"/voice/outbound/lead/intake?day={day}&time_slot={time_str}"
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
    print(f"[Intake] day={day}, time_slot={time_slot}, first_name={first_name}, last_name={last_name}, phone={phone}, alt_phone={alt_phone}, email={email}, zip_code={zip_code}, referral={referral}, imei={imei}, device_model={device_model}, passcode={passcode}, diagnostic={diagnostic}, answer={answer}")

    def repeat_q(prompt, **kwargs):
        print(f"[Prompting] {prompt} | next_params={kwargs}")
        params = "&".join([f"{k}={v}" for k, v in kwargs.items()])
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

    # --- your existing step-by-step checks here, unchanged ---
    # In your final ticket dict, replace:
    # "time_slot": time
    # with:
    # "day": day,
    # "time_slot": time_slot

    # And in your thank-you message:
    # vr_final.say(f"Thanks {first_name}, you're booked for {day} at {time_slot}. I've saved your ticket.")
    # Step-by-step checks
    
    if not first_name:
        if not answer:
            return repeat_q("What's your first name?", day=day, time_slot=time_slot)
        return repeat_q("Thanks. What's your last name?", day=day, time_slot=time_slot, first_name=answer)

    if not last_name:
        if not answer:
            return repeat_q("What's your last name?", day=day, time_slot=time_slot, first_name=first_name)
        return repeat_q("What's the best phone number to reach you?", day=day, time_slot=time_slot, first_name=first_name, last_name=answer)

    if not phone:
        if not answer:
            return repeat_q("What's the best phone number to reach you?",day=day, time_slot=time_slot, first_name=first_name, last_name=last_name)
        return repeat_q("Do you have an alternate phone number? If not, just say no.", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=answer)

    if not alt_phone:
        if not answer:
            return repeat_q("Do you have an alternate phone number? If not, just say no.", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone)
        return repeat_q("What's your email address?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=answer)

    if not email:
        if not answer:
            return repeat_q("What's your email address?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone)
        return repeat_q("What's your zip code?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=answer)

    if not zip_code:
        if not answer:
            return repeat_q("What's your zip code?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email)
        return repeat_q("How did you hear about us?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=answer)

    if not referral:
        if not answer:
            return repeat_q("How did you hear about us?", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=zip_code)
        return repeat_q("Can you provide the device IMEI? If not, just say no.", day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=zip_code, referral=answer)

    if not imei:
        imei_value = "" if answer.lower() in ["no", "nah", "none"] else answer
        return repeat_q("Do you have a passcode for the device? If not, say no.",
            day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone,
            alt_phone=alt_phone, email=email, zip_code=zip_code, referral=referral,
            imei=imei_value, device_model="")

    if not passcode:
        passcode_value = "" if answer.lower() in ["no", "nah", "none"] else answer
        return repeat_q("Briefly describe the issue with the device.",
            day=day, time_slot=time_slot, first_name=first_name, last_name=last_name, phone=phone,
            alt_phone=alt_phone, email=email, zip_code=zip_code, referral=referral,
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
        # Save ticket safely
        try:
            save_mock_ticket(ticket)
            print("[Ticket saved to Google Sheets]")
        except Exception as e:
            print(f"[ERROR saving ticket] {e}")

        vr_final = VoiceResponse()
        vr_final.say(f"Thanks {first_name}, you're booked for the {day} at {time_slot}. I've saved your ticket.")
        vr_final.say("I can also answer general phone repair questions if you have any.")
        vr_final.redirect("/voice/inbound")
        return Response(str(vr_final), media_type="application/xml")

    # If somehow we got here without matching a step, ask again
    return repeat_q("Sorry, I didn’t catch that. Could you repeat?", day=day, time_slot=time_slot)

@app.get("/health")
def health(): return {"ok": True}
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