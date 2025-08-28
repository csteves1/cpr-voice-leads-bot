import os, re
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from intents import is_hours, is_location, is_phone, is_landmarks, is_directions
from pricing import lookup_price
from repairq import get_inventory_by_sku, create_appointment
from calls import start_outbound_call
from pricing import lookup_price, lookup_price_rows

# --- NEW ---
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
    g = Gather(input="speech", action=action, method="POST", timeout=15, speech_timeout="auto", speech_model="phone_call",
               hints="repair, screen, battery, iPhone, price, directions, appointment, hours, address, yes, no, morning, afternoon")
    vr.append(g)
    return vr

# --- NEW: Save ticket to Google Sheets ---
def save_mock_ticket(data: dict):
    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    sheet_id = os.getenv("MOCK_TICKETS_SHEET_ID")
    service = build("sheets", "v4", credentials=creds)
    values = [[
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
    return Response(str(say_and_listen(vr, "Would you like to book a time today?", action="/voice/outbound/lead/process")),
                    media_type="application/xml")

# --- UPDATED: more forgiving, routes to intake flow ---
@app.post("/voice/outbound/lead/process")
async def lead_process(req: Request):
    form = await req.form()
    speech = (form.get("SpeechResult") or "").lower()
    vr = VoiceResponse()
    yes_words = ["yes","book","schedule","sure","yeah","yep","please","ok","okay","yup","why not"]
    if any(w in speech for w in yes_words):
        return Response(str(say_and_listen(vr, "Great. Morning or afternoon works better?", action="/voice/outbound/lead/time")),
                        media_type="application/xml")
    else:
        vr.say("No problem. If you need anything later, just call us.")
        vr.redirect("/voice/inbound")  # back to Q&A mode
        return Response(str(vr), media_type="application/xml")

# --- NEW: time slot -> full intake sequence ---
@app.post("/voice/outbound/lead/time")
async def lead_time(req: Request):
    form = await req.form()
    choice = (form.get("SpeechResult") or "").lower()
    if "morning" in choice:
        chosen_time = "morning"
    elif "afternoon" in choice:
        chosen_time = "afternoon"
    else:
                # Didn't understand → re-ask without hanging up
        return Response(str(say_and_listen(
            VoiceResponse(),
            "Sorry, I didn’t catch that. Morning or afternoon works better?",
            action="/voice/outbound/lead/time"
        )), media_type="application/xml")

    # Start the intake sequence — first name is first
    return Response(str(say_and_listen(
        VoiceResponse(),
        "Got it. What's your first name?",
        action=f"/voice/outbound/lead/intake?time={chosen_time}"
    )), media_type="application/xml")

@app.post("/voice/outbound/lead/intake")
async def lead_intake(
    req: Request,
    time: str,
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

    def repeat_q(prompt, **kwargs):
        params = "&".join([f"{k}={v}" for k, v in kwargs.items()])
        return Response(str(say_and_listen(
            VoiceResponse(), prompt,
            action=f"/voice/outbound/lead/intake?{params}"
        )), media_type="application/xml")

    if not first_name:
        if not answer:
            return repeat_q("What's your first name?", time=time)
        return repeat_q("Thanks. What's your last name?", time=time, first_name=answer)

    if not last_name:
        if not answer:
            return repeat_q("What's your last name?", time=time, first_name=first_name)
        return repeat_q("What's the best phone number to reach you?", time=time, first_name=first_name, last_name=answer)

    if not phone:
        if not answer:
            return repeat_q("What's the best phone number to reach you?", time=time, first_name=first_name, last_name=last_name)
        return repeat_q("Do you have an alternate phone number? If not, just say no.", time=time, first_name=first_name, last_name=last_name, phone=answer)

    if not alt_phone:
        if not answer:
            return repeat_q("Do you have an alternate phone number? If not, just say no.", time=time, first_name=first_name, last_name=last_name, phone=phone)
        return repeat_q("What's your email address?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=answer)

    if not email:
        if not answer:
            return repeat_q("What's your email address?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone)
        return repeat_q("What's your zip code?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=answer)

    if not zip_code:
        if not answer:
            return repeat_q("What's your zip code?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email)
        return repeat_q("How did you hear about us?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=answer)

    if not referral:
        if not answer:
            return repeat_q("How did you hear about us?", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=zip_code)
        return repeat_q("Can you provide the device IMEI? If not, just say no.", time=time, first_name=first_name, last_name=last_name, phone=phone, alt_phone=alt_phone, email=email, zip_code=zip_code, referral=answer)

    if not imei:
        imei_value = "" if answer.lower() in ["no", "nah", "none"] else answer
        # Optionally: IMEI API lookup here to auto-fill model
        return repeat_q("Do you have a passcode for the device? If not, say no.",
            time=time, first_name=first_name, last_name=last_name, phone=phone,
            alt_phone=alt_phone, email=email, zip_code=zip_code, referral=referral,
            imei=imei_value, device_model="")

    if not passcode:
        passcode_value = "" if answer.lower() in ["no", "nah", "none"] else answer
        return repeat_q("Briefly describe the issue with the device.",
            time=time, first_name=first_name, last_name=last_name, phone=phone,
            alt_phone=alt_phone, email=email, zip_code=zip_code, referral=referral,
            imei=imei, device_model=device_model, passcode=passcode_value)

    if not diagnostic:
        diagnostic_value = answer
        ticket = {
            "time_slot": time,
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
        save_mock_ticket(ticket)
        vr = VoiceResponse()
        vr.say(f"Thanks {first_name}, you're booked for the {time}. I've saved your ticket.")
        vr.say("I can also answer general phone repair questions if you have any.")
        vr.redirect("/voice/inbound")
        return Response(str(vr), media_type="application/xml")

    return repeat_q("Sorry, I didn’t catch that. Could you repeat?", time=time)

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