import os, re
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from intents import is_hours, is_location, is_phone, is_landmarks, is_directions
from pricing import lookup_price
from repairq import get_inventory_by_sku, create_appointment
from calls import start_outbound_call

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
               hints="repair, screen, battery, iPhone, price, directions, appointment, hours, address")
    vr.append(g)
    return vr

@app.post("/voice/inbound")
def voice_inbound():
    vr = VoiceResponse()
    return Response(str(say_and_listen(vr, f"Thanks for calling {STORE_INFO['name']}. How can I help?")),
                    media_type="application/xml")

@app.post("/voice/inbound/process")
async def voice_process(req: Request):
    form = await req.form()
    user_input = (form.get("SpeechResult") or "").strip()
    lower = user_input.lower()
    vr = VoiceResponse()

    if is_directions(lower):
        return Response(str(say_and_listen(vr, "Sure. What's your starting address or location?")), media_type="application/xml")
    if is_hours(lower):
        return Response(str(say_and_listen(vr, f"Our hours are {STORE_INFO['hours']}. Anything else I can help with?")),
                        media_type="application/xml")
    if is_location(lower):
        return Response(str(say_and_listen(vr, f"We're at {STORE_INFO['address']}. Do you need directions?")),
                        media_type="application/xml")
    if is_phone(lower):
        return Response(str(say_and_listen(vr, f"Our phone number is {STORE_INFO['phone']}. What else can I help with?")),
                        media_type="application/xml")
    if is_landmarks(lower):
        return Response(str(say_and_listen(vr, "We’re near Goodwill and Lowe’s, in the strip with Chipotle, McAlister’s, Sport Clips, and UPS."),
                        media_type="application/xml"))

    # Domain guard before LLM fallback
    repair_keywords = ["repair","screen","battery","cracked","broken","device","phone","tablet","hours","address","location","directions","price","quote","appointment"]
    if not any(t in lower for t in repair_keywords):
        return Response(str(say_and_listen(vr, "I can help with repairs, pricing, or booking. Is your question about a device or repair?")),
                        media_type="application/xml")

# LLM fallback with Sheets lookup
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Try to detect a priceable repair from their words
        detected = None
        for row in lookup_price_rows():  # you'd add this helper to pricing.py to get all rows
            if row['Device'].lower() in lower and row['RepairType'].lower() in lower:
                detected = row
                break

        if detected:
            vr.say(f"The current price for {detected['Device']} {detected['RepairType']} is ${detected['Price']}.")
            return Response(str(vr), media_type="application/xml")

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

    except Exception as e:
        return Response(str(say_and_listen(vr, "Sorry, I’m having trouble right now. Could you try again shortly?")),
                        media_type="application/xml")
    
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

@app.post("/voice/outbound/lead/process")
async def lead_process(req: Request):
    form = await req.form()
    speech = (form.get("SpeechResult") or "").lower()
    vr = VoiceResponse()
    if any(x in speech for x in ["yes","book","schedule","today","tomorrow"]):
        # You can gather a time window and then call create_appointment(...)
        return Response(str(say_and_listen(vr, "Great. Morning or afternoon works better?", action="/voice/outbound/lead/process")),
                        media_type="application/xml")
    return Response(str(say_and_listen(vr, "No problem. If you need anything later, just call us.")),
                    media_type="application/xml")

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