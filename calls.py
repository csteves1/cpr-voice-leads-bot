import os
from urllib.parse import quote
from twilio.rest import Client

def start_outbound_call(lead: dict):
    """Kick off an outbound call via Twilio to the given lead."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    # Fallback for APP_BASE_URL if not set
    base_url = os.getenv("APP_BASE_URL", "https://cpr-voice-leads-bot.onrender.com")

    # Encode query parameters to avoid Twilio 400 errors
    name = quote(lead.get("name", ""))
    device = quote(lead.get("device", ""))
    repair = quote(lead.get("repair_type", ""))
    source = quote(lead.get("source", "online_lead"))

    # Build the outbound URL safely
    url = f"{base_url}/voice/outbound/lead?name={name}&device={device}&repair={repair}&source={source}"

    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=lead["phone"],
        from_=from_number,
        url=url
    )

    print(f"[CALLS] Started outbound call SID={call.sid} to {lead['phone']}")
    return call