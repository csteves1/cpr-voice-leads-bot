import os
from twilio.rest import Client

def start_outbound_call(lead: dict):
    """Kick off an outbound call via Twilio to the given lead."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=lead["phone"],
        from_=from_number,
        url=f"{os.getenv('APP_BASE_URL')}/voice/outbound/lead"
            f"?name={lead['name']}&device={lead['device']}&repair={lead['repair_type']}&source={lead['source']}"
    )
    print(f"[CALLS] Started outbound call SID={call.sid} to {lead['phone']}")
    return call