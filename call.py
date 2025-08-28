import os
from twilio.rest import Client

SID = os.getenv("TWILIO_ACCOUNT_SID")
TOK = os.getenv("TWILIO_AUTH_TOKEN")
FROM = os.getenv("TWILIO_FROM_NUMBER")
BASE = os.getenv("APP_BASE_URL","").rstrip("/")

twilio = Client(SID, TOK)

def start_outbound_call(lead: dict):
    url = f"{BASE}/voice/outbound/lead?name={lead.get('name','')}&device={lead.get('device','')}&repair={lead.get('repair_type','')}&source={lead.get('source','online_lead')}"
    twilio.calls.create(to=lead["phone"], from_=FROM, url=url)