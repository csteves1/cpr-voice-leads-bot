import os, json, time
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
_SVC_JSON = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")  # full JSON string

_cache = {"rows": None, "ts": 0}
_TTL = 300  # 5 min

def _client():
    cred_dict = json.loads(_SVC_JSON)
    creds = Credentials.from_service_account_info(cred_dict, scopes=_SCOPES)
    return build("sheets", "v4", credentials=creds)

def get_price_rows(range_name="Prices!A:E"):
    now = time.time()
    if _cache["rows"] and now - _cache["ts"] < _TTL:
        return _cache["rows"]
    svc = _client()
    resp = svc.spreadsheets().values().get(
        spreadsheetId=_SPREADSHEET_ID, range=range_name
    ).execute()
    values = resp.get("values", [])
    if not values:
        return []
    headers, *rows = values
    mapped = [dict(zip(headers, r + [""]*(len(headers)-len(r)))) for r in rows]
    _cache["rows"], _cache["ts"] = mapped, now
    return mapped