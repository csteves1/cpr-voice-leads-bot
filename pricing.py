import re, os, json, gspread

def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def lookup_price(spoken):
    spoken_norm = normalize(spoken)

    # Synonym mapping
    synonyms = {
        "screen": "screen replacement",
        "battery": "battery replacement",
        "charging port": "charging port replacement"
    }
    for k, v in synonyms.items():
        if k in spoken_norm and v not in spoken_norm:
            spoken_norm = spoken_norm.replace(k, v)

    creds_json = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("[WARN] No Google Sheets creds")
        return None

    creds = gspread.service_account_from_dict(json.loads(creds_json))
    sheet = creds.open_by_key(os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")).sheet1
    rows = sheet.get_all_records()

    # Exact-ish match
    for row in rows:
        device_norm = normalize(row.get("Device", ""))
        repair_norm = normalize(row.get("RepairType", ""))
        if device_norm in spoken_norm and repair_norm in spoken_norm:
            return row

    # Partial match fallback
    for row in rows:
        device_norm = normalize(row.get("Device", ""))
        if device_norm in spoken_norm:
            return row

    return None

def lookup_price_rows():
    creds_json = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("[WARN] No Google Sheets creds")
        return []
    creds = gspread.service_account_from_dict(json.loads(creds_json))
    sheet = creds.open_by_key(os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")).sheet1
    return sheet.get_all_records()