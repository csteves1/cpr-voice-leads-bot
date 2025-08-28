import re
from sheets import get_price_rows

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def lookup_price(device_raw: str, repair_type_raw: str):
    dev = _norm(device_raw)
    rep = _norm(repair_type_raw)
    for row in get_price_rows():
        if _norm(row.get("Device","")) == dev and _norm(row.get("RepairType","")) == rep:
            return {
                "price": row.get("Price"),
                "sku": row.get("SKU"),
                "item_id": row.get("RepairQItemId"),
                "device": row.get("Device"),
                "repair_type": row.get("RepairType"),
            }
    return None