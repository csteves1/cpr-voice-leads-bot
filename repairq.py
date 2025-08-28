import os, requests

BASE = os.getenv("REPAIRQ_BASE_URL", "").rstrip("/")
KEY = os.getenv("REPAIRQ_API_KEY", "")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def get_inventory_by_sku(sku: str):
    r = requests.get(f"{BASE}/inventory/items", params={"sku": sku}, headers=H, timeout=10)
    r.raise_for_status()
    return r.json()

def create_appointment(customer: dict, device: str, repair_type: str, when_iso: str):
    # Adjust to your tenant’s appointment endpoint schema
    payload = {
        "customer": customer, "device": device,
        "repair_type": repair_type, "when": when_iso
    }
    r = requests.post(f"{BASE}/appointments", json=payload, headers=H, timeout=10)
    r.raise_for_status()
    return r.json()