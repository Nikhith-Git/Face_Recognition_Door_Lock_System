# rfid_auth.py
from SimpleMFRC522 import SimpleMFRC522
import json

reader = SimpleMFRC522()
db_file = "rfid_users.json"

def verify_rfid():

    try:
        with open(db_file, "r") as f:
            users = json.load(f)

        print("[RFID] Please scan your card...")
        id, _ = reader.read()
        id = str(id)

        if id in users:
            user = users[id]
            print(f"[RFID] Access granted for {user['name']} ({user['number']})")
            return user["name"]
        else:
            print("[RFID] Unauthorized card.")
            return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None