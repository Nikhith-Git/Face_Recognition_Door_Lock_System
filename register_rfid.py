# register_rfid.py

from SimpleMFRC522 import SimpleMFRC522
import json
import os


reader = SimpleMFRC522()
db_file = "rfid_users.json"

# Load existing users
if os.path.exists(db_file):

    with open(db_file, "r") as f:
        users = json.load(f)
else:
    users = {}

try:
    print("[INFO] Show the card to register...")

    id, _ = reader.read()
    id = str(id)
    print(f"[INFO] Card ID: {id}")

    if id in users:
        print("[WARN] Card already registered!")
    else:
        name = input("Enter name: ")
        number = input("Enter phone number: ")
        users[id] = {"name": name, "number": number}

        with open(db_file, "w") as f:
            json.dump(users, f, indent=4)
        print("[SUCCESS] Card registered!")
except Exception as e:
    print(f"[ERROR] {e}")
