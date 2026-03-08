import paho.mqtt.client as mqtt
import json
import csv
import os
from datetime import datetime

# -------------------------
# MQTT CONFIG
# -------------------------

BROKER = "broker.hivemq.com"  
PORT = 1883
TOPIC = "cnc/+/data"

CSV_FILE = "data/cnc_data_raw.csv"


# -------------------------
# REQUIRED KEYS
# -------------------------

REQUIRED_KEYS = [
    "machine_id",
    "timestamp",
    "temperature",
    "vibration",
    "speed",
    "energy",
    "status"
]


# -------------------------
# VALIDATION FUNCTION
# -------------------------

def validate_payload(data):

    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"❌ Missing key: {key}")
            return False

    try:
        float(data["temperature"])
        float(data["vibration"])
        int(data["speed"])
        float(data["energy"])
    except ValueError:
        print("❌ Wrong datatype")
        return False

    if not 20 <= data["temperature"] <= 120:
        print("❌ Invalid temperature")
        return False

    if not 0 <= data["vibration"] <= 6:
        print("❌ Invalid vibration")
        return False

    if data["status"] not in ["ON", "OFF", "IDLE"]:
        print("❌ Invalid status")
        return False

    return True


# -------------------------
# SAVE TO CSV
# -------------------------

def save_to_csv(data):

    file_exists = os.path.exists(CSV_FILE)

    print("Saving CSV to:", os.path.abspath(CSV_FILE))
    with open(CSV_FILE, mode="a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "machine_id",
                "temperature",
                "vibration",
                "speed",
                "energy",
                "status"
            ])

        writer.writerow([
            data["timestamp"],
            data["machine_id"],
            data["temperature"],
            data["vibration"],
            data["speed"],
            data["energy"],
            data["status"]
        ])
# -------------------------
# MQTT CALLBACKS
# -------------------------

def on_connect(client, userdata, flags, rc, properties=None):

    if rc == 0:
        print("✅ Connected to MQTT Broker")

        client.subscribe(TOPIC)

        print(f"📡 Subscribed to topic: {TOPIC}")

    else:
        print("❌ Connection failed")


def on_message(client, userdata, msg):

    raw = msg.payload.decode()

    print(f"\n📥 Message from {msg.topic}")
    print(raw)

    try:
        data = json.loads(raw)

    except json.JSONDecodeError:

        print("❌ Invalid JSON")

        return

    if validate_payload(data):

        print(
            f"✔ VALID → Temp:{data['temperature']} | Vib:{data['vibration']} | Speed:{data['speed']} | Status:{data['status']}"
        )

        save_to_csv(data)

        print("💾 Saved to CSV")

    else:

        print("⚠ Invalid message rejected")


# -------------------------
# MAIN
# -------------------------

def main():

    print("🚀 Starting MQTT Subscriber...")

    client = mqtt.Client(protocol=mqtt.MQTTv311)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)

    client.loop_forever()


if __name__ == "__main__":
    main()