import paho.mqtt.client as mqtt
import json
from datetime import datetime

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "cnc/+/data"   # subscribe to all CNC machines


# -------------------------
# 1. JSON VALIDATION LOGIC
# -------------------------

REQUIRED_KEYS = [
    "machine_id", "timestamp", "temperature",
    "vibration", "speed", "energy", "status"
]

def validate_payload(data):
    """Validate incoming CNC sensor data before DB insertion."""

    # Check keys
    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"❌ INVALID: Missing key '{key}'")
            return False

    # Validate types
    try:
        float(data["temperature"])
        float(data["vibration"])
        float(data["energy"])
        int(data["speed"])
    except ValueError:
        print("❌ INVALID: Wrong datatype in payload.")
        return False

    # Validate realistic ranges
    if not 20 <= data["temperature"] <= 120:
        print(f"❌ INVALID TEMP: {data['temperature']}")
        return False

    if not 0 <= data["vibration"] <= 6:
        print(f"❌ INVALID VIBRATION: {data['vibration']}")
        return False

    if data["status"] not in ["ON", "OFF", "IDLE"]:
        print(f"❌ INVALID STATUS: {data['status']}")
        return False

    return True



# -------------------------
# 2. MQTT CALLBACKS
# -------------------------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to topic: {TOPIC}")
    else:
        print("❌ Connection failed")


def on_message(client, userdata, msg):
    raw = msg.payload.decode()
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"\n[{timestamp}] 📥 Incoming message from {msg.topic}:")
    print(raw)

    # Try to parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ INVALID JSON FORMAT")
        return

    # Validate payload
    if validate_payload(data):
        print(f"✔ VALID → Temp: {data['temperature']} | Vib: {data['vibration']} | Speed: {data['speed']} | Status: {data['status']}")
        
        # TODO (Step 4): Insert into DB here
        # insert_into_db(data)

    else:
        print("⚠ Rejected invalid message.")



# -------------------------
# 3. START SUBSCRIBER
# -------------------------

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("🚀 Starting MQTT Subscriber...")
    client.connect(BROKER, PORT)
    client.loop_forever()


if __name__ == "__main__":
    main()
