import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime

BROKER = "broker.hivemq.com"
PORT = 1883
MACHINE_ID = "CNC_001"
TOPIC = f"cnc/{MACHINE_ID}/data"

# -----------------------
# Sensor State Variables
# -----------------------
temperature = 40.0
vibration = 0.5
speed = 1500
energy = 200.0
status = "IDLE"

def update_sensors():
    global temperature, vibration, speed, energy, status

    # Randomly select machine status
    status = random.choice(["ON", "IDLE", "OFF"])

    if status == "ON":
        temperature += random.uniform(-1, 2)
        vibration += random.uniform(-0.1, 0.3)
        speed = random.randint(1200, 3000)
        energy = random.uniform(150, 550)

    elif status == "IDLE":
        temperature += random.uniform(-0.5, 0.2)
        vibration = random.uniform(0.1, 0.6)
        speed = random.randint(200, 600)
        energy = random.uniform(50, 150)

    else:  # OFF
        temperature -= random.uniform(0.5, 1.0)
        vibration = 0.0
        speed = 0
        energy = random.uniform(10, 40)

    # Clamp values within safe realistic ranges
    temperature = max(25, min(temperature, 95))
    vibration = max(0, min(vibration, 6))
    energy = max(0, energy)

    return {
        "machine_id": MACHINE_ID,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 3),
        "speed": speed,
        "energy": round(energy, 2),
        "status": status
    }


def main():
    client = mqtt.Client()
    client.connect(BROKER, PORT)

    print(f"🚀 CNC Data Simulator Started for {MACHINE_ID}")
    print(f"Publishing to topic: {TOPIC}")

    while True:
        data = update_sensors()
        payload = json.dumps(data)

        client.publish(TOPIC, payload)
        print("📡 Published:", payload)

        time.sleep(1)  # publish every 1 second


if __name__ == "__main__":
    main()
