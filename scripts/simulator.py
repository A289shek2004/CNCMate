import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

DATA_PATH = "data/raw_sensor_data.csv"

os.makedirs("data", exist_ok=True)

# initialize file if not exists
if not os.path.exists(DATA_PATH):
    df = pd.DataFrame(columns=[
        "timestamp",
        "machine_id",
        "temperature",
        "vibration",
        "speed",
        "energy",
        "status"
    ])
    df.to_csv(DATA_PATH, index=False)

machine_id = "CNC_001"

print("Starting CNC sensor simulator...")

while True:

    timestamp = datetime.now()

    temperature = np.random.normal(30, 3)

    vibration = np.random.normal(0.2, 0.1)

    speed = np.random.choice([0,1500,2000,2500])

    energy = np.random.normal(200, 50)

    status = np.random.choice(["RUNNING","IDLE","MAINTENANCE"])

    row = {
        "timestamp": timestamp,
        "machine_id": machine_id,
        "temperature": round(temperature,2),
        "vibration": round(vibration,2),
        "speed": speed,
        "energy": round(energy,1),
        "status": status
    }

    df = pd.DataFrame([row])

    df.to_csv(DATA_PATH, mode="a", header=False, index=False)

    print("Generated sensor data:", row)

    time.sleep(1)