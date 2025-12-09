import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sqlite3

# -----------------------------
# SETTINGS
# -----------------------------

DAYS = 20
INTERVAL = 10  # seconds

start_time = datetime.now() - timedelta(days=DAYS)
rows = int((24*60*60 / INTERVAL) * DAYS)

data = []

temperature = 35
vibration = 1.0
speed = 2000
energy = 20
tool_usage = 0

failure_periods = []
failure_chance = 0.0005  # small probability every step

# -----------------------------
# SIMULATION LOOP
# -----------------------------

current_time = start_time

for i in range(rows):

    # -----------------------------
    # MACHINE STATUS
    # -----------------------------
    status = np.random.choice(["ON", "IDLE", "OFF"], p=[0.7, 0.2, 0.1])

    # -----------------------------
    # NORMAL BEHAVIOR
    # -----------------------------
    if status == "ON":
        temperature += np.random.normal(0.02, 0.1)
        vibration += np.random.normal(0.001, 0.01)
        speed += np.random.normal(0, 5)
        energy = np.random.normal(60, 5)
        tool_usage += 1
    else:
        temperature -= 0.05
        vibration = max(0.5, vibration - 0.05)
        speed = 0
        energy = 10

    # keep realistic bounds
    temperature = np.clip(temperature, 20, 90)
    vibration = np.clip(vibration, 0.3, 7)

    # -----------------------------
    # FAILURE INJECTION
    # -----------------------------
    if random.random() < failure_chance and status == "ON":
        # simulate pre-failure window
        temperature += np.random.uniform(5, 10)
        vibration += np.random.uniform(2, 3)
        speed += np.random.uniform(-200, 200)

    # -----------------------------
    # RECORD ROW
    # -----------------------------
    data.append([
        current_time,
        round(temperature, 3),
        round(vibration, 3),
        int(speed),
        round(energy, 2),
        status,
        tool_usage
    ])

    current_time += timedelta(seconds=INTERVAL)

# -----------------------------
# SAVE TO CSV
# -----------------------------
df = pd.DataFrame(data, columns=[
    "timestamp", "temperature", "vibration", "speed",
    "energy", "status", "tool_usage"
])

df.to_csv("data/cnc_data_raw.csv", index=False)

# -----------------------------
# SAVE TO SQLITE
# -----------------------------
conn = sqlite3.connect("data/cnc_raw.sqlite")
df.to_sql("cnc_raw", conn, if_exists="replace", index=False)
conn.close()

print("Data generation complete!")
print("Rows generated:", len(df))
print(df.head())
