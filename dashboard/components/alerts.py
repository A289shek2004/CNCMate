import csv
import os
from datetime import datetime


ALERT_FILE = "data/alerts_log.csv"


def init_alert_file():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(ALERT_FILE):

        with open(ALERT_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "alert_type",
                "value",
                "status",
                "action"
            ])


def log_alert(alert_type, value, status, action):

    with open(ALERT_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().isoformat(),
            alert_type,
            value,
            status,
            action
        ])


def vibration_alert(vibration, threshold):

    if vibration >= threshold:

        return {
            "alert_type": "VIBRATION",
            "status": "HIGH_VIBRATION",
            "action": "Inspect spindle and bearings"
        }

    return None


def probability_alert(probability, threshold):

    if probability >= threshold:

        return {
            "alert_type": "FAILURE_RISK",
            "status": "FAILURE_SOON",
            "action": "Schedule immediate maintenance"
        }

    return None