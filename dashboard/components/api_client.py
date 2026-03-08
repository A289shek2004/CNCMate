import requests


API_BASE = "http://127.0.0.1:8000"


def check_api_health():

    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except:
        return {"status": "offline"}


def predict_failure(data):

    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json=data,
            timeout=4
        )

        r.raise_for_status()

        return r.json()

    except Exception as e:

        return {
            "failure_probability": 0,
            "status": "API_ERROR",
            "recommended_action": str(e)
        }


def generate_report(payload):

    try:
        r = requests.post(
            f"{API_BASE}/generate_report",
            json=payload,
            timeout=20
        )

        if r.status_code == 200:
            return r.content

        return None

    except:
        return None