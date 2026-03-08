# fastapi_app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import traceback
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from datetime import datetime

from src.reports.generator import aggregate_report_data, render_pdf_from_text
from src.reports.llm_client import generate_narrative


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_PATH = "model/final_model.pkl"
LOG_PATH = "logs/api.log"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(
    title="CNCMate - Predictive Maintenance API",
    description="Predict CNC machine failure probability",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_cols = None


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found")

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    features = package["features"]

    logging.info("Model loaded successfully")

    return model, features


@app.on_event("startup")
def startup_event():

    global model, feature_cols

    try:
        model, feature_cols = load_model()

    except Exception as e:
        logging.error("Model load failed: %s", str(e))
        model = None
        feature_cols = None


# ---------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------

class PredictRequest(BaseModel):

    temperature: float
    vibration: float
    speed: float
    energy: float

    temp_roll_mean: float
    vib_roll_mean: float

    temp_change: float
    vib_change: float

    machine_stress: float
    status_encoded: int


class PredictResponse(BaseModel):

    failure_probability: float
    status: str
    recommended_action: str


# ---------------------------------------------------
# PREPROCESS INPUT
# ---------------------------------------------------

def preprocess_input(req: PredictRequest):

    vals = [
        req.temperature,
        req.vibration,
        req.speed,
        req.energy,
        req.temp_roll_mean,
        req.vib_roll_mean,
        req.temp_change,
        req.vib_change,
        req.machine_stress,
        req.status_encoded
    ]

    X = np.array(vals).reshape(1, -1)

    return X, vals


# ---------------------------------------------------
# RISK INTERPRETATION
# ---------------------------------------------------

def interpret_probability(prob):

    if prob >= 0.7:
        return "FAILURE_SOON", "Inspect machine immediately."

    elif prob >= 0.4:
        return "AT_RISK", "Schedule maintenance soon."

    else:
        return "NORMAL", "Machine operating normally."


# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ---------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:

        X, raw_vals = preprocess_input(req)

        prob = float(model.predict_proba(X)[0, 1])

        status, action = interpret_probability(prob)

        logging.info(
            "Input=%s Prediction=%.4f Status=%s",
            raw_vals, prob, status
        )

        return {
            "failure_probability": round(prob, 4),
            "status": status,
            "recommended_action": action
        }

    except Exception as e:

        logging.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------
# MODEL INFO
# ---------------------------------------------------

@app.get("/model_info")
def model_info():

    return {
        "model_loaded": model is not None,
        "features": feature_cols
    }


# ---------------------------------------------------
# REPORT REQUEST
# ---------------------------------------------------

class ReportRequest(BaseModel):

    machine_id: str
    start: str
    end: str


@app.post("/generate_report")
def generate_report(req: ReportRequest):

    try:

        df = pd.read_csv(
            "data/cnc_features.csv",
            parse_dates=["timestamp"]
        )

        df = df.sort_values("timestamp")

        start_dt = pd.to_datetime(req.start)
        end_dt = pd.to_datetime(req.end)

        data = aggregate_report_data(
            df,
            start_dt,
            end_dt,
            req.machine_id
        )

        if not data:
            raise HTTPException(
                status_code=404,
                detail="No data for selected range"
            )

        narrative = generate_narrative(data)

        filename = f"report_{req.machine_id}_{start_dt.date()}.pdf"

        outpath = os.path.join(
            "reports",
            "daily_reports",
            filename
        )

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        pdf_path = render_pdf_from_text(
            data,
            narrative,
            outpath=outpath
        )

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename
        )

    except Exception as e:

        logging.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

@app.get("/metrics")
def metrics():

    return {
        "model_loaded": model is not None,
        "model_type": str(type(model))
    }