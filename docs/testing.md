🟢 STAGE 1 — Project Setup (Environment)

First we prepare the Python environment.

Step 1 — Open terminal inside project folder

Navigate to your project:

cd cncmate
Step 2 — Create virtual environment
python -m venv venv

Activate it.

Windows:

venv\Scripts\activate

Linux / Mac:

source venv/bin/activate
Step 3 — Install required libraries

Install all dependencies:

pip install -r requirements.txt

Libraries include:

pandas

numpy

scikit-learn

fastapi

streamlit

paho-mqtt

matplotlib

Now your environment is ready.

🟢 STAGE 2 — Start Data Simulation

This simulates CNC machine sensor data.

Sensors include:

temperature
vibration
spindle speed
power usage
Step 1 — Start MQTT broker

You need an MQTT broker.

Example using Mosquitto.

Start broker:

mosquitto

Default port:

localhost:1883
Step 2 — Start sensor simulator

Run:

python src/collector/mqtt_simulator.py

What happens now:

Fake CNC sensors start generating data

Example generated data:

temperature = 65
vibration = 0.25
spindle_speed = 2500
power = 3.2

This data is sent to MQTT broker.

Step 3 — Start data subscriber

Run:

python src/collector/mqtt_subscriber.py

This script:

Reads sensor data from MQTT
Stores it in dataset

Flow now becomes:

Simulator → MQTT → Subscriber → CSV dataset

Now your project is collecting machine data.

🟢 STAGE 3 — Run ETL Pipeline

ETL means:

Extract
Transform
Load

Raw data is cleaned and converted into ML dataset.

Step 1 — Run ETL script

Run:

python notebooks/01_etl_pipeline.py

This script does:

Load raw sensor data
Remove noise
Handle missing values
Create new features

Example features created:

temperature rolling mean
vibration spike
machine stress score

Output file created:

data/cnc_features.csv

This is the training dataset.

🟢 STAGE 4 — Exploratory Data Analysis

Now understand the dataset.

Run:

python notebooks/02_eda.py

This script generates plots like:

temperature distribution
vibration trends
failure patterns
correlation heatmap

This helps answer:

Which sensor indicates machine failure?

EDA is not required to run system, but it helps understand data.

🟢 STAGE 5 — Train Machine Learning Model

Now we train the predictive maintenance model.

Step 1 — Run model training

Run:

python notebooks/03_ml_models.py

This script performs:

Load dataset
Split train/test
Train ML model
Evaluate model
Save model

Typical models used:

Random Forest
Isolation Forest

Dataset split:

80% training
20% testing
Step 2 — Model evaluation

Model metrics calculated:

Accuracy
Precision
Recall
Confusion Matrix

Example output:

Accuracy: 0.91
Precision: 0.88
Recall: 0.86
Step 3 — Save trained model

After training, the model is saved:

model/final_model.pkl

This file contains the trained ML model.

🟢 STAGE 6 — Test the Model

Before deployment, test prediction manually.

Example test script:

import pickle

model = pickle.load(open("model/final_model.pkl","rb"))

sample = [[65,0.25,2500,3.2]]

prediction = model.predict(sample)

print(prediction)

Output:

0 → No failure
1 → Failure predicted

Now model is ready.

🟢 STAGE 7 — Start FastAPI Prediction Service

Now we turn the model into a web service.

Run:

uvicorn fastapi_app:app --reload

Server starts at:

http://127.0.0.1:8000
API endpoints available

Swagger UI:

http://127.0.0.1:8000/docs

Example endpoint:

POST /predict

Example request:

{
"temperature": 65,
"vibration": 0.25,
"spindle_speed": 2500,
"power": 3.2
}

Example response:

{
"failure_probability": 0.72
}

Now the ML model is accessible via API.

🟢 STAGE 8 — Launch Monitoring Dashboard

Now run the Streamlit dashboard.

Run:

streamlit run dashboard/app.py

Dashboard opens:

http://localhost:8501

Dashboard shows:

machine sensor trends
failure probability
alerts
machine health

Now engineers can monitor machines visually.

🟢 STAGE 9 — Alert Generation

System checks:

if failure probability > threshold

Example threshold:

0.70

If exceeded:

Alert generated

Alert stored in:

data/alerts_log.csv

Example alert:

timestamp: 10:32
machine: CNC_3
alert: High vibration risk
🟢 STAGE 10 — AI Report Generation

Now generate maintenance reports.

Run:

python src/reports/generator.py

This creates report like:

Daily Machine Health Report

Example output:

Machine 4 shows rising vibration levels.
Failure probability increased by 12%.
Maintenance recommended within 24 hours.

Reports help management understand machine health.

🟢 STAGE 11 — Docker Deployment (Optional)

Now run project inside Docker.

Step 1 — Build Docker image
docker build -t cncmate .
Step 2 — Run container
docker run -p 8000:8000 cncmate

Now API runs inside Docker.

This makes deployment easier.

🔵 Complete Project Execution Flow

Your project pipeline:

1️⃣ Start MQTT Broker
2️⃣ Run Sensor Simulator
3️⃣ Run MQTT Subscriber
4️⃣ Run ETL Pipeline
5️⃣ Run EDA
6️⃣ Train ML Model
7️⃣ Save Model
8️⃣ Start FastAPI Service
9️⃣ Run Streamlit Dashboard
🔟 Generate AI Reports

System architecture becomes:

CNC Sensors
     ↓
MQTT Simulator
     ↓
MQTT Subscriber
     ↓
ETL Pipeline
     ↓
Feature Dataset
     ↓
ML Model Training
     ↓
Saved Model (.pkl)
     ↓
FastAPI Prediction API
     ↓
Streamlit Monitoring Dashboard
     ↓
Alerts + AI Reports
⭐ Important Tip (For Your Resume)

When explaining this project, say:

"I developed an end-to-end predictive maintenance system that simulates industrial CNC sensor data, processes it through an ETL pipeline, trains ML models for failure prediction, deploys them via FastAPI, and visualizes machine health through a Streamlit dashboard."

That sounds very strong in interviews.