# ⚙️ CNCMate: AI-Powered CNC Predictive Maintenance System

![CNCMate Architecture](docs/architecture_diagram.png)

**CNCMate** is an end-to-end industrial IoT platform designed to **predict CNC machine failures before they happen**.

The system simulates real-time sensor data, processes it through a data pipeline, applies machine learning models for predictive maintenance, and visualizes insights through an interactive monitoring dashboard.

This project demonstrates a **complete ML engineering pipeline** combining **data engineering, machine learning, API deployment, and real-time monitoring dashboards**.

---

# 🚀 Key Features

### 🔹 Real-Time Machine Monitoring
Live visualization of:

- Temperature
- Vibration
- Spindle Speed
- Energy Consumption
- Tool Wear

---

### 🔹 Predictive Maintenance
Machine learning models analyze sensor patterns to estimate **failure probability within the next 10 minutes**.

Model pipeline includes:

- Feature Engineering
- Logistic Regression Pipeline
- Random Forest comparison
- Isolation Forest anomaly detection

---

### 🔹 Anomaly Detection
Isolation Forest identifies abnormal machine behavior before failure occurs.

---

### 🔹 Alert System
Automatic alerts triggered when:

- Failure probability exceeds threshold
- Sensor readings exceed safety limits

Alerts can appear as:

- Dashboard warning banners
- Logged alert history
- Optional Telegram notifications

---

### 🔹 AI Maintenance Reports
Daily automated PDF reports including:

- Machine statistics
- Failure trends
- Sensor analysis
- AI-generated maintenance recommendations

---

### 🔹 Interactive Dashboard
Industrial monitoring dashboard built with **Streamlit** providing:

- Real-time machine health
- KPI monitoring
- Sensor trends
- Failure prediction
- Alert center
- Shift analytics
- Report generation

---

# 🧠 System Architecture


Sensor Simulator

↓

Raw Machine Data

↓

ETL Pipeline

↓

Feature Engineering

↓

Machine Learning Model

↓

FastAPI Prediction API

↓

Streamlit Monitoring Dashboard

↓

AI Maintenance Reports


This architecture mirrors **real industrial predictive maintenance systems**.

---

# 🛠 Technology Stack

### Programming
Python 3.11+

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- Logistic Regression
- Random Forest
- Isolation Forest

### Backend API
- FastAPI
- Uvicorn

### Dashboard
- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Reporting
- FPDF
- Jinja2

---

# 📂 Project Structure


CNCMate/

│

├── dashboard/

│ └── app.py # Streamlit dashboard

│

├── data/

│ ├── raw_sensor_data.csv

│ ├── cnc_features.csv

│ └── alerts_log.csv

│

├── docs/

│ └── architecture_diagram.png

│

├── model/

│ └── final_model.pkl

│

├── notebooks/

│ ├── 01_etl_pipeline.py

│ ├── 02_eda.py

│ └── 03_ml_models.py

│

├── reports/

│ └── daily_reports/

│

├── scripts/

│ ├── simulator.py

│ └── etl_runner.py

│

├── src/

│ ├── reports/

│ ├── data_simulation.py

│ ├── etl_pipeline.py

│ └── utils.py

│

├── fastapi_app.py

├── requirements.txt

└── README.md



---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/A289shek2004/CNCMate.git
cd CNCMate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Full System

Open 4 terminals.

1️⃣ Start Sensor Simulator

Simulates CNC machine sensor data.

python scripts/simulator.py
2️⃣ Start ETL Pipeline

Processes raw sensor data and generates features.

python scripts/etl_runner.py
3️⃣ Start FastAPI Backend

Serves prediction API and report generation.

uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload

API documentation:

http://127.0.0.1:8000/docs
4️⃣ Start Monitoring Dashboard
streamlit run dashboard/app.py

Dashboard:

http://localhost:8501
📊 Dashboard Capabilities

The dashboard provides:

Operational KPIs

Machine health monitoring

Failure probability visualization

Sensor trend analysis

Alert center

Shift analytics

AI report generation

📈 Machine Learning Model

Features used:

temperature
vibration
speed
energy
temp_roll_mean
vib_roll_mean
temp_change
vib_change
machine_stress
status_encoded

Target:

Failure within next 10 minutes

Model evaluation includes:

Accuracy

Precision

Recall

ROC-AUC

Confusion Matrix

📚 Documentation

Full documentation available in:

docs/

Includes:

Blackbook report

Architecture diagrams

Project explanation

👨‍💻 Author

Abhishek Gupta

GitHub
https://github.com/A289shek2004

LinkedIn
https://www.linkedin.com/in/1289shek2004/

Email
abhishekgup2004@gmail.com


---

# ⭐ Improvements I Recommend

Add a **screenshots section**:


docs/screenshots/dashboard.png
docs/screenshots/prediction.png
docs/screenshots/report.png


Then add in README:

```markdown
# Dashboard Preview

![Dashboard](docs/screenshots/dashboard.png)
