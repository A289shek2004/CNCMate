# CNCMate: AI-Powered CNC Predictive Maintenance System

![CNCMate Banner](docs/architecture_diagram.png)

**CNCMate** is an end-to-end industrial IoT application designed to predict CNC machine failures before they happen. It combines simulated sensor data, machine learning pipelines, a robust API, and an interactive dashboard to provide real-time analytics and actionable alerts for maintenance teams.

## 🚀 Key Features

* **Real-time Monitoring**: Live tracking of Temperature, Vibration, Speed, and Energy consumption.
* **Predictive Maintenance**: Machine Learning model (Random Forest) predicting failure probability in the next 10 minutes.
* **Anomaly Detection**: Early detection of abnormal behaviors using Isolation Forest.
* **Alert System**: Instant UI banners and optional Telegram notifications for critical thresholds.
* **AI Reporting**: Automated daily PDF reports with executive summaries, trend charts, and root-cause hints.
* **Interactive Dashboard**: User-friendly interface built with Streamlit.

## 🛠️ Technology Stack

* **Language**: Python 3.11+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Random Forest, Isolation Forest)
* **Backend API**: FastAPI, Uvicorn
* **Frontend Dashboard**: Streamlit, Matplotlib, Seaborn
* **Reporting**: FPDF, Jinja2

## 📂 Project Structure

```
CNCMate/
│── dashboard/
│   └── app.py              # Streamlit Dashboard application
│── data/
│   ├── cnc_features.csv    # Processed dataset
│   └── alerts_log.csv      # Log of generated alerts
│── docs/                   # Documentation and diagrams
│── model/
│   └── final_model.pkl     # Trained ML Pipeline
│── notebooks/
│   └── 03_ml_models.py     # Training and EDA scripts
│── reports/                # Generated PDF reports
│── src/
│   ├── reports/            # Report generation module
│   ├── data_simulation.py  # Data simulator
│   ├── etl_pipeline.py     # ETL processing script
│   └── utils.py            # Utility functions
│── fastapi_app.py          # FastAPI Backend entry point
│── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## ⚡ Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/A289shek2004/CNCMate.git
    cd CNCMate
    ```

2. **Install Dependencies**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

To run the full system, you need to start both the Backend API and the Dashboard. Open two terminal windows:

### 1. Start the API Server

The FastAPI backend serves predictions and generates reports.

```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

*API Docs available at: <http://127.0.0.1:8000/docs>*

### 2. Start the Dashboard

The Streamlit app allows you to visualize data and interact with the system.

```bash
streamlit run dashboard/app.py
```

*Dashboard will open at: <http://localhost:8501>*

## 📊 Documentation

Full project documentation, including the **Blackbook Report**, **Architecture Diagrams**, and **User Manual**, can be found in the `docs/` directory.

## ✉️ Contact

**Author**: Abhishek
**GitHub**: [A289shek2004](https://github.com/A289shek2004)

**Email**: <abhishekgup2004@gmail.com>

**LinkedIn**: [Abhishek](https://www.linkedin.com/in/1289shek2004/)

# CNCMate: AI-Powered CNC Predictive Maintenance System

![CNCMate Banner](docs/architecture_diagram.png)

**CNCMate** is an end-to-end industrial IoT application designed to predict CNC machine failures before they happen. It combines simulated sensor data, machine learning pipelines, a robust API, and an interactive dashboard to provide real-time analytics and actionable alerts for maintenance teams.

## 🚀 Key Features

* **Real-time Monitoring**: Live tracking of Temperature, Vibration, Speed, and Energy consumption.
* **Predictive Maintenance**: Machine Learning model (Random Forest) predicting failure probability in the next 10 minutes.
* **Anomaly Detection**: Early detection of abnormal behaviors using Isolation Forest.
* **Alert System**: Instant UI banners and optional Telegram notifications for critical thresholds.
* **AI Reporting**: Automated daily PDF reports with executive summaries, trend charts, and root-cause hints.
* **Interactive Dashboard**: User-friendly interface built with Streamlit.

## 🛠️ Technology Stack

* **Language**: Python 3.11+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Random Forest, Isolation Forest)
* **Backend API**: FastAPI, Uvicorn
* **Frontend Dashboard**: Streamlit, Matplotlib, Seaborn
* **Reporting**: FPDF, Jinja2

## 📂 Project Structure

```
CNCMate/
│── dashboard/
│   └── app.py              # Streamlit Dashboard application
│── data/
│   ├── cnc_features.csv    # Processed dataset
│   └── alerts_log.csv      # Log of generated alerts
│── docs/                   # Documentation and diagrams
│── model/
│   └── final_model.pkl     # Trained ML Pipeline
│── notebooks/
│   └── 03_ml_models.py     # Training and EDA scripts
│── reports/                # Generated PDF reports
│── src/
│   ├── reports/            # Report generation module
│   ├── data_simulation.py  # Data simulator
│   ├── etl_pipeline.py     # ETL processing script
│   └── utils.py            # Utility functions
│── fastapi_app.py          # FastAPI Backend entry point
│── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## ⚡ Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/A289shek2004/CNCMate.git
    cd CNCMate
    ```

2. **Install Dependencies**
   
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

To run the full system, you need to start both the Backend API and the Dashboard. Open two terminal windows:

### 1. Start the API Server

The FastAPI backend serves predictions and generates reports.

```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

*API Docs available at: <http://127.0.0.1:8000/docs>*

### 2. Start the Dashboard

The Streamlit app allows you to visualize data and interact with the system.

```bash
streamlit run dashboard/app.py
```

*Dashboard will open at: <http://localhost:8501>*

## 📊 Documentation

Full project documentation, including the **Blackbook Report**, **Architecture Diagrams**, and **User Manual**, can be found in the `docs/` directory.

## ✉️ Contact

**Author**: Abhishek Gupta

**GitHub**: [A289shek2004](https://github.com/A289shek2004)
