# Walkthrough - CNC ML Pipeline Refinement

I have successfully updated `03_ml_models.py` to implement the requested improvements for robust model training.

## Changes Created

### 1. Robust Data Loading & Feature Engineering

- **Merged `tool_usage`**: Added `pd.merge_asof` to simulate joining real-time controller data with sensor logs.
- **Leakage Prevention**: Created a forward-looking target `failure_in_next_10min` to ensure the model predicts *future* failure (precursors) rather than *current* failure state.
- **Time-Based Splitting**: Replaced random shuffle split with time-based split (first 80% time ordered for Train, last 20% for Test) to accurately simulate deployment.

### 2. Pipeline & Model Training

- **Pipeline Architecture**: Implemented `sklearn.pipeline.Pipeline` combining `StandardScaler` and `RandomForestClassifier`.
- **Handling Imbalance**: Used `class_weight="balanced"` to address the class disparity.
- **Imputation**: Added `SimpleImputer(strategy="median")` to handle missing values gracefully.

## Verification Results

### Execution

The script `notebooks/03_ml_models.py` runs successfully to completion.

### Model Performance Check

- **Accuracy**: ~85%
- **Issue Discovered**: The dataset is heavily imbalanced towards **FAILURE (87%)**.
  - `Original Failure Rate`: 87.0%
  - `Target (Next 10min) Rate`: 87.8%
- **Result**: Due to this extreme saturation of "failure" states in the data, the model maximizes accuracy by predicting "Fail" for almost all cases.
  - Precision for Class 0 (Normal): 0.00
  - Recall for Class 0 (Normal): 0.00
- **ROC AUC**: 0.75 (Indicates the model *has* some ranking ability, but the decision threshold is skewed).

### Anomaly Detection

- `IsolationForest` was also run. It flagged ~2% of data as anomalies, but these did not correlate strongly with the labeled failures (which are 87% of data).

## Next Steps

- **Data Review**: The high prevalence of "Failure" labels (87%) is unusual. You may need to review how the data was collected or labeled. If the machine is broken 87% of the time, predictive maintenance is difficult.
- **Threshold Tuning**: You can adjust the decision threshold of the Random Forest (currently 0.5) to catch more "Normal" states if needed, but fixing the data balance is higher priority.

### 3. Deployment (FastAPI)

- **API Server**: Created `src/fastapi_app.py` serving the model at `http://localhost:8000`.
- **Endpoints Verified**:
  - `/health`: Returns `{"status": "ok", "model_loaded": true}`
  - `/predict`: Successfully predicts failure probability.
- **Example Response**:

  ```json
  {
    "failure_probability": 0.6050,
    "status": "AT_RISK",
    "recommended_action": "Schedule inspection within next shift; monitor closely."
  }
  ```

- **Files Created**: `requirements.txt` and `Dockerfile` for containerization.

### 4. Interactive Dashboard (Streamlit)

- **App Code**: `dashboard/app.py` created.
- **Features**:
  - **Machine Overview**: Real-time metrics + Failure Probability from API.
  - **Trends**: Interactive charts for Temperature, Vibration, and Tool Wear.
  - **Reports**: Daily summary with CSV and PDF export.
- **Run Instructions**:

  ```bash
  streamlit run dashboard/app.py
  ```

- **Verification**: Verified app launches successfully.

### 5. Alert System

- **Integration**: Added to `dashboard/app.py`.
- **Features**:
  - Configurable thresholds for Vibration and Probability.
  - UI Banners & Browser Notifications.
  - Logging to `data/alerts_log.csv`.
  - Optional Telegram integration.
- **Integration**: Added to `dashboard/app.py`.
- **Features**:
  - Configurable thresholds for Vibration and Probability.
  - UI Banners & Browser Notifications.
  - Logging to `data/alerts_log.csv`.
  - Optional Telegram integration.
- **Verification**: Logging function writes to CSV as expected.

### 6. Documentation

- **Report**: Generated `CNCMate_Blackbook_Report.md`.
- **Content**: Includes Abstract, System Architecture, DFD, ER Diagrams, ML Workflow, and Results.
