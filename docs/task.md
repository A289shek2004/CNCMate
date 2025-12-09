# Refining CNC Machine Learning Pipeline

- [x] Inspect columns & NaNs (Step 1)
- [x] Merge `tool_usage` from raw data if missing (Step 1 & 7)
- [x] Create `failure_in_next_10min` predictive target (Step 2)
- [x] Impute NaNs (Step 3)
- [x] Implement Time-based train-test split (Step 4)
- [x] Setup Pipeline with Scaler and RandomForest (Step 5)
- [x] Handle class imbalance (Step 6)
- [x] Retrain and Evaluate (Step 8)
- [x] Save Model (Step 8)
- [x] Anomaly Detection Check (Step 9)
- [x] Final Correlation Check (Step 10)

# Model Deployment (FastAPI)

- [x] Create `src/fastapi_app.py`
- [x] Create `requirements.txt`
- [x] Create `Dockerfile`
- [x] Verify API (run locally)

# Streamlit Dashboard

- [x] Create `dashboard/app.py`
- [x] Update `requirements.txt`
- [x] Verify Dashboard runs

# Alert System

- [x] Modify `dashboard/app.py` to include Alerts
- [x] Verify functionality

# Documentation & Report

- [ ] Create `CNCMate_Blackbook_Report.md`
- [ ] Add Mermaid Diagrams (Architecture, DFD, ER, Workflow)
- [ ] Include Model Accuracy Summary
