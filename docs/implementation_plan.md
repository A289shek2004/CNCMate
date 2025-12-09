# Alert System

## Goal Description

Enhance the Dashboard with an alert system that detects critical conditions (high vibration, high failure probability) and notifies the user via UI banners, browser notifications, and optionally Telegram.

## Proposed Changes

### Dashboard Code

#### [MODIFY] [app.py](file:///e:/CNCMATE/cncmate/dashboard/app.py)

- **Imports**: Add `os`, `csv`, `time`, `json`, `requests` (already there), `streamlit.components.v1`.
- **Sidebar**: Add threshold configurations and notification toggles.
- **Helper Functions**:
  - `log_alert(type, value, status, action, note)`: Appends to CSV.
  - `browser_notify(title, message)`: Injects JS for pop-up.
  - `send_telegram(msg, token, chat_id)`: Sends request to Telegram Bot API.
- **Logic**:
  - Calculate prediction *before* layout columns.
  - Check `vibration > threshold` and `probability > threshold`.
  - Trigger alerts if conditions met.
- **UI**:
  - Display alert banners (`st.error`, `st.warning`).
  - Add "Recent Alerts" table.

## Verification Plan

### Manual Verification

- Run `streamlit run dashboard/app.py`.
- Lower thresholds in sidebar to trigger alerts.
- Check if banners appear.
- Check `data/alerts_log.csv` for new entries.
