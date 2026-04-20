# Streamlit App Instructions

- Keep Streamlit code thin; use `tcco2_accuracy.ui_api` or package functions rather than duplicating math in the app layer.
- Do not add persistence, telemetry, PHI storage, or uploads of patient-level data unless explicitly requested and documented.
- Use synthetic examples and small fixtures only.
- App copy should be cautious about uncertainty and should not overclaim clinical validation.
- Verify app-facing changes with package tests plus a Streamlit import/smoke check when practical.
