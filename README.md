# Phase 3 - FHIR EHR integration (Streamlit demo)

Synthetic-data demo of the Phase 3 scope in the Vital Sign Anomaly Detector PRD.

**Inbound (EHR -> detector):** reads Patient, Condition, MedicationRequest, Procedure over FHIR R4.
A profiling agent *proposes* an age/chronic profile, a custom SpO2 range (e.g. COPD 88-92%) and
re-baselines (new diagnosis, medication change, recent procedure). A clinician confirms or rejects each one.

**Outbound (detector -> EHR):** alert events are written as Observation (LOINC) + Flag in one
transaction Bundle. Only anomaly events leave, not the raw stream (PRD section 5 privacy rationale).

**Audit:** every confirm/reject/write-back is recorded in a decision log and as a FHIR AuditEvent.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `fhir_layer.py`, `requirements.txt` to a GitHub repo.
2. Go to share.streamlit.io, click **Create app**, pick the repo/branch, set main file to `app.py`.
3. Click **Deploy**.

## Important
- Community Cloud is not HIPAA-ready (no BAA). Use synthetic data only. A production deployment
  needs a compliant host plus the controls in PRD sections 6-7.
- The built-in mock server is the default. The external-server option points at a FHIR R4 base URL
  (public HAPI test server by default); write-back is off unless you tick it.
- Assumptions to review clinically: outside a custom SpO2 range, population bands apply; the
  recency windows (14 d meds, 30 d procedures/diagnoses) and median window/debounce (3 and 4 samples
  at 5 s) are placeholders; chronic classification uses a small SNOMED/keyword table.
- Detection runs in-app here only to simulate the edge stream; in the real design it stays on the edge gateway.
