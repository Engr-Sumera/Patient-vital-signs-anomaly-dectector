"""Phase 1 + 2 + 3 demo: Patient Vital Sign Anomaly Detector — FHIR R4 EHR integration.

Synthetic data only. In production the detection loop runs on the edge gateway;
this app plays the cloud/integration layer and simulates the edge stream.

PRD v2.1 features implemented:
  * Phase 1: HR + SpO2 rule-based detection with debounce, artifact rejection,
              rate-of-change detection
  * Phase 2: Composite monotonic logistic risk score (HR/SpO2/SBP/DBP/Temp),
              trajectory early-warning, Phase 2 FHIR RiskAssessment write-back
  * Phase 3: FHIR R4 EHR integration (inbound profiling, outbound transaction Bundle)
  * Clinician action set: Soft reset / Hard reset / Re-baseline / Snooze / Flag for review
  * Alert response tracking: ack/dismiss with structured reasons, adjudicated outcomes
  * Alarm-fatigue metrics dashboard (ack rate, p95 TTA, FP rate, retuning flag)
  * Patient summary card: risk level, baseline status, repeat-alert rate
  * Patient-specific baseline blending (cold-start -> established -> overridden)
"""
import json
from datetime import datetime, timedelta

import importlib

import altair as alt
import pandas as pd
import streamlit as st

import fhir_layer as fl
importlib.reload(fl)

st.set_page_config(
    page_title="Vital Sign Anomaly Detector — FHIR",
    page_icon="🩺",
    layout="wide",
)

# ─────────────────────────────────────────────────────────── CSS polish
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
.alert-high   { background:#fde8e8; border-left:4px solid #c0392b; padding:6px 10px; border-radius:4px; margin-bottom:4px; }
.alert-mod    { background:#fef3e2; border-left:4px solid #e67e22; padding:6px 10px; border-radius:4px; margin-bottom:4px; }
.alert-low    { background:#fef9e7; border-left:4px solid #d4a017; padding:6px 10px; border-radius:4px; margin-bottom:4px; }
.ew-badge     { background:#2980b9; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8rem; }
.retune-warn  { background:#c0392b; color:#fff; padding:6px 12px; border-radius:6px; font-weight:bold; }
.fallback-warn{ background:#7f8c8d; color:#fff; padding:6px 12px; border-radius:6px; font-weight:bold; }
.explain-box  { background:#eaf4fb; border-left:3px solid #2980b9; padding:4px 10px; border-radius:4px; font-size:.85rem; margin-top:4px; }
.agent-prop   { background:#f4f6f7; border:1px solid #d5d8dc; padding:6px 12px; border-radius:6px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────── sidebar / session state
st.sidebar.title("Vital Sign Anomaly Detector")
st.sidebar.markdown("**PRD v2.2** — Phase 1 + 2 + 3")

mode = st.sidebar.radio("FHIR source",
                         ["Built-in mock server (synthetic)", "External FHIR R4 server"])
is_mock = mode.startswith("Built-in")
base_url, allow_writeback = None, True
if not is_mock:
    base_url = st.sidebar.text_input("Base URL", "https://hapi.fhir.org/baseR4")
    st.sidebar.warning("Public test servers are world-readable. Synthetic data only — never real PHI.")
    allow_writeback = st.sidebar.checkbox("Enable write-back (POST) to this server", value=False)

key = (is_mock, base_url)
if st.session_state.get("client_key") != key:
    st.session_state.client_key   = key
    st.session_state.client       = fl.FHIRClient(None if is_mock else base_url)
    st.session_state.proposals    = {}
    st.session_state.accepted     = {}
    st.session_state.decision_log = []
    st.session_state.audit        = []
    st.session_state.outbox       = None
    st.session_state.all_alerts   = []   # flat list across all patients/params
    st.session_state.baselines    = {}   # pid -> {param: PatientBaseline}
    # PRD 4.9 lifecycle agents
    st.session_state.alert_tuning_agent = fl.AlertTuningAgent()
    st.session_state.drift_monitor      = fl.DriftMonitorAgent()
    st.session_state.evidence_agent     = fl.EvidencePackAgent()
    st.session_state.retraining_agent   = fl.RetrainingAgent()

client = st.session_state.client

# ── Acting Clinician / Care Team Selector
st.sidebar.subheader("Acting Clinician / Care Team")
if is_mock:
    _pracs = client.list_practitioners()
    prac_display_map = {
        f"{p['name'][0]['text']} — {p.get('extension', [{}])[0].get('valueString', 'Staff')}": p
        for p in _pracs
    }
    prac_choice = st.sidebar.selectbox("Active Medical Staff", list(prac_display_map.keys()) + ["Custom Clinician..."])
    if prac_choice == "Custom Clinician...":
        clinician = st.sidebar.text_input("Clinician name & credentials", "Dr. Demo, MD")
        clinician_id = "prac-custom"
        clinician_role = "Visiting Physician"
    else:
        chosen_prac = prac_display_map[prac_choice]
        clinician = chosen_prac["name"][0]["text"]
        clinician_id = chosen_prac["id"]
        clinician_role = chosen_prac.get("qualification", [{}])[0].get("code", {}).get("text", "Physician")
        npi_val = (chosen_prac.get("identifier") or [{}])[0].get("value", "n/a")
        pager_val = (chosen_prac.get("telecom") or [{}])[0].get("value", "n/a")
        dept_val = (chosen_prac.get("extension") or [{}])[0].get("valueString", "Medicine")
        shift_val = (chosen_prac.get("extension") or [{}, {}])[1].get("valueString", "On-Duty")
        st.sidebar.caption(
            f"**NPI:** `{npi_val}` · **Dept:** {dept_val}  \n"
            f"**Contact:** `{pager_val}` · **Shift:** {shift_val}"
        )
else:
    clinician = st.sidebar.text_input("Acting clinician", "Dr. Demo, MD")
    clinician_id = "prac-external"
    clinician_role = "External Clinician"

scenario  = st.sidebar.selectbox(
    "Simulated edge stream",
    ["Stable", "SpO2 decline", "Tachycardia", "Sensor artifact", "Sepsis (early)"])
min_tier  = st.sidebar.selectbox("Minimum alert tier to publish", ["Low", "Moderate", "High"], index=1)

st.sidebar.divider()
st.sidebar.subheader("Clinician Actions (PRD 4.2.4)")
action_soft   = st.sidebar.button("Soft reset",  help="Clears alerts & metrics; keeps baseline and profile.")
action_hard   = st.sidebar.button("Hard reset",  help="New admission: wipes everything.")
action_rebase = st.sidebar.button("Re-baseline on demand", help="Restarts the personal baseline from now.")

st.sidebar.divider()
st.sidebar.subheader("Model Health (PRD 4.8.3)")
model_healthy = st.sidebar.checkbox(
    "Phase 2 model healthy", value=True,
    help="Uncheck to simulate model failure → automatic rules-only fallback (PRD 4.8.3).")


def log_decision(action, summary, refs):
    entry = {"time": fl.now_utc().isoformat(timespec="seconds"),
             "patient": st.session_state.get("current_pid", "-"),
             "actor": clinician, "actor_id": clinician_id, "action": action,
             "summary": summary, "sources": ", ".join(refs)}
    st.session_state.decision_log.append(entry)
    st.session_state.audit.append(
        fl.build_audit_event("E", f"{clinician} ({clinician_id})",
                             st.session_state.get("current_pid", "-"),
                             f"{action}: {summary}"))


# ─────────────────────────────────────────────────────────── header
st.title("🩺 Patient Vital Sign Anomaly Detector")
st.caption("Phase 1 + 2 + 3 · FHIR R4 EHR integration · PRD v2.2 · Synthetic data only")

# ─────────────────────────────────────────────────────────── patient selection
if is_mock:
    _all_patients = client.list_patients()
    patients = {}
    for p in _all_patients:
        p_name = fl.patient_name(p)
        p_mrn = (p.get("identifier") or [{}])[0].get("value", p["id"])
        exts = p.get("extension") or []
        ward_str = exts[0].get("valueString", "Ward ?") if len(exts) > 0 else "Ward ?"
        room_str = exts[1].get("valueString", "Bed ?") if len(exts) > 1 else "Bed ?"
        label = f"{p_name} ({p_mrn}) · {ward_str} [{room_str}]"
        patients[label] = p["id"]
    pid = patients[st.selectbox("Patient (Hospital Roster)", list(patients.keys()))]
else:
    st.markdown("##### 🌐 External FHIR Server Mode")
    ext_patients = client.list_patients(count=10)
    ext_options = {}
    for p in ext_patients:
        p_name = fl.patient_name(p)
        p_id = p.get("id", "unknown")
        ext_options[f"{p_name} (ID: {p_id})"] = p_id

    col_ep1, col_ep2 = st.columns([2.5, 1.5])
    with col_ep1:
        if ext_options:
            selected_ext = st.selectbox("Existing Patients on Server", list(ext_options.keys()) + ["Enter Custom ID..."])
            default_pid = ext_options[selected_ext] if selected_ext != "Enter Custom ID..." else "example"
        else:
            default_pid = "example"
        init_pid = st.session_state.get("ext_query_pid", default_pid)
        pid = st.text_input("Patient ID to Query", init_pid).strip()
    with col_ep2:
        st.write(" ")
        st.write(" ")
        if st.button("📤 Seed 'p-001' to Server", help="Uploads synthetic patient p-001 (Amina Rahman) with full condition, med, and vital records to this FHIR server."):
            with st.spinner("Uploading p-001 bundle to external server..."):
                try:
                    new_pid = client.upload_mock_patient_to_server("p-001")
                    st.session_state["ext_query_pid"] = new_pid
                    st.success(f"Uploaded p-001 as Patient/{new_pid}!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as ex:
                    st.error(f"Upload failed: {ex}")

if not pid:
    st.info("Enter a patient ID to begin.")
    st.stop()

st.session_state.current_pid = pid


@st.cache_data(show_spinner=False, ttl=60)
def _load(_client, source, pid):
    p = _client.get_patient(pid)
    if not p:
        return (None, [], [], [], [])
    return (p,
            _client.related("Condition", pid),
            _client.related("MedicationRequest", pid),
            _client.related("Procedure", pid),
            _client.related("Observation", pid))


try:
    patient, conditions, meds, procs, baseline_obs = _load(client, str(key), pid)
except Exception as e:
    st.error(f"Could not load patient from FHIR server: {e}")
    st.stop()

if not patient:
    st.warning(f"⚠️ Patient `{pid}` was not found on `{base_url}` (HTTP 404).")
    st.markdown(
        f"""
        **Why did this happen?**
        * The ID `{pid}` does not exist on the external FHIR server (`{base_url}`).
        * If you are testing the built-in clinical cohort (**`p-001` through `p-014`**), switch back to **Built-in mock server (synthetic)** in the left sidebar.
        * If you want to use `p-001` on this external server, click **📤 Seed 'p-001' to Server** above to upload this patient and their clinical data.
        """
    )
    st.stop()

# ─────────────────────────────────────────────────────────── clinician actions
acc = st.session_state.accepted.get(pid, {"custom": {}, "rebaseline": []})

if action_soft:
    st.session_state.all_alerts = [a for a in st.session_state.all_alerts
                                    if a.get("param") not in ("HR", "SpO2")]
    log_decision("soft reset", "Alerts and metrics cleared; baseline retained.", [f"Patient/{pid}"])
    st.toast("Soft reset complete.", icon="🔄")

if action_hard:
    st.session_state.proposals.pop(pid, None)
    st.session_state.accepted.pop(pid, None)
    st.session_state.baselines.pop(pid, None)
    st.session_state.all_alerts = [a for a in st.session_state.all_alerts
                                    if a.get("param") not in ("HR", "SpO2")]
    log_decision("hard reset", "Full admission reset: profile, overrides, baseline, alerts wiped.", [f"Patient/{pid}"])
    st.toast("Hard reset complete.", icon="♻️")
    st.rerun()

if action_rebase:
    bls = st.session_state.baselines.setdefault(pid, {})
    for param in ("HR", "SpO2", "SBP", "DBP", "Temp"):
        bls.setdefault(param, fl.PatientBaseline(param)).invalidate()
    log_decision("re-baseline on demand", "Personal baseline restarted from current point.", [f"Patient/{pid}"])
    st.toast("Re-baseline started.", icon="📊")


def decide(prop, accept):
    prop["status"] = "accepted" if accept else "rejected"
    a = st.session_state.accepted.setdefault(pid, {"custom": {}, "rebaseline": []})
    if accept and prop["kind"] == "custom_range":
        for param, rng in prop["payload"].items():
            a["custom"][param] = tuple(rng)
    if accept and prop["kind"] == "rebaseline":
        a["rebaseline"].append(prop["title"])
        # auto-trigger re-baseline in PatientBaseline
        bls = st.session_state.baselines.setdefault(pid, {})
        for param in ("HR", "SpO2", "SBP", "DBP", "Temp"):
            bls.setdefault(param, fl.PatientBaseline(param)).invalidate()
    log_decision(f"agent proposal {prop['status']}", prop["title"], prop["source_refs"])


# ─────────────────────────────────────────────────────────── chart helper
def vital_chart(rows, alerts, param, p2_alerts=None):
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("time:T", title=None))
    layers = [
        base.mark_line(opacity=0.3, color="#888").encode(
            y=alt.Y("raw:Q", title=param, scale=alt.Scale(zero=False))),
        base.mark_line(color="#1f77b4", strokeWidth=2).encode(y="filtered:Q"),
    ]
    if alerts:
        adf = pd.DataFrame(alerts)
        layers.append(alt.Chart(adf).mark_point(size=140, filled=True).encode(
            x="time:T", y="value:Q",
            tooltip=["tier", "value", "time"],
            color=alt.Color("tier:N", scale=alt.Scale(
                domain=["Low", "Moderate", "High"],
                range=["#d4a017", "#e67e22", "#c0392b"]))))
    if p2_alerts:
        p2df = pd.DataFrame(p2_alerts)
        layers.append(alt.Chart(p2df).mark_point(shape="triangle-up", size=180, filled=True,
                                                   color="#8e44ad").encode(
            x="time:T", y="value:Q", tooltip=["tier", "value", "time"]))
    return alt.layer(*layers).properties(height=220)


# ─────────────────────────────────────────────────────────── tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1 · EHR → Profile",
    "2 · Monitor & Alerts",
    "3 · Phase 2 Risk Score",
    "4 · Alert Management",
    "5 · FHIR Inspector",
    "6 · Audit & Decision Log",
    "7 · Lifecycle & Governance",
])

# ═══════════════════════════════════════════════════════════ TAB 1 — EHR → Profile
with tab1:
    age = fl.age_from(patient)
    exts = patient.get("extension") or []
    ward = exts[0].get("valueString", "n/a") if len(exts) > 0 else "n/a"
    room = exts[1].get("valueString", "Bed ?") if len(exts) > 1 else "Bed ?"
    code_stat = exts[2].get("valueString", "FULL CODE") if len(exts) > 2 else "FULL CODE"
    adm_date = exts[3].get("valueString", "Recent") if len(exts) > 3 else "Recent"
    primary_nurse = exts[4].get("valueString", "Unassigned") if len(exts) > 4 else "Unassigned"
    mrn = (patient.get("identifier") or [{}])[0].get("value", patient["id"])
    attending_doc = (patient.get("generalPractitioner") or [{}])[0].get("display", "Staff Attending")

    # ── Patient Hospital Identity Banner
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1.5, 1.5])
    c1.metric("Patient", f"{fl.patient_name(patient)}")
    c2.metric("MRN", mrn)
    c3.metric("Age / Sex", f"{age if age is not None else 'n/a'} / {patient.get('gender', 'n/a')[0].upper()}")
    c4.metric("Location", f"{ward}", room)
    c5.metric("Code Status", code_stat)

    # ── Assigned Care Team Card
    with st.container(border=True):
        ct1, ct2, ct3 = st.columns([2, 2, 2])
        ct1.markdown(f"👨‍⚕️ **Attending Physician:**  \n{attending_doc}")
        ct2.markdown(f"👩‍⚕️ **Primary Nurse:**  \n{primary_nurse}")
        ct3.markdown(f"🩺 **Active Reviewer (You):**  \n{clinician} `({clinician_role})`")

    # ── Recent Baseline Vital Sign Observations in FHIR Store
    if baseline_obs:
        st.markdown("##### 📋 Last Recorded Ward Vital Signs (FHIR Observation Store)")
        obs_map = {o["code"]["coding"][0]["code"]: o["valueQuantity"]["value"]
                   for o in baseline_obs if "valueQuantity" in o and o.get("code", {}).get("coding")}
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Heart Rate", f"{obs_map.get('8867-4', '—')} bpm")
        v2.metric("SpO2", f"{obs_map.get('59408-5', '—')}%")
        v3.metric("Blood Pressure", f"{obs_map.get('8480-6', '—')} / {obs_map.get('8462-4', '—')} mmHg")
        v4.metric("Temperature", f"{obs_map.get('8310-5', '—')} °C")
        v5.metric("Recorded", "Ward Baseline (T-2h)")

    # ── Patient Summary Card (PRD 4.5)
    bls = st.session_state.baselines.get(pid, {})
    st.subheader("Patient Summary Card (PRD 4.5)")
    sc1, sc2, sc3, sc4 = st.columns(4)

    # Current risk level — based on all unacked High alerts
    active_highs = [a for a in st.session_state.all_alerts
                    if a.get("tier") == "High" and not a.get("ack_ts")]
    active_mods  = [a for a in st.session_state.all_alerts
                    if a.get("tier") == "Moderate" and not a.get("ack_ts")]
    if active_highs:
        risk_label = "🔴 HIGH"
    elif active_mods:
        risk_label = "🟠 MODERATE"
    else:
        risk_label = "🟢 LOW / Normal"
    sc1.metric("Current Risk", risk_label)

    # Baseline status for HR
    hr_bl = bls.get("HR")
    sc2.metric("HR Baseline", hr_bl.status.title() if hr_bl else "Cold-start",
               f"{hr_bl.confidence*100:.0f}% personal" if hr_bl else "0% personal")
    spo2_bl = bls.get("SpO2")
    sc3.metric("SpO2 Baseline", spo2_bl.status.title() if spo2_bl else "Cold-start",
               f"{spo2_bl.confidence*100:.0f}% personal" if spo2_bl else "0% personal")

    # Repeat-alert count this session (alarm fatigue proxy, PRD 4.4)
    repeat_count = len([a for a in st.session_state.all_alerts
                        if a.get("tier") in ("Moderate", "High")])
    sc4.metric("Alerts This Session", repeat_count)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Inbound EHR Data")
        st.write("**Conditions:** " +
                 (", ".join(fl.classify_condition(c)[0] for c in conditions) or "none"))
        st.write("**Medications:** " +
                 (", ".join(fl._text(m.get("medicationCodeableConcept")) for m in meds) or "none"))
        st.write("**Procedures:** " +
                 (", ".join(fl._text(p.get("code")) for p in procs) or "none"))

    with right:
        st.subheader("Profiling Agent Proposals (PRD 4.2.3)")
        if pid not in st.session_state.proposals:
            st.session_state.proposals[pid] = fl.propose(patient, conditions, meds, procs)
        for prop in st.session_state.proposals[pid]:
            with st.container(border=True):
                st.markdown(f"**{prop['title']}**")
                st.caption(prop["rationale"] + "  \nSource: " + ", ".join(prop["source_refs"]))
                if prop["status"] == "pending":
                    b1, b2, _ = st.columns([1, 1, 3])
                    b1.button("Confirm", key=f"ok-{prop['id']}", on_click=decide, args=(prop, True))
                    b2.button("Reject",  key=f"no-{prop['id']}", on_click=decide, args=(prop, False))
                else:
                    st.write(f"Status: **{prop['status']}** by {clinician}")

    st.caption("The agent only suggests. Nothing changes the detector until a clinician confirms (PRD 4.2.3).")

# ═══════════════════════════════════════════════════════════ TAB 2 — Monitor & Alerts
with tab2:
    acc_val = st.session_state.accepted.get(pid, {"custom": {}, "rebaseline": []})
    custom  = acc_val["custom"]
    st.write("**Active overrides:** " +
             (", ".join(f"{k} {v[0]}-{v[1]}" for k, v in custom.items()) or
              "none (population thresholds)"))
    if acc_val["rebaseline"]:
        st.write("**Re-baseline confirmed:** " + "; ".join(acc_val["rebaseline"]))

    spo2_base = float(sum(custom["SpO2"]) / 2) if "SpO2" in custom else 97.0
    series    = fl.simulate(scenario, spo2_base=spo2_base)

    # Phase 1 processing for HR and SpO2
    session_new_alerts = []
    bls = st.session_state.baselines.setdefault(pid, {})

    for param in ("HR", "SpO2"):
        rows, alerts = fl.process(series[param], param, custom)
        session_new_alerts.extend(alerts)

        # Feed stable readings into baseline
        bl = bls.setdefault(param, fl.PatientBaseline(param, custom))
        for row in rows:
            if not row["rejected"] and row["filtered"] is not None:
                bl.feed(row["time"], row["filtered"], row["tier"] or "Normal")

        st.markdown(f"**{param}** — Baseline: `{bl.status}` "
                    f"({bl.confidence*100:.0f}% personal confidence)")
        blo, bhi = bl.blended_corridor()
        st.caption(f"Blended corridor: {blo:.1f} – {bhi:.1f}")
        st.altair_chart(vital_chart(rows, alerts, param), use_container_width=True)
        n_rej = sum(r["rejected"] for r in rows)
        if n_rej:
            st.caption(f"{n_rej} implausible sample(s) rejected as artifact (PRD 4.1).")

    # Merge new alerts into session store (avoid duplicates by id)
    existing_ids = {a["id"] for a in st.session_state.all_alerts}
    for a in session_new_alerts:
        if a["id"] not in existing_ids:
            st.session_state.all_alerts.append(a)

    publishable = [a for a in session_new_alerts
                   if fl.RANK[a["tier"]] >= fl.RANK[min_tier]
                   and not a.get("snoozed")]

    st.subheader("Phase 1 Alerts")
    if session_new_alerts:
        disp = [{k: v for k, v in a.items()
                 if k in ("time", "param", "tier", "value", "source")}
                for a in session_new_alerts]
        st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)
    else:
        st.write("No Phase 1 alerts in this window.")

    def publish():
        resources = []
        for a in publishable:
            resources += [fl.build_observation(pid, a["param"], a["value"], a["time"]),
                          fl.build_flag(pid, a)]
        bundle = fl.build_transaction(resources)
        st.session_state.outbox = bundle
        try:
            client.transaction(bundle)
            log_decision("write-back (Phase 1)",
                         f"{len(publishable)} alert(s) published as Observation+Flag",
                         [f"Patient/{pid}"])
            st.session_state.publish_msg = ("ok", f"Published {len(resources)} resources.")
        except Exception as e:
            st.session_state.publish_msg = ("err", f"Write-back failed: {e}")

    can_publish = bool(publishable) and allow_writeback
    st.button(f"Publish {len(publishable)} Phase 1 alert(s) to EHR",
              on_click=publish, disabled=not can_publish)
    if not allow_writeback:
        st.caption("Write-back is disabled for the external server (sidebar).")
    msg = st.session_state.pop("publish_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])

# ═══════════════════════════════════════════════════════════ TAB 3 — Phase 2 Risk Score
with tab3:
    st.subheader("Phase 2 Composite Risk Score (PRD 4.2.5)")
    st.caption(f"Model: `{fl.P2_MODEL_VERSION}` · "
               "Cut-offs: Low ≥ 0.25, High ≥ 0.50 (clinical placeholders, PRD 4.2.5)")

    acc_val = st.session_state.accepted.get(pid, {"custom": {}, "rebaseline": []})
    custom  = acc_val["custom"]
    spo2_base = float(sum(custom["SpO2"]) / 2) if "SpO2" in custom else 97.0
    series2 = fl.simulate(scenario, spo2_base=spo2_base)

    # Use last reading of each parameter as "current"
    current_vals = {param: series2[param][-1][1] for param in ("HR", "SpO2", "SBP", "DBP", "Temp")}

    result = fl.phase2_score(current_vals, series2, custom_ranges=custom or None,
                             model_healthy=model_healthy)

    # PRD 4.8.3: fallback banner
    if result.get("fallback"):
        st.markdown('<div class="fallback-warn">&#x26A0;&#xFE0F; Phase 2 model UNAVAILABLE — '
                    'running in rules-only mode (PRD 4.8.3 guardrail)</div>',
                    unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    tier_color = {"Normal": "🟢", "Low": "🟡", "Moderate": "🟠", "High": "🔴"}.get(result["tier"], "⚪")
    r1.metric("Overall Tier", f"{tier_color} {result['tier']}")
    score_disp = f"{result['score']:.3f}" if result["score"] is not None else "N/A"
    r2.metric("Composite Score", score_disp)
    sc_disp = f"{result['score_current']:.3f}" if result["score_current"] is not None else "N/A"
    r3.metric("Current-State Score", sc_disp)

    t1, t2, t3 = st.columns(3)
    st_disp = f"{result['score_traj']:.3f}" if result["score_traj"] is not None else "N/A"
    t1.metric("Trajectory Score", st_disp)
    t2.metric("Early Warning", "⚠️ YES" if result["early_warning"] else "No")
    t3.metric("Parameters Used", ", ".join(result["params_used"]) or "none")

    # PRD 4.8.2 — Plain-language explanation
    if result.get("plain_language"):
        st.info(f"💬 **Model explanation:** {result['plain_language']}")

    # PRD 4.8.2 — Top contributing factors
    if result.get("top_factors"):
        st.markdown("**Top contributing factors (by deviation):**")
        factor_cols = st.columns(len(result["top_factors"][:5]))
        for col, (param, dev) in zip(factor_cols, result["top_factors"][:5]):
            col.metric(param, f"dev={dev:.3f}")

    if result["early_warning"]:
        st.markdown('<div class="ew-badge">⚠️ Trajectory-based early warning: deterioration projected</div>',
                    unsafe_allow_html=True)
        st.caption("Risk tier is elevated by trend projection, not yet by current absolute values (PRD 4.2.5).")

    st.divider()
    st.subheader("Current Parameter Values vs. Corridors")
    corridor_rows = []
    for param in ("HR", "SpO2", "SBP", "DBP", "Temp"):
        lo, hi = fl.P2_CORRIDORS[param]
        val    = current_vals[param]
        dev    = fl._param_deviation(param, val, custom or None)
        corridor_rows.append({
            "Parameter": param, "Value": val,
            "Normal Low": lo, "Normal High": hi,
            "Deviation": round(dev, 3),
        })
    st.dataframe(pd.DataFrame(corridor_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("SBP / DBP / Temp Trend Charts (Phase 2 inputs)")
    for param in ("SBP", "DBP", "Temp"):
        rows_p2 = [{"time": ts, "raw": v, "filtered": v, "tier": "Normal", "rejected": False}
                   for ts, v in series2[param]]
        st.markdown(f"**{param}**")
        st.altair_chart(vital_chart(rows_p2, [], param), use_container_width=True)

    # Phase 2 FHIR write-back
    st.divider()
    st.subheader("Phase 2 FHIR Write-back (RiskAssessment + Flag)")
    if result["tier"] != "Normal" and fl.RANK[result["tier"]] >= fl.RANK[min_tier] and not result.get("fallback"):
        def publish_p2():
            ts_now = fl.now_utc()
            ra = fl.build_risk_assessment(
                pid, result["score"], result["tier"], result["early_warning"],
                result["params_used"], ts_now,
                note="Early-warning from trajectory." if result["early_warning"] else "")
            alert_proxy = {
                "tier": result["tier"], "param": "Composite",
                "value": round(result["score"] * 100, 1), "time": ts_now}
            flag = fl.build_flag(pid, alert_proxy)
            bundle = fl.build_transaction([ra, flag])
            st.session_state.outbox_p2 = bundle
            try:
                if allow_writeback:
                    client.transaction(bundle)
                log_decision("write-back (Phase 2)",
                             f"RiskAssessment {result['tier']} ({result['score']:.3f}) published",
                             [f"Patient/{pid}"])
                st.session_state.p2_msg = ("ok", "Phase 2 RiskAssessment + Flag published.")
            except Exception as e:
                st.session_state.p2_msg = ("err", f"Write-back failed: {e}")

        st.button("Publish Phase 2 RiskAssessment to EHR",
                  on_click=publish_p2, disabled=not allow_writeback)
        if not allow_writeback:
            st.caption("Write-back disabled (sidebar).")
        msg2 = st.session_state.pop("p2_msg", None)
        if msg2:
            (st.success if msg2[0] == "ok" else st.error)(msg2[1])
    else:
        st.info("No Phase 2 alert to publish (tier is Normal or below min tier).")

# ═══════════════════════════════════════════════════════════ TAB 4 — Alert Management
with tab4:
    st.subheader("Alert Response Tracking (PRD 4.4)")

    all_alerts = st.session_state.all_alerts
    if not all_alerts:
        st.info("No alerts yet. Run the monitor tab to generate alerts.")
    else:
        # Per-alert ack/dismiss UI
        st.markdown("#### Unacknowledged Alerts")
        unacked = [a for a in all_alerts if not a.get("ack_ts")]
        if unacked:
            for a in unacked:
                tier_cls = {"High": "alert-high", "Moderate": "alert-mod", "Low": "alert-low"}.get(a["tier"], "")
                ts_str = a["time"].strftime("%H:%M:%S") if hasattr(a["time"], "strftime") else str(a["time"])
                st.markdown(
                    f'<div class="{tier_cls}"><b>{a["tier"]}</b> · {a["param"]} = {a["value"]} '
                    f'· {ts_str} · <code>{a.get("model_version", "")}</code></div>',
                    unsafe_allow_html=True)
                # PRD 4.8.2 explanation
                if a.get("explanation"):
                    st.markdown(f'<div class="explain-box">💡 {a["explanation"]}</div>',
                                unsafe_allow_html=True)
                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
                aid = a["id"]
                if col_a.button("Acknowledge", key=f"ack-{aid}"):
                    fl.ack_alert(a, clinician, "ack")
                    log_decision("ack alert", f"{a['tier']} {a['param']} {a['value']}", [f"Patient/{pid}"])
                    st.rerun()
                reason = col_d.selectbox("Dismiss reason", fl.DISMISS_REASONS, key=f"dr-{aid}", label_visibility="collapsed")
                if col_b.button("Dismiss", key=f"dis-{aid}"):
                    fl.ack_alert(a, clinician, "dismiss", dismiss_reason=reason)
                    log_decision("dismiss alert", f"{a['tier']} {a['param']} reason={reason}", [f"Patient/{pid}"])
                    st.rerun()
                # Snooze only for non-High
                if a["tier"] != "High":
                    if col_c.button("Snooze", key=f"snz-{aid}"):
                        fl.ack_alert(a, clinician, "snooze")
                        log_decision("snooze alert", f"{a['tier']} {a['param']}", [f"Patient/{pid}"])
                        st.rerun()
                else:
                    col_c.caption("🔴 Cannot snooze High")
                # Flag for review
                if not a.get("flagged_for_review"):
                    if st.button("Flag for review", key=f"flg-{aid}"):
                        fl.ack_alert(a, clinician, "flag")
                        log_decision("flag for review", f"{a['tier']} {a['param']}", [f"Patient/{pid}"])
                        st.rerun()
                st.divider()
        else:
            st.success("All alerts acknowledged.")

        # Adjudication section (PRD 4.4)
        st.markdown("#### Adjudicate Outcomes (Training Data, PRD 4.4)")
        non_adjudicated = [a for a in all_alerts if a.get("ack_ts") and not a.get("adjudication")]
        if non_adjudicated:
            for a in non_adjudicated[:10]:   # show up to 10
                ac1, ac2, ac3 = st.columns([2, 1, 1])
                ac1.write(f"{a['tier']} · {a['param']} = {a['value']}")
                if ac2.button("Confirmed deterioration", key=f"adj-yes-{a['id']}"):
                    fl.adjudicate_alert(a, "confirmed_deterioration", clinician)
                    log_decision("adjudication", f"confirmed_deterioration: {a['param']} {a['value']}", [f"Patient/{pid}"])
                    st.rerun()
                if ac3.button("No deterioration", key=f"adj-no-{a['id']}"):
                    fl.adjudicate_alert(a, "no_deterioration", clinician)
                    log_decision("adjudication", f"no_deterioration: {a['param']} {a['value']}", [f"Patient/{pid}"])
                    st.rerun()
        else:
            st.info("No acknowledged alerts awaiting adjudication.")

        # Alarm-fatigue metrics (PRD 4.4)
        st.divider()
        st.subheader("Alarm-Fatigue Metrics (PRD 4.4)")
        metrics = fl.compute_alarm_fatigue_metrics(all_alerts)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Alerts", metrics["total"])
        m2.metric("Ack Rate", f"{metrics['ack_rate']*100:.0f}%" if metrics["ack_rate"] is not None else "—")
        m3.metric("FP Rate", f"{metrics['fp_rate']*100:.0f}%" if metrics["fp_rate"] is not None else "—")
        m4.metric("Median TTA", f"{metrics['median_tta_s']:.0f}s" if metrics["median_tta_s"] is not None else "—")
        m5.metric("Escalation Rate", f"{metrics['escalation_rate']*100:.0f}%" if metrics["escalation_rate"] is not None else "—")

        if metrics["retuning_flag"]:
            st.markdown('<div class="retune-warn">⚠️ False-positive rate exceeds 80% — '
                        'flag this alert type for clinical threshold review (PRD 4.4)</div>',
                        unsafe_allow_html=True)

        if metrics.get("dismissal_reasons"):
            st.write("**Dismissal reason breakdown:**")
            st.bar_chart(metrics["dismissal_reasons"])

# ═══════════════════════════════════════════════════════════ TAB 5 — FHIR Inspector
with tab5:
    st.write("Inbound: FHIR R4 searches. Outbound: transaction Bundle of Observation (LOINC) + Flag (Phase 1) "
             "or RiskAssessment + Flag (Phase 2).")

    f_sub1, f_sub2, f_sub3 = st.tabs(["Patient EHR Resources", "Practitioners & Care Team", "Outbound Transaction Bundle"])

    with f_sub1:
        st.subheader(f"EHR Resources for {fl.patient_name(patient)} ({pid})")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("##### Patient & Conditions")
            st.json({"Patient": patient, "Conditions": conditions}, expanded=False)
        with cb:
            st.markdown("##### Meds, Procedures & Baseline Vitals")
            st.json({
                "MedicationRequests": meds,
                "Procedures": procs,
                "BaselineObservations": baseline_obs,
            }, expanded=False)

    with f_sub2:
        st.subheader("Hospital Medical Staff & Practitioner Directory (FHIR R4)")
        if is_mock:
            pracs_all = client.list_practitioners()
            roles_all = client.list_all_resources("PractitionerRole")
            st.caption(f"Showing {len(pracs_all)} registered medical practitioners and {len(roles_all)} clinical roles on the FHIR server.")

            # Summary table
            prac_rows = []
            for p in pracs_all:
                prac_rows.append({
                    "ID": p["id"],
                    "Name": p["name"][0]["text"],
                    "Department": (p.get("extension") or [{}])[0].get("valueString", "General"),
                    "Role": p.get("qualification", [{}])[0].get("code", {}).get("text", "Physician"),
                    "NPI": (p.get("identifier") or [{}])[0].get("value", "n/a"),
                    "Contact": (p.get("telecom") or [{}])[0].get("value", "n/a"),
                    "Shift": (p.get("extension") or [{}, {}])[1].get("valueString", "On-Duty"),
                })
            st.dataframe(pd.DataFrame(prac_rows), use_container_width=True, hide_index=True)

            with st.expander("Full Practitioner JSON Resources"):
                st.json(pracs_all, expanded=False)
            with st.expander("Full PractitionerRole JSON Resources"):
                st.json(roles_all, expanded=False)
        else:
            st.info("Connected to external FHIR server. Query Practitioner endpoint to inspect staff records.")

    with f_sub3:
        st.subheader("Outbound Transaction Bundle (Last Publish)")
        outbox = st.session_state.get("outbox") or st.session_state.get("outbox_p2")
        if outbox:
            st.json(outbox, expanded=False)
            st.download_button("Download Bundle JSON",
                               json.dumps(outbox, indent=2, default=str),
                               file_name="alerts-transaction-bundle.json",
                               mime="application/fhir+json")
        else:
            st.write("Nothing published yet. Run monitoring and publish alerts to generate outbound FHIR transaction Bundles.")

# ═══════════════════════════════════════════════════════════ TAB 6 — Audit & Decision Log
with tab6:
    st.subheader("Agent & Clinician Decision Log (PRD 4.4)")
    if st.session_state.decision_log:
        st.dataframe(pd.DataFrame(st.session_state.decision_log),
                     use_container_width=True, hide_index=True)
    else:
        st.write("No decisions recorded yet.")

    st.subheader("Full Alert Response Log")
    if st.session_state.all_alerts:
        log_rows = []
        for a in st.session_state.all_alerts:
            log_rows.append({
                "param": a.get("param"), "tier": a.get("tier"),
                "value": a.get("value"),
                "generated": a.get("generated_ts"),
                "ack_action": a.get("ack_action"),
                "responder": a.get("responder"),
                "dismiss_reason": a.get("dismiss_reason"),
                "snoozed": a.get("snoozed"),
                "flagged": a.get("flagged_for_review"),
                "adjudication": a.get("adjudication"),
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No alerts recorded yet.")

    st.subheader("FHIR AuditEvent Resources (PRD 7.2)")
    if st.session_state.audit:
        st.json(st.session_state.audit, expanded=False)
    else:
        st.write("None yet.")
    st.caption("Session-scoped demo log. Production requires the immutable audit store from PRD 7.2.")

# ═══════════════════════════════════════════════════════════ TAB 7 — Lifecycle & Governance
with tab7:
    st.subheader("Development Lifecycle & Governance (PRD 4.7 / 4.8 / 4.9)")
    st.caption("ℹ️ Agents watch and propose; humans approve. Nothing changes a clinical standard automatically (PRD 4.9).")

    # ── PRD 4.9 Agent table
    st.markdown("#### Lifecycle Agent Overview (PRD 4.9)")
    agent_table = [
        {"Agent": "Profiling agent (exists)",             "Watches": "New patient EHR data",
         "Proposes": "Profile, custom range, re-baseline", "Human gate": "Clinician confirm"},
        {"Agent": "Alert-tuning agent",                   "Watches": "FP dismissals, escalation rate",
         "Proposes": "Threshold review request",           "Human gate": "Clinical review committee"},
        {"Agent": "Drift & KPI monitor agent",            "Watches": "Input distributions, sensitivity, calibration",
         "Proposes": "Investigation or retraining request","Human gate": "Clinical + biomed review"},
        {"Agent": "Retraining agent",                     "Watches": "New adjudicated outcomes",
         "Proposes": "Candidate model",                   "Human gate": "Envelope + two sign-offs (PRD 4.7)"},
        {"Agent": "Evidence-pack agent",                  "Watches": "All of the above",
         "Proposes": "Summary for standards review",       "Human gate": "Clinical committee"},
    ]
    st.dataframe(pd.DataFrame(agent_table), use_container_width=True, hide_index=True)

    # ── PRD 4.7 Model Retraining & Dual Sign-Off Governance
    st.divider()
    st.markdown("#### Model Governance: Retraining & PCCP Deployment (PRD 4.7)")
    retrain_agent = st.session_state.retraining_agent

    rg1, rg2, rg3 = st.columns(3)
    rg1.metric("Active Deployed Model", retrain_agent.deployed_version)
    rg2.metric("Deployed AUC", f"{retrain_agent.deployed_metrics['auc']:.3f}")
    rg3.metric("Deployed Sensitivity", f"{retrain_agent.deployed_metrics['sensitivity']*100:.1f}%")

    adjudicated_cases = [a for a in st.session_state.all_alerts if a.get("adjudication")]
    st.write(f"**Human-adjudicated alert outcomes accrued:** `{len(adjudicated_cases)}` cases available for candidate retraining.")

    rc1, rc2 = st.columns([1.5, 3])
    with rc1:
        if st.button("🔄 Propose Candidate Model (Retrain)", type="primary"):
            cand = retrain_agent.train_candidate(st.session_state.all_alerts)
            log_decision("candidate model proposed", f"Version {cand['version']} with {len(adjudicated_cases)} cases", [f"Patient/{pid}"])
            st.toast("Candidate model trained & envelope evaluated.", icon="🧪")
            st.rerun()

    cand = retrain_agent.candidate
    if cand:
        with st.container(border=True):
            st.markdown(f"##### 🧪 Candidate Model: `{cand['version']}`")
            m = cand["metrics"]
            env = cand["envelope_checks"]

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Candidate AUC", f"{m['auc']:.3f}",
                       "✅ PASS (>=0.85)" if env["auc_pass"] else "❌ FAIL (<0.85)")
            cm2.metric("Sensitivity", f"{m['sensitivity']*100:.1f}%",
                       "✅ PASS (>=75%)" if env["sensitivity_pass"] else "❌ FAIL (<75%)")
            cm3.metric("False Alarm Rate", f"{m['false_alarm_rate']*100:.1f}%",
                       "✅ PASS (<=15%)" if env["false_alarm_pass"] else "❌ FAIL (>15%)")
            cm4.metric("Degradation Drop", f"{m['sensitivity_drop']*100:.1f}%",
                       "✅ PASS (<=2%)" if env["degradation_pass"] else "❌ FAIL (>2%)")

            if cand["envelope_passed"]:
                st.success("🎯 Performance Envelope PASSED: Candidate meets all regulatory criteria for clinical deployment.")
            else:
                st.error("🚫 Performance Envelope FAILED: Candidate does not meet safety thresholds. Cannot deploy.")

            # Dual Sign-Off Panel
            st.markdown("###### Required Dual Human Sign-Offs (PRD 4.7):")
            so1, so2 = st.columns(2)
            with so1:
                c_so = retrain_agent.sign_offs["clinician"]
                if c_so:
                    st.markdown(f"<div style='background:#d4edda;border:1px solid #c3e6cb;border-radius:4px;padding:10px;color:#155724'>✅ <b>Clinician Sign-off:</b> {c_so['actor']} ({c_so['role']})<br><small style='color:#6c757d'>{c_so['ts']}</small></div>", unsafe_allow_html=True)
                else:
                    if st.button(f"Sign Off as Clinician ({clinician})", key="sign_clinician"):
                        retrain_agent.sign_off_clinician(clinician, clinician_role)
                        log_decision("clinician sign-off", f"Signed off {cand['version']}", [f"Patient/{pid}"])
                        st.rerun()
            with so2:
                b_so = retrain_agent.sign_offs["biomed"]
                if b_so:
                    st.markdown(f"<div style='background:#d4edda;border:1px solid #c3e6cb;border-radius:4px;padding:10px;color:#155724'>✅ <b>Biomed Sign-off:</b> {b_so['actor']} ({b_so['role']})<br><small style='color:#6c757d'>{b_so['ts']}</small></div>", unsafe_allow_html=True)
                else:
                    biomed_name = st.text_input("Biomed Engineer Name", "Dr. Marcus Vance, PhD, CCE", key="biomed_actor")
                    if st.button("Sign Off as Biomed Lead", key="sign_biomed"):
                        retrain_agent.sign_off_biomed(biomed_name, "Lead Clinical Biomed Engineer")
                        log_decision("biomed sign-off", f"Signed off {cand['version']} by {biomed_name}", [f"Patient/{pid}"])
                        st.rerun()

            # Deployment and Rollback
            dep_col1, dep_col2 = st.columns([1, 1])
            if retrain_agent.ready_for_deployment:
                if dep_col1.button("🚀 Explicitly Deploy to Production Gateway (PRD 4.7)", type="primary"):
                    retrain_agent.deploy_candidate()
                    log_decision("deploy candidate model", f"Deployed {cand['version']}", [f"Patient/{pid}"])
                    st.toast("Model deployed to production gateway!", icon="🎉")
                    st.rerun()
            else:
                dep_col1.caption("⚠️ Deployment locked until envelope passes and BOTH sign-offs are complete.")

            if retrain_agent.archived_models:
                prev_version = retrain_agent.archived_models[-1]["version"]
                if dep_col2.button(f"⏪ Roll Back to Previous Model ({prev_version})"):
                    rolled = retrain_agent.rollback()
                    log_decision("rollback model", f"Rolled back to {rolled}", [f"Patient/{pid}"])
                    st.toast(f"Rolled back to {rolled}", icon="↩️")
                    st.rerun()

    # ── PRD 4.8.1 ML Validation: Subgroup Performance & Calibration
    st.divider()
    st.markdown("#### ML Validation: Subgroup Performance & Calibration (PRD 4.8.1)")
    val_t1, val_t2 = st.tabs(["Subgroup Performance Matrix", "Calibration Analysis"])

    with val_t1:
        st.caption("Subgroup performance reported by age band, sex, chronic status, and baseline maturity (PRD 4.8.1).")
        sub_data = fl.compute_subgroup_metrics()
        sub_df = pd.DataFrame([
            {"Subgroup": k, "N": v["N"], "Sensitivity": f"{v['sensitivity']*100:.1f}%",
             "False Alarm Rate": f"{v['false_alarm_rate']*100:.1f}%", "AUC": f"{v['auc']:.3f}",
             "Lead Time": f"{v['lead_time_min']:.1f} min"}
            for k, v in sub_data.items()
        ])
        st.dataframe(sub_df, use_container_width=True, hide_index=True)

    with val_t2:
        st.caption("Calibration curve: stated probability decile versus observed empirical deterioration rate (PRD 4.8.1).")
        cal_data = fl.compute_calibration_curve()
        cal_df = pd.DataFrame(cal_data)
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

        # Altair calibration plot
        cal_chart = alt.Chart(cal_df).mark_line(point=True, color="#2980b9").encode(
            x=alt.X("mean_predicted:Q", title="Mean Predicted Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed_rate:Q", title="Observed Deterioration Rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["predicted_bin", "mean_predicted", "observed_rate", "samples"]
        ).properties(height=220)
        # 45-degree reference line
        ref_df = pd.DataFrame([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}])
        ref_chart = alt.Chart(ref_df).mark_line(strokeDash=[4, 4], color="#888").encode(x="x:Q", y="y:Q")
        st.altair_chart(ref_chart + cal_chart, use_container_width=True)

    # ── Run lifecycle agents
    alert_tuning = st.session_state.alert_tuning_agent
    drift_mon    = st.session_state.drift_monitor
    ev_agent     = st.session_state.evidence_agent

    st.divider()
    st.markdown("#### Alert-Tuning Agent Proposals")
    new_tuning = alert_tuning.evaluate(st.session_state.all_alerts)
    if alert_tuning.proposals:
        for p in alert_tuning.proposals:
            status_icon = "🟡" if p["status"] == "pending" else "✅"
            st.markdown(f'<div class="agent-prop">{status_icon} <b>{p["agent"]}</b>: {p["title"]}<br>'
                        f'<small>Created: {p["created_ts"]} | Status: {p["status"]}</small></div>',
                        unsafe_allow_html=True)
            ca1, ca2 = st.columns([1, 1])
            if p["status"] == "pending":
                if ca1.button("Accept review request", key=f"at-acc-{p['id']}"):
                    p["status"] = "accepted"
                    log_decision("accept threshold-review", p["title"], [f"Patient/{pid}"])
                    st.rerun()
                if ca2.button("Reject", key=f"at-rej-{p['id']}"):
                    p["status"] = "rejected"
                    log_decision("reject threshold-review", p["title"], [f"Patient/{pid}"])
                    st.rerun()
    else:
        st.info("No alert-tuning proposals yet (FP rate below 80% threshold, or not enough data).")

    # ── Drift monitor
    st.divider()
    st.markdown("#### Drift & KPI Monitor Agent")
    snap = drift_mon.snapshot(st.session_state.all_alerts)
    new_drift = drift_mon.evaluate()
    if drift_mon.proposals:
        for p in drift_mon.proposals:
            st.markdown(f'<div class="agent-prop">🟠 <b>{p["agent"]}</b>: {p["title"]}</div>',
                        unsafe_allow_html=True)
    else:
        st.info(f"No drift proposals. {len(drift_mon.snapshots)} KPI snapshot(s) recorded so far.")
    with st.expander("Last KPI snapshot"):
        st.json(snap)

    # ── Evidence pack
    st.divider()
    st.markdown("#### Evidence Pack (PRD 4.9)")
    pack = ev_agent.build_pack(alert_tuning, drift_mon, st.session_state.decision_log)
    st.info(pack["summary"])
    with st.expander("Full evidence pack JSON"):
        st.json(pack)
    st.download_button("Download evidence pack",
                       json.dumps(pack, indent=2, default=str),
                       file_name="evidence-pack.json", mime="application/json")

    # ── Model card (PRD 4.8.2)
    st.divider()
    st.markdown("#### Model Card (PRD 4.8.2)")
    mc = fl.MODEL_CARD
    mc1, mc2 = st.columns(2)
    with mc1:
        st.write(f"**Model version:** `{mc['model_version']}`")
        st.write(f"**Intended use:** {mc['intended_use']}")
        st.write("**Excluded populations:** " + ", ".join(mc["excluded_populations"]))
        st.write(f"**Training data:** {mc['training_data']}")
        st.write("**PCCP boundary:** " + mc["pccp_boundary"])
        st.write("**Regulatory alignment:** " + mc["regulatory_alignment"])
    with mc2:
        st.write("**Performance (overall):**")
        st.json(mc["performance_overall"])
        st.write(f"**Subgroup performance:** {mc['performance_by_subgroup']}")
        st.write("**Known limitations:**")
        for lim in mc["known_limitations"]:
            st.markdown(f"- {lim}")
        st.write("**Guardrails:**")
        for g in mc["guardrails"]:
            st.markdown(f"- ✅ {g}")



