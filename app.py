import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="AI Motor Digital Twin",
    page_icon="⚙️",
    layout="wide"
)

# -------------------------------------------------
# LOAD MOTOR DATA
# -------------------------------------------------
df = pd.read_csv("motor_data.csv")

# Load trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# -------------------------------------------------
# MOTOR CONDITION SIMULATION
# -------------------------------------------------

st.sidebar.header("🧪 Motor Simulation")

selected_condition = st.sidebar.selectbox(
    "Select Motor Condition",
    [
        "Normal",
        "Overheating",
        "Overloading",
        "Mechanical Fault"
    ]
)

st.sidebar.subheader("🎛️ What-If Simulation")

temperature_input = st.sidebar.slider(
    "Temperature (°C)",
    20.0, 80.0, 36.0
)

current_input = st.sidebar.slider(
    "Current (A)",
    2.0, 6.0, 3.2
)

vibration_input = st.sidebar.slider(
    "Vibration",
    0.0, 1.5, 0.18
)

rpm_input = st.sidebar.slider(
    "RPM",
    1200, 1500, 1475
)

# Representative simulated motor values
simulation_data = {
    "Normal": {
        "voltage": 415.5,
        "current": 3.19,
        "temperature": 36.0,
        "vibration": 0.18,
        "rpm": 1475,
        "load": 60.6
    },

    "Overheating": {
        "voltage": 415.0,
        "current": 4.01,
        "temperature": 61.9,
        "vibration": 0.30,
        "rpm": 1430,
        "load": 70.5
    },

    "Overloading": {
        "voltage": 414.8,
        "current": 5.19,
        "temperature": 68.1,
        "vibration": 0.39,
        "rpm": 1383,
        "load": 85.1
    },

    "Mechanical Fault": {
        "voltage": 414.8,
        "current": 3.79,
        "temperature": 45.2,
        "vibration": 0.90,
        "rpm": 1373,
        "load": 64.7
    }
}

# Select simulated motor values
motor = pd.Series(simulation_data[selected_condition])

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("⚙️ AI-Enabled Digital Twin")
st.subheader("Three-Phase Induction Motor")

st.divider()

# -------------------------------------------------
# AI MOTOR CONDITION PREDICTION
# -------------------------------------------------

ai_input = pd.DataFrame({
    "temperature": [temperature_input],
    "current": [current_input],
    "vibration": [vibration_input],
    "rpm": [rpm_input]
})

condition = model.predict(ai_input)[0]

probabilities = model.predict_proba(ai_input)[0]

confidence = max(probabilities) * 100

# -------------------------------------------------
# VIRTUAL MOTOR
# -------------------------------------------------

st.header("Virtual Motor")

col_motor, col_status = st.columns([2, 1])

# Condition-reactive motor display
if condition == "Normal":
    motor_status = "🟢 NORMAL"
    motor_symbol = "⚙️"
    status_message = "Motor operating normally"

elif condition == "Overheating":
    motor_status = "🟠 OVERHEATING"
    motor_symbol = "⚙️"
    status_message = "High temperature detected"

elif condition == "Overloading":
    motor_status = "🟠 OVERLOADING"
    motor_symbol = "⚙️"
    status_message = "High motor load detected"

else:
    motor_status = "🔴 MECHANICAL FAULT"
    motor_symbol = "⚙️"
    status_message = "Abnormal vibration detected"

with col_motor:
    st.markdown(
        f"""
        <div style="
            border: 3px solid #888;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            background-color: #202020;
        ">

        <h2>⚡ THREE-PHASE MOTOR ⚡</h2>

        <div style="
            margin: 25px auto;
            width: 350px;
            height: 160px;
            border: 5px solid #aaa;
            border-radius: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">

        <div style="
            width: 110px;
            height: 110px;
            border: 8px solid #aaa;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 45px;
        ">
        {motor_symbol}
        </div>

        </div>

        <h3>{motor_status}</h3>

        <p>● L1 &nbsp;&nbsp; ● L2 &nbsp;&nbsp; ● L3</p>

        <p>Three-Phase Electrical Supply → Motor</p>

        <p>{status_message}</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with col_status:
    st.subheader("Motor Status")
    st.metric("AI Condition", condition)

with col_status:

    st.subheader("Motor Status")

# Prototype motor health score
health_scores = {
    "Normal": 95,
    "Overheating": 65,
    "Overloading": 55,
    "Mechanical Fault": 35
}

health_score = health_scores.get(condition, 50)

if condition == "Normal":
    st.success("🟢 MOTOR RUNNING")
    st.success("Condition: NORMAL")

elif condition == "Overheating":
    st.warning("🟠 WARNING")
    st.warning("Condition: OVERHEATING")

elif condition == "Overloading":
    st.warning("🟠 WARNING")
    st.warning("Condition: OVERLOADING")

else:
    st.error("🔴 FAULT")
    st.error("Condition: MECHANICAL FAULT")

st.subheader("🤖 AI Prediction")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.metric("Predicted Condition", condition)

with pred_col2:
    st.metric("AI Confidence", f"{confidence:.1f}%")

with pred_col3:
    st.metric("Motor Health", f"{health_score}/100")

st.progress(
    health_score / 100,
    text=f"Motor Health: {health_score}%"
)

st.metric("Operating Speed", f"{rpm_input:.0f} RPM")

st.progress(health_score / 100, text=f"Motor Health: {health_score}%")

# -------------------------------------------------
# MAINTENANCE RECOMMENDATION
# -------------------------------------------------

st.subheader("🔧 Maintenance Recommendation")

recommendations = {
    "Normal": {
        "message": "Motor operating normally. Continue routine monitoring.",
        "type": "success"
    },

    "Overheating": {
        "message": "Check motor cooling system and monitor temperature.",
        "type": "warning"
    },

    "Overloading": {
        "message": "Check motor load and current consumption.",
        "type": "warning"
    },

    "Mechanical Fault": {
        "message": "Inspect bearings, shaft alignment and mechanical components.",
        "type": "error"
    }
}

recommendation = recommendations.get(
    condition,
    {
        "message": "Inspect motor condition.",
        "type": "warning"
    }
)

if recommendation["type"] == "success":
    st.success("✅ " + recommendation["message"])

elif recommendation["type"] == "warning":
    st.warning("⚠️ " + recommendation["message"])

else:
    st.error("🚨 " + recommendation["message"])

# -------------------------------------------------
# AI ALERT
# -------------------------------------------------

if condition == "Normal":
    st.info("🟢 No maintenance action required.")

elif condition == "Overheating":
    st.warning("⚠️ ALERT: High motor temperature detected.")

elif condition == "Overloading":
    st.warning("⚠️ ALERT: Excessive motor load detected.")

else:
    st.error("🚨 ALERT: Possible mechanical fault detected.")

# -------------------------------------------------
# MOTOR PARAMETERS
# -------------------------------------------------
st.divider()

st.header("Real-Time Motor Parameters")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric("Voltage", f"{motor['voltage']:.2f} V")

    st.metric("Temperature", f"{temperature_input:.2f} °C")

with col2:

    st.metric("Current", f"{current_input:.2f} A")
    st.metric("Vibration", f"{vibration_input:.2f}")

with col3:

    st.metric("RPM", f"{rpm_input:.0f}")
    simulated_load = 60 + (current_input - 3.2) * 15 + (1475 - rpm_input) * 0.03
    simulated_load = max(0, min(100, simulated_load))

    st.metric("Load", f"{simulated_load:.1f} %")

# -------------------------------------------------
# MOTOR TREND GRAPHS
# -------------------------------------------------
st.divider()

st.header("📊 Motor Parameter Trends")

# Temperature
fig_temperature = px.line(
    df,
    x="timestamp",
    y="temperature",
    title="🌡️ Temperature vs Time",
    labels={
        "timestamp": "Time / Sample",
        "temperature": "Temperature (°C)"
    }
)

st.plotly_chart(
    fig_temperature,
    use_container_width=True
)

# Current
fig_current = px.line(
    df,
    x="timestamp",
    y="current",
    title="⚡ Current vs Time",
    labels={
        "timestamp": "Time / Sample",
        "current": "Current (A)"
    }
)

st.plotly_chart(
    fig_current,
    use_container_width=True
)

# Vibration
fig_vibration = px.line(
    df,
    x="timestamp",
    y="vibration",
    title="📳 Vibration vs Time",
    labels={
        "timestamp": "Time / Sample",
        "vibration": "Vibration"
    }
)

st.plotly_chart(
    fig_vibration,
    use_container_width=True
)

# RPM
fig_rpm = px.line(
    df,
    x="timestamp",
    y="rpm",
    title="⚙️ RPM vs Time",
    labels={
        "timestamp": "Time / Sample",
        "rpm": "RPM"
    }
)

st.plotly_chart(
    fig_rpm,
    use_container_width=True
)

# Load
fig_load = px.line(
    df,
    x="timestamp",
    y="load",
    title="📊 Load vs Time",
    labels={
        "timestamp": "Time / Sample",
        "load": "Load (%)"
    }
)

st.plotly_chart(
    fig_load,
    use_container_width=True
)