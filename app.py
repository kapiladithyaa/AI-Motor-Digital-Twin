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

# Use first reading for the virtual motor
motor = df.iloc[0]

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("⚙️ AI-Enabled Digital Twin")
st.subheader("Three-Phase Induction Motor")

st.divider()

# -------------------------------------------------
# VIRTUAL MOTOR
# -------------------------------------------------
st.header("Virtual Motor")

col_motor, col_status = st.columns([2, 1])

with col_motor:

    st.markdown(
        """
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
        ⚙️
        </div>

        </div>

        <p>● L1 &nbsp;&nbsp; ● L2 &nbsp;&nbsp; ● L3</p>

        <p>
        Three-Phase Electrical Supply → Motor
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

with col_status:

    st.subheader("Motor Status")

# -------------------------------------------------
# AI MOTOR CONDITION PREDICTION
# -------------------------------------------------

ai_input = pd.DataFrame({
    "temperature": [motor["temperature"]],
    "current": [motor["current"]],
    "vibration": [motor["vibration"]],
    "rpm": [motor["rpm"]]
})

# Predict motor condition using Random Forest
condition = model.predict(ai_input)[0]

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

st.metric(
    "Operating Speed",
    f"{motor['rpm']:.0f} RPM"
)

# -------------------------------------------------
# MOTOR PARAMETERS
# -------------------------------------------------
st.divider()

st.header("Real-Time Motor Parameters")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Voltage",
        f"{motor['voltage']:.2f} V"
    )

    st.metric(
        "Temperature",
        f"{motor['temperature']:.2f} °C"
    )

with col2:

    st.metric(
        "Current",
        f"{motor['current']:.2f} A"
    )

    st.metric(
        "Vibration",
        f"{motor['vibration']:.2f}"
    )

with col3:

    st.metric(
        "RPM",
        f"{motor['rpm']:.0f}"
    )

    st.metric(
        "Load",
        f"{motor['load']:.1f} %"
    )

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