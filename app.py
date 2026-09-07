import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Motor Digital Twin",
    page_icon="⚙️",
    layout="wide"
)

# Title
st.title("⚙️ AI-Enabled Digital Twin")
st.subheader("Three-Phase Induction Motor")

# Load motor dataset
df = pd.read_csv("motor_data.csv")

# Select the first motor reading
motor = df.iloc[0]

# Display motor parameters
st.header("Motor Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Voltage", f"{motor['voltage']:.2f} V")
    st.metric("Temperature", f"{motor['temperature']:.2f} °C")

with col2:
    st.metric("Current", f"{motor['current']:.2f} A")
    st.metric("Vibration", f"{motor['vibration']:.2f}")

with col3:
    st.metric("RPM", f"{motor['rpm']:.0f}")
    st.metric("Load", f"{motor['load']:.1f} %")

# Motor condition
st.header("Motor Condition")

condition = motor["condition"]

if condition == "Normal":
    st.success("Motor Status: NORMAL")

elif condition == "Overheating":
    st.warning("Motor Status: OVERHEATING")

elif condition == "Overloading":
    st.warning("Motor Status: OVERLOADING")

else:
    st.error("Motor Status: MECHANICAL FAULT")