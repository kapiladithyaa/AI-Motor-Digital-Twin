import streamlit as st

import pandas as pd

import numpy as np

import joblib

import plotly.graph_objects as go

import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(

    page_title="MotorTwin | AI Predictive Maintenance",

    page_icon="⚙️",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ============================================================

# LOAD DATA

# ============================================================

@st.cache_data

def load_data():

    return pd.read_csv("motor_data.csv")

@st.cache_resource

def load_model_package():

    return joblib.load("motor_condition_model.pkl")

df = load_data()

model_package = load_model_package()

model = model_package["model"]

model_features = model_package["features"]

# ============================================================

# MOTOR PRESETS

# ============================================================

MOTOR_PRESETS = {

    "Normal": {

        "voltage": 415.0,

        "current": 0.749,

        "temperature": 37.527,

        "vibration": 0.401,

        "rpm": 1248,

        "load": 50.054

    },

    "Overheating": {

        "voltage": 415.0,

        "current": 1.128,

        "temperature": 46.824,

        "vibration": 0.740,

        "rpm": 1057,

        "load": 78.640

    },

    "Overloading": {

        "voltage": 415.0,

        "current": 0.799,

        "temperature": 40.586,

        "vibration": 0.534,

        "rpm": 1264,

        "load": 37.341

    },

    "Mechanical Fault": {

        "voltage": 415.0,

        "current": 1.078,

        "temperature": 49.172,

        "vibration": 0.887,

        "rpm": 1116,

        "load": 47.243

    }

}

# ============================================================

# ============================================================

# CSS — MODERN INDUSTRIAL CONTROL ROOM UI

# ============================================================

st.markdown(

    """

    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {

        --bg: #07111f;

        --panel: rgba(14, 29, 49, 0.82);

        --panel-strong: rgba(17, 37, 61, 0.94);

        --line: rgba(127, 177, 224, 0.18);

        --text: #eaf4ff;

        --muted: #8ba5c2;

        --cyan: #4dd9f5;

        --blue: #6e9cff;

        --mint: #5ce6b1;

        --amber: #ffc46b;

        --red: #ff7185;

    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {

        color: var(--text);

        background:

          radial-gradient(ellipse at 10% -10%, rgba(55, 120, 255, .19), transparent 38%),

          radial-gradient(ellipse at 95% 12%, rgba(32, 220, 207, .10), transparent 30%),

          linear-gradient(145deg, #050b15 0%, #081626 48%, #07111f 100%);

    }

    .main .block-container { max-width: 1600px; padding-top: 1.35rem; padding-bottom: 2.5rem; }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; color: #f2f8ff; }

    h1 { font-size: clamp(1.8rem, 3vw, 2.55rem) !important; }

    h2 { font-size: 1.55rem !important; }

    h3 { font-size: 1.15rem !important; }

    p, label, .stMarkdown { color: #d8e7f8; }

    [data-testid="stHeader"] { background: rgba(5, 12, 23, .55); }

    [data-testid="stToolbar"] { right: 1rem; }

    /* Sidebar: control-room navigation */

    section[data-testid="stSidebar"] {

        background: linear-gradient(180deg, rgba(9, 22, 38, .98), rgba(5, 13, 24, .99));

        border-right: 1px solid var(--line);

    }

    section[data-testid="stSidebar"] > div { padding-top: 1.15rem; }

    section[data-testid="stSidebar"] * { color: #dcecff; }

    section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .35rem; }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {

        padding: .58rem .7rem; border: 1px solid transparent; border-radius: 11px;

        transition: all .18s ease; background: rgba(255,255,255,.015);

    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {

        background: rgba(77,217,245,.08); border-color: rgba(77,217,245,.2);

    }

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,

    section[data-testid="stSidebar"] [data-testid="stSlider"] {

        border-radius: 12px;

    }

    /* Glass panels and hero header */

    .top-header {

        position: relative; overflow: hidden;

        background: linear-gradient(120deg, rgba(18, 45, 76, .96), rgba(8, 24, 42, .95));

        border: 1px solid rgba(100, 180, 255, .26); border-radius: 22px;

        padding: clamp(22px, 3vw, 34px); margin: 0 0 24px 0;

        box-shadow: 0 18px 50px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.04);

    }

    .top-header:after {

        content: ''; position: absolute; width: 300px; height: 300px; right: -80px; top: -170px;

        border-radius: 50%; background: rgba(77,217,245,.12); filter: blur(2px); pointer-events: none;

    }

    .project-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(23px, 2.5vw, 34px); line-height: 1.2; font-weight: 700; color: #f6fbff; max-width: 900px; }

    .project-subtitle { font-size: 13px; color: #a2c1df; margin-top: 10px; letter-spacing: .02em; }

    .online-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 0 5px rgba(92,230,177,.1), 0 0 18px rgba(92,230,177,.7); margin-right: 10px; }

    .online { color: var(--mint); font-weight: 700; font-size: 12px; letter-spacing: .11em; }

    .panel, .ai-card {

        background: linear-gradient(145deg, rgba(15, 34, 56, .91), rgba(8, 21, 37, .92));

        border: 1px solid var(--line); border-radius: 18px; padding: 20px;

        box-shadow: 0 12px 34px rgba(0,0,0,.17), inset 0 1px 0 rgba(255,255,255,.025);

        backdrop-filter: blur(12px);

    }

    .panel-title { color: #e9f5ff; font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700; letter-spacing: -.02em; }

    .panel-subtitle { color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.6; }

    .ai-card { background: linear-gradient(130deg, rgba(11, 54, 78, .78), rgba(10, 28, 48, .95)); border-color: rgba(77,217,245,.24); }

    .ai-title { color: var(--cyan); font-size: 12px; font-weight: 700; letter-spacing: .13em; }

    .ai-condition { color: #f3f9ff; font-family: 'Space Grotesk', sans-serif; font-size: 23px; font-weight: 700; margin-top: 5px; }

    /* KPI cards */

    .kpi-card {

        position: relative; overflow: hidden; height: 100%; min-height: 122px;

        background: linear-gradient(150deg, rgba(18, 42, 68, .96), rgba(9, 24, 41, .96));

        border: 1px solid rgba(125, 175, 224, .18); border-radius: 16px; padding: 18px 17px;

        box-shadow: 0 9px 25px rgba(0,0,0,.14); transition: transform .2s ease, border-color .2s ease;

    }

    .kpi-card:hover { transform: translateY(-3px); border-color: rgba(77,217,245,.38); }

    .kpi-card:after { content: ''; position: absolute; width: 90px; height: 90px; right: -45px; bottom: -55px; border-radius: 50%; background: rgba(77,217,245,.09); }

    .kpi-icon { float: right; font-size: 22px; filter: drop-shadow(0 0 10px rgba(77,217,245,.25)); }

    .kpi-label { color: #9ab3ce; font-size: 11px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; }

    .kpi-value { color: #f4f9ff; font-family: 'Space Grotesk', sans-serif; font-size: clamp(21px, 2vw, 28px); font-weight: 700; margin-top: 10px; line-height: 1.15; }

    .kpi-sub { color: #7894b3; font-size: 10px; margin-top: 8px; }

    /* Status states */

    .status-normal, .status-warning, .status-danger { border-radius: 13px; padding: 14px; font-weight: 600; }

    .status-normal { color: var(--mint); background: rgba(39,213,146,.08); border: 1px solid rgba(92,230,177,.25); }

    .status-warning { color: var(--amber); background: rgba(255,184,77,.08); border: 1px solid rgba(255,196,107,.24); }

    .status-danger { color: var(--red); background: rgba(255,83,111,.08); border: 1px solid rgba(255,113,133,.25); }

    /* Inputs, sliders, buttons */

    .stButton > button, .stDownloadButton > button {

        border: 1px solid rgba(77,217,245,.28); border-radius: 11px;

        background: linear-gradient(120deg, rgba(29,92,130,.86), rgba(26,58,99,.9));

        color: #f4fbff; font-weight: 700; min-height: 42px; transition: all .18s ease;

    }

    .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--cyan); color: white; transform: translateY(-1px); box-shadow: 0 7px 20px rgba(26,173,220,.14); }

    [data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 4px rgba(77,217,245,.13); }

    [data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {

        background: rgba(8, 21, 37, .85); border-color: rgba(127,177,224,.22); border-radius: 10px; color: #edf7ff;

    }

    [data-testid="stMetric"] { background: rgba(15,34,56,.66); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }

    [data-testid="stMetricLabel"] { color: #9ab3ce; }

    [data-testid="stMetricValue"] { color: #f3f9ff; font-family: 'Space Grotesk', sans-serif; }

    /* Charts, tables, expanders, alerts */

    [data-testid="stPlotlyChart"] { border-radius: 16px; overflow: hidden; }

    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; }

    [data-testid="stExpander"] { background: rgba(13,31,51,.72); border: 1px solid var(--line); border-radius: 13px; }

    [data-testid="stAlert"] { border-radius: 13px; }

    hr { border-color: rgba(127,177,224,.15); }

    .footer { text-align: center; color: #7894b3; font-size: 12px; padding: 24px 12px; margin-top: 28px; border-top: 1px solid rgba(127,177,224,.12); }

    ::-webkit-scrollbar { width: 8px; height: 8px; }

    ::-webkit-scrollbar-track { background: #07111f; }

    ::-webkit-scrollbar-thumb { background: #25415f; border-radius: 8px; }

    ::-webkit-scrollbar-thumb:hover { background: #3b6a8e; }

    @media (max-width: 900px) {

        .main .block-container { padding-left: 1rem; padding-right: 1rem; }

        .top-header { border-radius: 16px; }

        .kpi-card { min-height: 110px; padding: 14px; }

        .panel, .ai-card { padding: 15px; }

    }

    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; } }

    </style>

    """,

    unsafe_allow_html=True

)

# ============================================================

# SIDEBAR

# ============================================================

with st.sidebar:

    # Logo

    st.html(

        """

        <div style="

            padding:8px 5px 20px 5px;

            border-bottom:

                1px solid rgba(100,160,220,0.18);

            margin-bottom:20px;

        ">

            <div style="

                font-size:27px;

                font-weight:750;

                color:#f0f7ff;

            ">

                ⚙️ MotorTwin

            </div>

            <div style="

                font-size:12px;

                color:#7898b9;

                margin-top:5px;

            ">

                AI-Powered Digital Twin

            </div>

        </div>

        """

    )

    st.markdown("### Navigation")

    # IMPORTANT:

    # This navigation controls which page is displayed.

    page = st.radio(

        "",

        [

            "🏠 Dashboard",

            "📡 Live Monitoring",

            "🎛️ What-If Simulation",

            "🧠 AI Analysis",

            "📈 Trend Analysis",

            "ℹ️ About"

        ],

        label_visibility="collapsed"

    )

    st.markdown("---")

    # ========================================================

    # SIMULATION CONTROLS

    # ========================================================

    st.markdown("### ⚙️ Simulation Control")

    selected_condition = st.selectbox(

        "Motor Condition",

        list(MOTOR_PRESETS.keys())

    )

    preset = MOTOR_PRESETS[

        selected_condition

    ]

    # Session values

    if "voltage_input" not in st.session_state:

        st.session_state.voltage_input = float(preset["voltage"])

    if "temperature_input" not in st.session_state:

        st.session_state.temperature_input = (

            preset["temperature"]

        )

    if "current_input" not in st.session_state:

        st.session_state.current_input = (

            preset["current"]

        )

    if "vibration_input" not in st.session_state:

        st.session_state.vibration_input = (

            preset["vibration"]

        )

    if "rpm_input" not in st.session_state:

        st.session_state.rpm_input = (

            preset["rpm"]

        )

    if "load_input" not in st.session_state:

        st.session_state.load_input = preset["load"]

    live_simulation = st.toggle(
        "🔴 Enable Live Simulation",
        value=st.session_state.get("live_simulation", False),
        key="live_simulation"
    )

    if live_simulation:
        refresh_count = st_autorefresh(interval=2000, key="motor_live_refresh")
        # Generate a new plausible sensor reading every 2 seconds.
        # The trained model receives these values below on each rerun.
        rng = np.random.default_rng()
        st.session_state.voltage_input = float(np.clip(
            st.session_state.voltage_input + rng.normal(0, 2.0), 380, 450
        ))
        st.session_state.current_input = float(np.clip(
            st.session_state.current_input + rng.normal(0, 0.035), 0.3, 1.5
        ))
        st.session_state.temperature_input = float(np.clip(
            st.session_state.temperature_input + rng.normal(0, 0.7), 20, 80
        ))
        st.session_state.vibration_input = float(np.clip(
            st.session_state.vibration_input + rng.normal(0, 0.025), 0, 1.5
        ))
        st.session_state.rpm_input = int(np.clip(
            st.session_state.rpm_input + rng.normal(0, 8), 850, 1500
        ))
        st.session_state.load_input = float(np.clip(
            st.session_state.load_input + rng.normal(0, 1.5), 10, 90
        ))

    if st.button(

        "Load Selected Condition",

        use_container_width=True

    ):

        st.session_state.voltage_input = float(preset["voltage"])

        st.session_state.temperature_input = (

            preset["temperature"]

        )

        st.session_state.current_input = (

            preset["current"]

        )

        st.session_state.vibration_input = (

            preset["vibration"]

        )

        st.session_state.rpm_input = (

            preset["rpm"]

        )

        st.session_state.load_input = preset["load"]

        st.rerun()


    voltage_input = st.slider(

        "🔌 Voltage (V)",

        min_value=380.0,

        max_value=450.0,

        step=1.0,

        key="voltage_input"

    )

    temperature_input = st.slider(

        "🌡️ Temperature (°C)",

        20.0,

        80.0,

        key="temperature_input"

    )

    current_input = st.slider(

        "⚡ Current (A)",

        min_value=0.3,

        max_value=1.5,

        value=float(st.session_state.current_input),

        key="current_input"

    )

    vibration_input = st.slider(

        "〰️ Vibration (mm/s)",

        0.0,

        1.5,

        key="vibration_input"

    )

    rpm_input = st.slider(

        "🔄 Speed (RPM)",

        min_value=850,

        max_value=1500,

        value=int(st.session_state.rpm_input),

        key="rpm_input"

    )

    load_input = st.slider(

        "⚙️ Load (%)",

        min_value=10.0,

        max_value=90.0,

        value=float(st.session_state.load_input),

        key="load_input"

    )

    st.markdown("---")

    st.html(

        """

        <div style="

            color:#6f8eae;

            font-size:12px;

            line-height:1.7;

        ">

            <b style="color:#dcecff;">

                Simulation Mode

            </b>

            <br>

            Software-based motor simulation

            <br>

            AI prediction enabled

        </div>

        """

    )

# ============================================================

# AI PREDICTION — MEMBER 2 MODEL

# ============================================================

sensor_values = {

    "voltage": voltage_input,

    "current": current_input,

    "temperature": temperature_input,

    "vibration": vibration_input,

    "rpm": rpm_input,

    "load": load_input

}

ai_input = pd.DataFrame([sensor_values])

# Ensure the inputs match the trained model's feature order

ai_input = ai_input[model_features]

condition = model.predict(ai_input)[0]

probabilities = model.predict_proba(ai_input)[0]

confidence = max(probabilities) * 100

# ============================================================

# HEALTH SCORE

# ============================================================

health_scores = {

    "Normal": 95,

    "Overheating": 65,

    "Overloading": 55,

    "Mechanical Fault": 35

}

health = health_scores.get(

    condition,

    50

)

# ============================================================

# LOAD CALCULATION

# ============================================================

simulated_load = load_input

# ============================================================

# HELPER: STATUS COLOR

# ============================================================

if health >= 80:

    health_color = "#25e77f"

elif health >= 50:

    health_color = "#ffb84d"

else:

    health_color = "#ff5662"

if condition == "Normal":

    status_class = "status-normal"

elif condition in [

    "Overheating",

    "Overloading"

]:

    status_class = "status-warning"

else:

    status_class = "status-danger"

# ============================================================

# HEADER

# ============================================================

st.html(

    """

    <div class="top-header">

        <div class="project-title">

            AI Predictive Maintenance System for Industrial Motors

        </div>

        <div class="project-subtitle">

            Real-time Monitoring

            &nbsp; | &nbsp;

            AI-based Fault Detection

            &nbsp; | &nbsp;

            Digital Twin Simulation

        </div>

        <div style="

            margin-top:14px;

        ">

            <span class="online-dot"></span>

            <span class="online">

                SYSTEM ONLINE

            </span>

            <span style="

                color:#6f8cae;

                margin-left:15px;

            ">

                Simulation Mode

            </span>

        </div>

    </div>

    """

)

# ============================================================

# KPI FUNCTION

# ============================================================

def kpi_card(

    icon,

    label,

    value,

    sub

):

    st.html(

        f"""

        <div class="kpi-card">

            <div class="kpi-icon">

                {icon}

            </div>

            <div class="kpi-label">

                {label}

            </div>

            <div class="kpi-value">

                {value}

            </div>

            <div class="kpi-sub">

                {sub}

            </div>

        </div>

        """

    )

# ============================================================

# KPI ROW

# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:

    kpi_card(

        "⚡",

        "Voltage",

        f"{voltage_input:.0f} V",

        "Nominal: 415 V"

    )

with k2:

    kpi_card(

        "〰️",

        "Current",

        f"{current_input:.2f} A",

        "Dataset-based simulation value"

    )

with k3:

    kpi_card(

        "🌡️",

        "Temperature",

        f"{temperature_input:.1f} °C",

        "Normal: < 60 °C"

    )

with k4:

    kpi_card(

        "〰️",

        "Vibration",

        f"{vibration_input:.2f}",

        "mm/s | Normal: < 0.5"

    )

with k5:

    kpi_card(

        "🔄",

        "Speed",

        f"{rpm_input}",

        "RPM | Nominal: 1440–1500"

    )

with k6:

    kpi_card(

        "📊",

        "Load",

        f"{simulated_load:.0f}%",

        "Simulated Motor Load"

    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================

# 3D MOTOR FUNCTIONS

# ============================================================

def cylinder_surface(

    x_start,

    x_end,

    radius,

    theta_start=0,

    theta_end=2 * np.pi,

    segments=60

):

    theta = np.linspace(

        theta_start,

        theta_end,

        segments

    )

    x_values = np.array(

        [x_start, x_end]

    )

    X = np.tile(

        x_values.reshape(-1, 1),

        (1, segments)

    )

    Y = radius * np.tile(

        np.cos(theta),

        (2, 1)

    )

    Z = radius * np.tile(

        np.sin(theta),

        (2, 1)

    )

    return X, Y, Z

def add_cylinder(

    fig,

    x1,

    x2,

    radius,

    color_scale,

    theta1=0,

    theta2=2 * np.pi

):

    X, Y, Z = cylinder_surface(

        x1,

        x2,

        radius,

        theta1,

        theta2

    )

    fig.add_trace(

        go.Surface(

            x=X,

            y=Y,

            z=Z,

            colorscale=color_scale,

            showscale=False,

            hoverinfo="skip"

        )

    )

def add_ring(

    fig,

    x,

    radius,

    color="#8ab7d8",

    width=6

):

    theta = np.linspace(

        0,

        2 * np.pi,

        100

    )

    fig.add_trace(

        go.Scatter3d(

            x=np.full_like(

                theta,

                x

            ),

            y=radius * np.cos(theta),

            z=radius * np.sin(theta),

            mode="lines",

            line=dict(

                color=color,

                width=width

            ),

            hoverinfo="skip",

            showlegend=False

        )

    )

def create_motor_3d():

    fig = go.Figure()

    # ========================================================

    # MAIN BLUE MOTOR BODY

    #

    # Leave the front-facing section open so the inside can

    # be seen.

    # ========================================================

    add_cylinder(

        fig,

        -2.65,

        2.65,

        1.55,

        [

            [0, "#123653"],

            [0.45, "#1d5680"],

            [0.75, "#2b75a6"],

            [1, "#4b98c8"]

        ],

        theta1=np.deg2rad(48),

        theta2=np.deg2rad(312)

    )

    # ========================================================

    # LEFT CLOSED END COVER

    # ========================================================

    add_cylinder(

        fig,

        -2.95,

        -2.55,

        1.68,

        [

            [0, "#12324f"],

            [0.5, "#235d87"],

            [1, "#4a91bd"]

        ]

    )

    # ========================================================

    # LEFT FRONT RINGS

    # ========================================================

    add_ring(

        fig,

        -2.98,

        1.45,

        "#73acd0",

        5

    )

    add_ring(

        fig,

        -3.01,

        0.95,

        "#6d9fbe",

        4

    )

    # ========================================================

    # INNER STATOR CORE

    # ========================================================

    add_cylinder(

        fig,

        -2.15,

        2.10,

        1.22,

        [

            [0, "#69747b"],

            [0.5, "#aab3b8"],

            [1, "#d6dcdf"]

        ],

        theta1=np.deg2rad(48),

        theta2=np.deg2rad(312)

    )

    # ========================================================

    # ROTOR

    # ========================================================

    add_cylinder(

        fig,

        -2.05,

        2.25,

        0.82,

        [

            [0, "#626b71"],

            [0.45, "#aeb6bb"],

            [0.75, "#d2d7da"],

            [1, "#777f84"]

        ]

    )

    # ========================================================

    # ROTOR SHAFT

    # ========================================================

    add_cylinder(

        fig,

        -3.45,

        4.20,

        0.29,

        [

            [0, "#4e585f"],

            [0.45, "#dce1e4"],

            [0.65, "#aeb7bc"],

            [1, "#59636a"]

        ]

    )

    # ========================================================

    # SHAFT COLLAR

    # ========================================================

    add_cylinder(

        fig,

        2.35,

        2.95,

        0.47,

        [

            [0, "#4e5961"],

            [0.5, "#d1d7da"],

            [1, "#68747b"]

        ]

    )

    # ========================================================

    # COPPER WINDINGS

    # ========================================================

    theta = np.linspace(

        0,

        2 * np.pi,

        100

    )

    coil_positions = [

        -1.55,

        -0.75,

        0.15,

        1.05,

        1.75

    ]

    for x_position in coil_positions:

        y = 1.34 * np.cos(theta)

        z = 1.34 * np.sin(theta)

        fig.add_trace(

            go.Scatter3d(

                x=np.full_like(

                    theta,

                    x_position

                ),

                y=y,

                z=z,

                mode="lines",

                line=dict(

                    color="#c97924",

                    width=9

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # COPPER CONNECTIONS

    # ========================================================

    for x_position in [

        -1.55,

        1.05

    ]:

        z_curve = np.linspace(

            -1.2,

            1.25,

            60

        )

        y_curve = np.full_like(

            z_curve,

            1.28

        )

        x_curve = np.full_like(

            z_curve,

            x_position

        )

        fig.add_trace(

            go.Scatter3d(

                x=x_curve,

                y=y_curve,

                z=z_curve,

                mode="lines",

                line=dict(

                    color="#b86a24",

                    width=11

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # RED INSULATION BLOCKS

    # ========================================================

    for x_position in [

        -1.55,

        1.05

    ]:

        z_values = np.array(

            [-1.35, -0.85]

        )

        fig.add_trace(

            go.Scatter3d(

                x=[

                    x_position,

                    x_position

                ],

                y=[

                    1.28,

                    1.28

                ],

                z=z_values,

                mode="lines",

                line=dict(

                    color="#d9382e",

                    width=18

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # COOLING FAN

    # ========================================================

    fan_center_x = 2.72

    fan_radius = 1.10

    for blade_angle in np.linspace(

        0,

        2 * np.pi,

        8,

        endpoint=False

    ):

        r1 = 0.25

        r2 = 1.0

        a1 = blade_angle

        a2 = blade_angle + 0.32

        y_values = [

            r1 * np.cos(a1),

            r2 * np.cos(a2)

        ]

        z_values = [

            r1 * np.sin(a1),

            r2 * np.sin(a2)

        ]

        fig.add_trace(

            go.Scatter3d(

                x=[

                    fan_center_x,

                    fan_center_x

                ],

                y=y_values,

                z=z_values,

                mode="lines",

                line=dict(

                    color="#dfe7eb",

                    width=14

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # FAN CENTER

    # ========================================================

    add_ring(

        fig,

        fan_center_x,

        0.22,

        "#e1e7ea",

        10

    )

    # ========================================================

    # CLOSED RIGHT FAN GUARD

    # ========================================================

    add_ring(

        fig,

        2.95,

        1.55,

        "#4d9aca",

        8

    )

    # ========================================================

    # VENTILATION GRILL

    # ========================================================

    for angle in np.linspace(

        0,

        2 * np.pi,

        14,

        endpoint=False

    ):

        y1 = 1.18 * np.cos(angle)

        z1 = 1.18 * np.sin(angle)

        y2 = 1.48 * np.cos(angle)

        z2 = 1.48 * np.sin(angle)

        fig.add_trace(

            go.Scatter3d(

                x=[

                    2.98,

                    2.98

                ],

                y=[

                    y1,

                    y2

                ],

                z=[

                    z1,

                    z2

                ],

                mode="lines",

                line=dict(

                    color="#0c2840",

                    width=5

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # TERMINAL BOX

    # ========================================================

    fig.add_trace(

        go.Mesh3d(

            x=[

                -0.9, 0.8, 0.8, -0.9,

                -0.9, 0.8, 0.8, -0.9

            ],

            y=[

                -0.48, -0.48, 0.48, 0.48,

                -0.48, -0.48, 0.48, 0.48

            ],

            z=[

                1.35, 1.35, 1.35, 1.35,

                2.15, 2.15, 2.15, 2.15

            ],

            i=[0, 0, 1, 4, 4, 5],

            j=[1, 2, 3, 5, 6, 6],

            k=[2, 3, 2, 6, 7, 7],

            color="#27658f",

            opacity=1,

            hoverinfo="skip",

            showlegend=False

        )

    )

    # ========================================================

    # LIFTING EYE

    # ========================================================

    eye_theta = np.linspace(

        0,

        2 * np.pi,

        80

    )

    fig.add_trace(

        go.Scatter3d(

            x=np.zeros_like(

                eye_theta

            ),

            y=0.30 * np.cos(

                eye_theta

            ),

            z=2.55 + 0.30 * np.sin(

                eye_theta

            ),

            mode="lines",

            line=dict(

                color="#b8c5cf",

                width=8

            ),

            hoverinfo="skip",

            showlegend=False

        )

    )

    # ========================================================

    # COOLING FINS

    # ========================================================

    for x_position in np.linspace(

        -2.2,

        2.15,

        12

    ):

        fig.add_trace(

            go.Scatter3d(

                x=[

                    x_position,

                    x_position

                ],

                y=[

                    1.45,

                    1.68

                ],

                z=[

                    0,

                    0

                ],

                mode="lines",

                line=dict(

                    color="#0a263e",

                    width=8

                ),

                hoverinfo="skip",

                showlegend=False

            )

        )

    # ========================================================

    # 3D SCENE

    # ========================================================

    fig.update_layout(

        height=540,

        margin=dict(

            l=0,

            r=0,

            t=0,

            b=0

        ),

        paper_bgcolor="#081827",

        scene=dict(

            bgcolor="#081827",

            xaxis=dict(

                visible=False,

                showgrid=False,

                zeroline=False

            ),

            yaxis=dict(

                visible=False,

                showgrid=False,

                zeroline=False

            ),

            zaxis=dict(

                visible=False,

                showgrid=False,

                zeroline=False

            ),

            aspectmode="manual",

            aspectratio=dict(

                x=2.7,

                y=1.5,

                z=1.5

            ),

            camera=dict(

                eye=dict(

                    x=1.6,

                    y=1.5,

                    z=1.0

                )

            )

        ),

        showlegend=False

    )

    return fig

# ============================================================

# MAINTENANCE RECOMMENDATION

# ============================================================

recommendations = {

    "Normal": (

        "Routine monitoring",

        "No immediate action required. Continue regular monitoring."

    ),

    "Overheating": (

        "Check cooling system",

        "Inspect ventilation, cooling fan and temperature trend."

    ),

    "Overloading": (

        "Check motor load",

        "Inspect load, current level and operating conditions."

    ),

    "Mechanical Fault": (

        "Inspect mechanical components",

        "Check bearings, shaft alignment and vibration."

    )

}

rec_title, rec_text = recommendations.get(

    condition,

    (

        "Monitor motor",

        "Continue monitoring motor parameters."

    )

)

# ============================================================

# PAGE 1 — DASHBOARD

# ============================================================

if page == "🏠 Dashboard":

    left, right = st.columns(

        [2.15, 1]

    )

    # ========================================================

    # 3D MOTOR

    # ========================================================

    with left:

        st.html(

            """

            <div class="panel">

                <div class="panel-title">

                    ⚙️ Physical Motor + Holographic Digital Twin

                </div>

                <div class="panel-subtitle">

                    Visual representation

                    &nbsp; | &nbsp;

                    AI-powered condition monitoring

                </div>

            </div>

            """

        )

        st.image(

            "digital_twin_hero.png",

            use_container_width=True

        )

        st.caption(

            "Physical motor + holographic digital twin | "

            "AI-powered condition monitoring"

        )

    # ========================================================

    # RIGHT PANEL

    # ========================================================

    with right:

        # ----------------------------------------------------

        # HEALTH

        # ----------------------------------------------------

        st.html(

            f"""

            <div class="panel">

                <div class="panel-title">

                    Motor Health

                </div>

                <div style="

                    text-align:center;

                    padding:18px 0;

                ">

                    <div style="

                        width:170px;

                        height:170px;

                        margin:auto;

                        border-radius:50%;

                        background:

                        conic-gradient(

                            {health_color} {health}%,

                            #172c40 {health}% 100%

                        );

                        display:flex;

                        align-items:center;

                        justify-content:center;

                    ">

                        <div style="

                            width:135px;

                            height:135px;

                            border-radius:50%;

                            background:#081827;

                            display:flex;

                            flex-direction:column;

                            align-items:center;

                            justify-content:center;

                        ">

                            <div style="

                                font-size:45px;

                                font-weight:750;

                                color:{health_color};

                            ">

                                {health}

                            </div>

                            <div style="

                                color:#7693b3;

                                font-size:12px;

                            ">

                                / 100

                            </div>

                        </div>

                    </div>

                </div>

                <div class="{status_class}">

                    <b>

                        ● {condition.upper()}

                    </b>

                    <div style="

                        margin-top:7px;

                        color:#9bb4cd;

                        font-size:12px;

                    ">

                        Motor condition detected by AI model.

                    </div>

                </div>

            </div>

            """

        )

        st.markdown(

            "<br>",

            unsafe_allow_html=True

        )

        # ----------------------------------------------------

        # AI PREDICTION

        # ----------------------------------------------------

        st.html(

            f"""

            <div class="ai-card">

                <div class="ai-title">

                    🧠 AI PREDICTION

                </div>

                <div style="

                    display:flex;

                    justify-content:space-between;

                    align-items:center;

                    margin-top:12px;

                ">

                    <div>

                        <div style="

                            color:#7898b8;

                            font-size:12px;

                        ">

                            Predicted Condition

                        </div>

                        <div class="ai-condition">

                            🟢 {condition}

                        </div>

                    </div>

                    <div style="

                        text-align:right;

                    ">

                        <div style="

                            color:#7898b8;

                            font-size:12px;

                        ">

                            Confidence

                        </div>

                        <div style="

                            font-size:24px;

                            font-weight:750;

                            color:#67c7ff;

                        ">

                            {confidence:.1f}%

                        </div>

                    </div>

                </div>

            </div>

            """

        )

        st.markdown(

            "<br>",

            unsafe_allow_html=True

        )

        # ----------------------------------------------------

        # MAINTENANCE

        # ----------------------------------------------------

        st.html(

            f"""

            <div class="panel">

                <div class="panel-title">

                    🔧 Maintenance Recommendation

                </div>

                <div style="

                    margin-top:12px;

                    color:{health_color};

                    font-size:17px;

                    font-weight:700;

                ">

                    {rec_title}

                </div>

                <div style="

                    margin-top:7px;

                    color:#8ba6c2;

                    font-size:12px;

                    line-height:1.6;

                ">

                    {rec_text}

                </div>

            </div>

            """

        )

# ============================================================

# PAGE 2 — LIVE MONITORING

# ============================================================

elif page == "📡 Live Monitoring":

    st.title("📡 Live Motor Monitoring")

    st.write(

        "Monitor the simulated operating parameters "

        "of the three-phase induction motor."

    )

    # --------------------------------------------------------

    # PARAMETER CARDS

    # --------------------------------------------------------

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(

            "Voltage",

            f"{voltage_input:.0f} V"

        )

    with a2:

        st.metric(

            "Current",

            f"{current_input:.2f} A"

        )

    with a3:

        st.metric(

            "Temperature",

            f"{temperature_input:.1f} °C"

        )

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(

            "Vibration",

            f"{vibration_input:.2f} mm/s"

        )

    with b2:

        st.metric(

            "Speed",

            f"{rpm_input} RPM"

        )

    with b3:

        st.metric(

            "Motor Load",

            f"{simulated_load:.0f}%"

        )

    st.markdown("---")

    # --------------------------------------------------------

    # GAUGES

    # --------------------------------------------------------

    g1, g2 = st.columns(2)

    with g1:

        temperature_gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=temperature_input,

                title={

                    "text": "Temperature °C"

                },

                gauge={

                    "axis": {

                        "range": [20, 80]

                    },

                    "bar": {

                        "color": "#ff5662"

                    },

                    "bgcolor": "#13283b",

                    "borderwidth": 0

                }

            )

        )

        temperature_gauge.update_layout(

            height=350,

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white"

        )

        st.plotly_chart(

            temperature_gauge,

            use_container_width=True

        )

    with g2:

        load_gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=simulated_load,

                title={

                    "text": "Motor Load %"

                },

                gauge={

                    "axis": {

                        "range": [0, 100]

                    },

                    "bar": {

                        "color": "#329cff"

                    },

                    "bgcolor": "#13283b",

                    "borderwidth": 0

                }

            )

        )

        load_gauge.update_layout(

            height=350,

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white"

        )

        st.plotly_chart(

            load_gauge,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("Current Operating Status")

    if condition == "Normal":

        st.success(

            "Motor is operating within the simulated normal condition."

        )

    elif condition == "Overheating":

        st.warning(

            "Elevated temperature detected by the AI model."

        )

    elif condition == "Overloading":

        st.warning(

            "High load/current operating condition detected."

        )

    else:

        st.error(

            "Possible mechanical fault detected."

        )

# ============================================================

# PAGE 3 — WHAT-IF SIMULATION

# ============================================================

elif page == "🎛️ What-If Simulation":

    st.title("🎛️ What-If Motor Simulation")

    st.write(

        """

        Adjust the motor parameters from the sidebar.

        The Random Forest model immediately evaluates the

        modified operating condition.

        """

    )

    st.info(
        "Simulation mode: changing a sidebar control reruns the Python backend, "
        "passes all six current values to the Random Forest model, and refreshes "
        "the prediction and dashboard outputs."
    )

    # --------------------------------------------------------

    # RESULT

    # --------------------------------------------------------

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(

            "Predicted Condition",

            condition

        )

    with r2:

        st.metric(

            "AI Confidence",

            f"{confidence:.1f}%"

        )

    with r3:

        st.metric(

            "Motor Health",

            f"{health}/100"

        )

    st.markdown("---")

    # --------------------------------------------------------

    # CURRENT INPUTS

    # --------------------------------------------------------

    st.subheader(

        "Current Simulation Inputs"

    )

    input_df = pd.DataFrame(

        {

            "Parameter": [

                "Voltage", "Current", "Temperature", "Vibration", "RPM", "Load"

            ],

            "Value": [

                f"{voltage_input:.0f} V",

                f"{current_input:.2f} A",

                f"{temperature_input:.1f} °C",

                f"{vibration_input:.2f} mm/s",

                f"{rpm_input} RPM",

                f"{load_input:.1f}%"

            ]

        }

    )

    st.dataframe(

        input_df,

        hide_index=True,

        use_container_width=True

    )

    st.markdown("---")

    # --------------------------------------------------------

    # AI MESSAGE

    # --------------------------------------------------------

    if condition == "Normal":

        st.success(

            "AI assessment: Normal operating condition."

        )

    elif condition == "Overheating":

        st.warning(

            "AI assessment: Overheating condition."

        )

    elif condition == "Overloading":

        st.warning(

            "AI assessment: Overloading condition."

        )

    else:

        st.error(

            "AI assessment: Mechanical fault condition."

        )

    # --------------------------------------------------------

    # 3D MOTOR

    # --------------------------------------------------------

    st.subheader(

        "Digital Twin Response"

    )

    st.plotly_chart(

        create_motor_3d(),

        use_container_width=True,

        config={

            "displaylogo": False,

            "scrollZoom": True,

            "responsive": True

        }

    )

# ============================================================

# PAGE 4 — AI ANALYSIS

# ============================================================

elif page == "🧠 AI Analysis":

    st.title("🧠 AI Motor Condition Analysis")

    # --------------------------------------------------------

    # MAIN RESULT

    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Predicted Condition",

            condition

        )

    with c2:

        st.metric(

            "AI Confidence",

            f"{confidence:.2f}%"

        )

    with c3:

        st.metric(

            "Motor Health",

            f"{health}/100"

        )

    st.markdown("---")

    # --------------------------------------------------------

    # PROBABILITY

    # --------------------------------------------------------

    st.subheader(

        "AI Condition Probability"

    )

    probability_data = pd.DataFrame(

        {

            "Condition":

                model.classes_,

            "Probability":

                probabilities * 100

        }

    )

    fig_probability = px.bar(

        probability_data,

        x="Condition",

        y="Probability",

        text="Probability"

    )

    fig_probability.update_traces(

        texttemplate="%{text:.1f}%",

        textposition="outside"

    )

    fig_probability.update_layout(

        height=430,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff",

        yaxis_title="Probability (%)"

    )

    st.plotly_chart(

        fig_probability,

        use_container_width=True

    )

    # --------------------------------------------------------

    # FEATURE IMPORTANCE

    # --------------------------------------------------------

    st.subheader(

        "Feature Importance"

    )

    feature_names = [

        "Voltage",

        "Current",

        "Temperature",

        "Vibration",

        "RPM",

        "Load"

    ]

    importance_data = pd.DataFrame(

        {

            "Feature":

                feature_names,

            "Importance":

                model.feature_importances_

        }

    )

    importance_data = (

        importance_data

        .sort_values(

            "Importance",

            ascending=True

        )

    )

    fig_importance = px.bar(

        importance_data,

        x="Importance",

        y="Feature",

        orientation="h",

        text="Importance"

    )

    fig_importance.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside"

    )

    fig_importance.update_layout(

        height=400,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff",

        xaxis_title="Importance"

    )

    st.plotly_chart(

        fig_importance,

        use_container_width=True

    )

# ============================================================

# PAGE 5 — TREND ANALYSIS

# ============================================================

elif page == "📈 Trend Analysis":

    st.title("📈 Motor Trend Analysis")

    st.write(

        "Historical parameter trends from the motor dataset."

    )

    trend_df = df.tail(100)

    # --------------------------------------------------------

    # TEMPERATURE

    # --------------------------------------------------------

    fig_temperature = px.line(

        trend_df,

        x="timestamp",

        y="temperature",

        title="Temperature Trend"

    )

    fig_temperature.update_layout(

        height=350,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff"

    )

    st.plotly_chart(

        fig_temperature,

        use_container_width=True

    )

    # --------------------------------------------------------

    # CURRENT

    # --------------------------------------------------------

    fig_current = px.line(

        trend_df,

        x="timestamp",

        y="current",

        title="Current Trend"

    )

    fig_current.update_layout(

        height=350,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff"

    )

    st.plotly_chart(

        fig_current,

        use_container_width=True

    )

    # --------------------------------------------------------

    # VIBRATION

    # --------------------------------------------------------

    fig_vibration = px.line(

        trend_df,

        x="timestamp",

        y="vibration",

        title="Vibration Trend"

    )

    fig_vibration.update_layout(

        height=350,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff"

    )

    st.plotly_chart(

        fig_vibration,

        use_container_width=True

    )

    # --------------------------------------------------------

    # RPM

    # --------------------------------------------------------

    fig_rpm = px.line(

        trend_df,

        x="timestamp",

        y="rpm",

        title="Motor Speed Trend"

    )

    fig_rpm.update_layout(

        height=350,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#dcecff"

    )

    st.plotly_chart(

        fig_rpm,

        use_container_width=True

    )

# ============================================================

# PAGE 6 — ABOUT

# ============================================================

elif page == "ℹ️ About":

    st.title(

        "ℹ️ About MotorTwin"

    )

    st.markdown(

        """

        ## AI-Enabled Digital Twin for Industrial Motors

        MotorTwin is a software-based Digital Twin platform

        designed for monitoring and predictive maintenance of

        three-phase induction motors.

        ### Core Technologies

        **Digital Twin**

        A virtual representation of the industrial motor

        used to visualize its operating condition.

        **Machine Learning**

        A Random Forest classifier analyzes motor parameters

        and predicts the operating condition.

        **Condition Monitoring**

        The system monitors:

        - Temperature

        - Current

        - Vibration

        - RPM

        **Predictive Maintenance**

        The AI prediction is used to provide a

        maintenance recommendation.

        ### Detected Conditions

        - Normal

        - Overheating

        - Overloading

        - Mechanical Fault

        ### Project Architecture

        Motor Parameters

        ↓

        Data Processing

        ↓

        Random Forest Model

        ↓

        Condition Prediction

        ↓

        Digital Twin

        ↓

        Health Score

        ↓

        Maintenance Recommendation

        """

    )

# ============================================================

# FOOTER

# ============================================================

st.divider()

st.html(

    """

    <div class="footer">

        ⚙️

        <b style="color:#9bb6d2;">

            AI Predictive Maintenance System for Industrial Motors

        </b>

        <br><br>

        Bannari Amman Institute of Technology

        &nbsp; | &nbsp;

        Department of Electrical & Electronics Engineering

        <br><br>

        Digital Twin

        &nbsp; • &nbsp;

        Machine Learning

        &nbsp; • &nbsp;

        Predictive Maintenance

    </div>

    """

)