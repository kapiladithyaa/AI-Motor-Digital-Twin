import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px


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
def load_model():
    return joblib.load("random_forest_model.pkl")


df = load_data()
model = load_model()


# ============================================================
# MOTOR PRESETS
# ============================================================

MOTOR_PRESETS = {

    "Normal": {
        "voltage": 415,
        "current": 3.2,
        "temperature": 36.0,
        "vibration": 0.18,
        "rpm": 1475
    },

    "Overheating": {
        "voltage": 415,
        "current": 4.0,
        "temperature": 61.9,
        "vibration": 0.30,
        "rpm": 1430
    },

    "Overloading": {
        "voltage": 415,
        "current": 5.2,
        "temperature": 68.1,
        "vibration": 0.40,
        "rpm": 1385
    },

    "Mechanical Fault": {
        "voltage": 415,
        "current": 3.8,
        "temperature": 45.2,
        "vibration": 0.90,
        "rpm": 1375
    }
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 5%,
                rgba(0, 115, 255, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 15% 80%,
                rgba(0, 210, 180, 0.05),
                transparent 28%
            ),
            #06111d;

        color: #e8f1ff;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #071522 0%,
                #081b2c 55%,
                #06111d 100%
            );

        border-right:
            1px solid rgba(75, 160, 230, 0.22);
    }

    section[data-testid="stSidebar"] * {
        color: #dcecff;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .top-header {

        background:
            linear-gradient(
                135deg,
                rgba(11, 36, 59, 0.98),
                rgba(6, 20, 35, 0.98)
            );

        border:
            1px solid rgba(75, 160, 230, 0.25);

        border-radius: 18px;

        padding: 23px 28px;

        margin-bottom: 18px;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.22);
    }

    .project-title {

        font-size: 29px;

        font-weight: 750;

        color: #f5f9ff;
    }

    .project-subtitle {

        font-size: 14px;

        color: #82a7cf;

        margin-top: 6px;
    }

    .online-dot {

        display: inline-block;

        width: 10px;

        height: 10px;

        border-radius: 50%;

        background: #28e783;

        box-shadow:
            0 0 12px #28e783;

        margin-right: 8px;
    }

    .online {

        color: #2ee986;

        font-weight: 700;

        font-size: 13px;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {

        background:
            linear-gradient(
                145deg,
                rgba(13, 37, 60, 0.98),
                rgba(7, 20, 34, 0.98)
            );

        border:
            1px solid rgba(80, 155, 225, 0.23);

        border-radius: 14px;

        padding: 16px;

        min-height: 105px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.16);
    }

    .kpi-icon {

        float: right;

        font-size: 23px;
    }

    .kpi-label {

        color: #86a5c5;

        font-size: 12px;
    }

    .kpi-value {

        color: #f4f8ff;

        font-size: 26px;

        font-weight: 750;

        margin-top: 5px;
    }

    .kpi-sub {

        color: #6685a7;

        font-size: 10px;

        margin-top: 5px;
    }


    /* ========================================================
       PANELS
       ======================================================== */

    .panel {

        background:
            linear-gradient(
                145deg,
                rgba(10, 31, 49, 0.98),
                rgba(6, 18, 31, 0.98)
            );

        border:
            1px solid rgba(70, 150, 230, 0.20);

        border-radius: 16px;

        padding: 18px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.18);
    }

    .panel-title {

        color: #dcecff;

        font-size: 20px;

        font-weight: 700;
    }

    .panel-subtitle {

        color: #718eae;

        font-size: 12px;

        margin-top: 4px;
    }


    /* ========================================================
       AI CARD
       ======================================================== */

    .ai-card {

        background:
            linear-gradient(
                135deg,
                rgba(7, 40, 63, 0.98),
                rgba(7, 23, 39, 0.98)
            );

        border:
            1px solid rgba(45, 174, 255, 0.25);

        border-radius: 14px;

        padding: 17px;
    }

    .ai-title {

        color: #63c5ff;

        font-size: 13px;

        font-weight: 700;
    }

    .ai-condition {

        color: #f3f8ff;

        font-size: 23px;

        font-weight: 750;

        margin-top: 5px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-normal {

        color: #31e98a;

        background:
            rgba(35, 220, 125, 0.08);

        border:
            1px solid rgba(35, 220, 125, 0.22);

        border-radius: 10px;

        padding: 12px;
    }

    .status-warning {

        color: #ffb84d;

        background:
            rgba(255, 184, 77, 0.08);

        border:
            1px solid rgba(255, 184, 77, 0.22);

        border-radius: 10px;

        padding: 12px;
    }

    .status-danger {

        color: #ff5e68;

        background:
            rgba(255, 75, 85, 0.08);

        border:
            1px solid rgba(255, 75, 85, 0.22);

        border-radius: 10px;

        padding: 12px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        border-radius: 9px;

        border:
            1px solid rgba(65, 160, 255, 0.30);

        background:
            rgba(20, 61, 95, 0.75);

        color: white;

        font-weight: 600;
    }

    .stButton > button:hover {

        border-color: #329cff;

        background:
            rgba(25, 87, 135, 0.90);
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align: center;

        color: #6d89a6;

        font-size: 12px;

        padding: 20px;

        margin-top: 25px;
    }

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


    if st.button(
        "Load Selected Condition",
        use_container_width=True
    ):

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

        st.rerun()


    temperature_input = st.slider(
        "🌡️ Temperature (°C)",
        20.0,
        80.0,
        key="temperature_input"
    )


    current_input = st.slider(
        "⚡ Current (A)",
        2.0,
        6.0,
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
        1200,
        1500,
        key="rpm_input"
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
# AI PREDICTION
# ============================================================

ai_input = pd.DataFrame(
    {
        "temperature": [temperature_input],
        "current": [current_input],
        "vibration": [vibration_input],
        "rpm": [rpm_input]
    }
)


condition = model.predict(
    ai_input
)[0]


probabilities = model.predict_proba(
    ai_input
)[0]


confidence = (
    max(probabilities) * 100
)


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

simulated_load = (

    60

    + (current_input - 3.2) * 15

    + (1475 - rpm_input) * 0.03
)


simulated_load = max(
    0,
    min(
        100,
        simulated_load
    )
)


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
        "415 V",
        "Nominal: 415 V"
    )


with k2:

    kpi_card(
        "〰️",
        "Current",
        f"{current_input:.2f} A",
        "Nominal: 3.0 – 4.5 A"
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
                    ⚙️ 3D Motor Digital Twin
                </div>

                <div class="panel-subtitle">
                    Interactive cutaway motor
                    &nbsp; | &nbsp;
                    Drag to rotate
                    &nbsp; | &nbsp;
                    Scroll to zoom
                    &nbsp; | &nbsp;
                    Shift + drag to pan
                </div>

            </div>
            """
        )


        motor_fig = create_motor_3d()


        st.plotly_chart(
            motor_fig,

            use_container_width=True,

            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True
            }
        )


        st.caption(
            "🖱️ Drag = Rotate  •  "
            "Scroll = Zoom  •  "
            "Shift + Drag = Pan  •  "
            "Double Click = Reset"
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
            "415 V"
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
                "Temperature",
                "Current",
                "Vibration",
                "RPM"
            ],

            "Value": [
                f"{temperature_input:.1f} °C",
                f"{current_input:.2f} A",
                f"{vibration_input:.2f} mm/s",
                f"{rpm_input} RPM"
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
        "Temperature",
        "Current",
        "Vibration",
        "RPM"
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