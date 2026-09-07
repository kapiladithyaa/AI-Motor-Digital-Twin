import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("motor_data.csv")

# Temperature comparison
fig1 = px.box(
    df,
    x="condition",
    y="temperature",
    title="Temperature Distribution by Motor Condition"
)
fig1.show()

# Current comparison
fig2 = px.box(
    df,
    x="condition",
    y="current",
    title="Current Distribution by Motor Condition"
)
fig2.show()

# Vibration comparison
fig3 = px.box(
    df,
    x="condition",
    y="vibration",
    title="Vibration Distribution by Motor Condition"
)
fig3.show()

# RPM comparison
fig4 = px.box(
    df,
    x="condition",
    y="rpm",
    title="RPM Distribution by Motor Condition"
)
fig4.show()