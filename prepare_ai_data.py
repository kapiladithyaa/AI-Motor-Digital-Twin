import pandas as pd

# Load motor dataset
df = pd.read_csv("motor_data.csv")

# Input features for the AI model
features = [
    "temperature",
    "current",
    "vibration",
    "rpm"
]

# Separate input and output
X = df[features]
y = df["condition"]

print("AI Input Features:")
print(X.head())

print("\nTarget Output:")
print(y.head())

print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nConditions:")
print(y.unique())