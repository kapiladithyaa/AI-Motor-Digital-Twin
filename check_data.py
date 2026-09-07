import pandas as pd

# Load the dataset
df = pd.read_csv("motor_data.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCondition distribution:")
print(df["condition"].value_counts())

print("\nAverage values for each condition:")
print(
    df.groupby("condition")[[
        "voltage",
        "current",
        "temperature",
        "vibration",
        "rpm",
        "load"
    ]].mean().round(2)
)