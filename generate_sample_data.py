import numpy as np
import pandas as pd

np.random.seed(42)

data = []

# Generate 250 samples for each motor condition
for condition in ["Normal", "Overheating", "Overloading", "Mechanical Fault"]:

    for i in range(250):

        voltage = np.random.normal(415, 3)

        if condition == "Normal":
            current = np.random.normal(3.2, 0.25)
            temperature = np.random.normal(36, 2)
            vibration = np.random.normal(0.18, 0.04)
            rpm = np.random.normal(1475, 15)
            load = np.random.normal(60, 5)

        elif condition == "Overheating":
            current = np.random.normal(4.0, 0.3)
            temperature = np.random.normal(62, 4)
            vibration = np.random.normal(0.30, 0.06)
            rpm = np.random.normal(1430, 20)
            load = np.random.normal(70, 5)

        elif condition == "Overloading":
            current = np.random.normal(5.2, 0.4)
            temperature = np.random.normal(68, 4)
            vibration = np.random.normal(0.40, 0.08)
            rpm = np.random.normal(1385, 25)
            load = np.random.normal(85, 5)

        else:  # Mechanical Fault
            current = np.random.normal(3.8, 0.3)
            temperature = np.random.normal(45, 3)
            vibration = np.random.normal(0.90, 0.12)
            rpm = np.random.normal(1375, 30)
            load = np.random.normal(65, 6)

        data.append([
            i + 1,
            voltage,
            current,
            temperature,
            vibration,
            rpm,
            load,
            condition
        ])

# Create DataFrame
columns = [
    "timestamp",
    "voltage",
    "current",
    "temperature",
    "vibration",
    "rpm",
    "load",
    "condition"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("motor_data.csv", index=False)

print("Sample motor dataset created successfully!")
print(f"Total samples: {len(df)}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\nCondition distribution:")
print(df["condition"].value_counts())