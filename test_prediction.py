import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# New motor sensor readings
new_motor_data = pd.DataFrame({
    "temperature": [36, 62, 68, 45],
    "current": [3.2, 4.0, 5.2, 3.8],
    "vibration": [0.18, 0.30, 0.40, 0.90],
    "rpm": [1475, 1430, 1385, 1375]
})

# Prediction probability
predictions = model.predict(new_motor_data)

print("\nPredictions:")

for i, prediction in enumerate(predictions):
    print(f"Test {i + 1}: {prediction}")