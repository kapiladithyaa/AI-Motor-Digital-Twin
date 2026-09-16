import pandas as pd
import joblib

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

model = joblib.load("random_forest_model.pkl")

# -------------------------------------------------
# TEST CASES
# -------------------------------------------------

test_data = pd.DataFrame({
    "Test": [
        "Normal Test",
        "Overheating Test",
        "Overloading Test",
        "Mechanical Fault Test"
    ],

    "temperature": [
        36.0,
        61.9,
        68.1,
        45.2
    ],

    "current": [
        3.2,
        4.0,
        5.2,
        3.8
    ],

    "vibration": [
        0.18,
        0.30,
        0.40,
        0.90
    ],

    "rpm": [
        1475,
        1430,
        1385,
        1375
    ],

    "Expected": [
        "Normal",
        "Overheating",
        "Overloading",
        "Mechanical Fault"
    ]
})

# -------------------------------------------------
# AI PREDICTION
# -------------------------------------------------

features = [
    "temperature",
    "current",
    "vibration",
    "rpm"
]

predictions = model.predict(test_data[features])

probabilities = model.predict_proba(test_data[features])

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print("\n==============================================")
print("       AI MOTOR CONDITION TESTING")
print("==============================================")

correct = 0

for i in range(len(test_data)):

    predicted = predictions[i]
    confidence = max(probabilities[i]) * 100
    expected = test_data["Expected"][i]

    if predicted == expected:
        result = "PASS"
        correct += 1
    else:
        result = "FAIL"

    print("\nTest:", test_data["Test"][i])
    print("Expected Condition :", expected)
    print("AI Prediction       :", predicted)
    print(f"AI Confidence       : {confidence:.2f}%")
    print("Result              :", result)

# -------------------------------------------------
# TEST SUMMARY
# -------------------------------------------------

accuracy = (correct / len(test_data)) * 100

print("\n==============================================")
print("TEST SUMMARY")
print("==============================================")

print("Total Tests :", len(test_data))
print("Passed      :", correct)
print("Failed      :", len(test_data) - correct)
print(f"Test Accuracy : {accuracy:.2f}%")

print("==============================================")