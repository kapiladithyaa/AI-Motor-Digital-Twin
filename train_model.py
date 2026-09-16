import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

df = pd.read_csv("motor_data.csv")

print("==============================================")
print("       AI MOTOR CONDITION MODEL TRAINING")
print("==============================================")

print("\nDataset Shape:")
print(df.shape)

# -------------------------------------------------
# INPUT FEATURES
# -------------------------------------------------

features = [
    "temperature",
    "current",
    "vibration",
    "rpm"
]

X = df[features]
y = df["condition"]

print("\nInput Features:")
print(features)

print("\nMotor Conditions:")
print(y.unique())

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# -------------------------------------------------
# RANDOM FOREST MODEL
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed.")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------------
# ACCURACY
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================================")
print("MODEL PERFORMANCE")
print("==============================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# -------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save classification report
report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "classification_report.csv"
)

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.to_csv(
    "confusion_matrix.csv"
)

# Create confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Random Forest Motor Condition Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================")

print(feature_importance)

# Save feature importance
feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

# -------------------------------------------------
# SAVE TRAINED MODEL
# -------------------------------------------------

joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("\n==============================================")
print("MODEL SAVING")
print("==============================================")

print("Model saved successfully as:")
print("random_forest_model.pkl")

print("\nAdditional result files created:")
print("classification_report.csv")
print("confusion_matrix.csv")
print("feature_importance.csv")
print("confusion_matrix.png")

print("\n==============================================")
print("       TRAINING AND EVALUATION COMPLETE")
print("==============================================")