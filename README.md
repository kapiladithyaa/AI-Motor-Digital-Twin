# AI Predictive Maintenance System for Industrial Motors Using Digital Twin

An AI-powered predictive maintenance solution that uses Machine Learning and Digital Twin technology to monitor industrial motor parameters and identify potential faults.

## 📌 Project Overview

Industrial motors are essential in manufacturing and automation. Unexpected failures can cause downtime, increased maintenance costs, and reduced productivity.

This project combines a Random Forest machine learning model with an interactive Streamlit dashboard to analyze motor operating parameters and predict abnormal conditions.

## 🎯 Objectives

- Monitor industrial motor operating parameters.
- Identify potential motor faults using Machine Learning.
- Visualize motor conditions through an interactive dashboard.
- Demonstrate the Digital Twin concept for motor condition monitoring.
- Support predictive maintenance and reduce unexpected downtime.

## ✨ Key Features

- **Interactive Dashboard:** Displays motor information and prediction results.
- **AI-Based Fault Prediction:** Uses a trained Random Forest classifier.
- **Motor Parameter Analysis:** Analyzes voltage, current, temperature, vibration, RPM, and load.
- **Fault Identification:** Classifies conditions such as Normal, Overheating, Overloading, and Mechanical Fault.
- **Digital Twin Visualization:** Provides a visual representation of motor condition monitoring.
- **Model Evaluation:** Includes classification reports, confusion matrix, and feature importance analysis.

## 🛠️ Technologies Used

- Python
- Machine Learning
- Random Forest
- Streamlit
- Pandas
- Plotly
- Joblib

## 🏗️ System Workflow

1. Load the motor dataset.
2. Process the motor operating parameters.
3. Use the trained Random Forest model to predict motor conditions.
4. Display predictions and visualizations in the Streamlit dashboard.
5. Analyze the results to support predictive maintenance decisions.

## ⚙️ Motor Parameters

| Parameter | Description |
|---|---|
| Voltage | Motor supply voltage |
| Current | Electrical current drawn by the motor |
| Temperature | Motor operating temperature |
| Vibration | Motor vibration level |
| RPM | Motor rotational speed |
| Load | Motor operating load |

## 🚨 Supported Motor Conditions

- **Normal:** Operating parameters indicate normal conditions.
- **Overheating:** Indicates a potential excessive-temperature condition.
- **Overloading:** Indicates a potential excessive-load condition.
- **Mechanical Fault:** Indicates a potential mechanical abnormality.

Predictions depend on the dataset and the trained model; they should not be treated as definitive safety diagnoses.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kapiladithyaa/AI-Motor-Digital-Twin.git
cd AI-Motor-Digital-Twin
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

After creating the project's `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

### 4. Launch the modern dashboard

```bash
python -m streamlit run app_modern_new.py
```

The application will open in your browser.

## 📁 Important Project Files

| File | Purpose |
|---|---|
| `app_modern_new.py` | Modern Streamlit dashboard |
| `app.py` | Original dashboard |
| `motor_data.csv` | Motor dataset |
| `motor_condition_model.pkl` | Trained model package |
| `random_forest_model.pkl` | Additional trained model file |
| `digital_twin_hero.png` | Digital Twin dashboard image |
| `classification_report.csv` | Classification evaluation results |
| `confusion_matrix.csv` | Confusion matrix data |
| `confusion_matrix.png` | Confusion matrix visualization |
| `feature_importance.csv` | Feature importance results |

## 🔬 Project Scope and Limitations

This is a software-based prototype using a motor dataset and a trained machine learning model. It demonstrates motor condition prediction and Digital Twin concepts through a dashboard.

The current prototype does not represent a hardware-connected, real-time industrial motor installation. Prediction quality depends on the quality and representativeness of the training data.

## 🔮 Future Enhancements

- Integrate real motor sensors and IoT devices.
- Enable live data acquisition from industrial motors.
- Add cloud-based monitoring and alerts.
- Improve fault prediction using additional datasets.
- Validate the system against real industrial motor data.

## 👨‍💻 Project Information

**Project Title:** AI Predictive Maintenance System for Industrial Motors Using Digital Twin

**Domain:** Artificial Intelligence, Machine Learning, Digital Twin, and Industrial Automation

**Project Type:** College Mini Project

## 📄 License

A license has not yet been specified. Add an appropriate license if you intend to distribute or reuse this project publicly.
