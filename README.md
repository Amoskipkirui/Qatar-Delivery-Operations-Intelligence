# Qatar Delivery Operations Intelligence

A delivery analytics and machine learning project built to explore delivery performance, identify operational delays, and estimate late-delivery risk.

## Overview

Timely delivery is an important part of customer experience in the food and e-commerce delivery industry. Delays can arise from traffic conditions, preparation time, delivery distance, weather, and courier experience.

This project brings these factors together in a single application. It combines operational KPIs with a machine learning model that estimates the likelihood of a delivery arriving later than its promised delivery time.

The project was developed as a portfolio demonstration using a synthetic dataset designed around delivery operations in Qatar.

## Features

- **Operations dashboard:** Summarises total orders, completed deliveries, cancellations, late deliveries, on-time delivery rate, and average delivery time.
- **Late-delivery prediction:** Estimates delivery risk using order and operating conditions.
- **Risk assessment:** Classifies predictions into low, moderate, and high risk categories.
- **Operational guidance:** Provides suggested actions based on the predicted outcome.
- **Flask API:** Exposes prediction functionality through an HTTP endpoint.
- **Power BI reporting:** Supports further exploration of delivery performance and operational trends.

## Technology Stack

| Area | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Web application | Flask, HTML, CSS |
| Model persistence | Joblib |
| Business intelligence | Power BI |
| Deployment | Render |
| Version control | Git and GitHub |

## Dataset

The project uses a synthetic dataset containing 5,020 delivery records and 14 columns.

The data includes order dates and times, delivery zones, order types, distance, traffic, weather, preparation time, courier experience, promised delivery time, actual delivery time, order status, and the original late-delivery label.

Data preparation includes standardising categorical values, converting data types, removing exact duplicate rows, identifying missing values, and reviewing repeated order IDs.

For model training, the target is defined from the delivery times:

- **Late:** Actual delivery time exceeds promised delivery time.
- **On time:** Actual delivery time is less than or equal to promised delivery time.

Cancelled orders are excluded from model training because their delivery outcome cannot be evaluated in the same way.

## Machine Learning Approach

A Random Forest classifier is trained using delivery conditions and order timing features. Missing numerical and categorical values are handled through a preprocessing pipeline, and categorical variables are encoded before model training.

The model is evaluated using a held-out test set, with particular attention to precision, recall, F1-score, confusion matrix, and ROC-AUC.

The initial model achieved a ROC-AUC of **0.784** on the test set. Its recall for late deliveries was approximately **0.58**, while precision was approximately **0.31**. These results provide a baseline for further model improvement, especially in reducing false alarms while detecting delayed orders.

The model is intended for demonstration and experimentation, not production dispatch decisions.

## Application

The Flask application provides an operations overview and a form for assessing an individual delivery.

Users can enter the delivery zone, order type, distance, traffic and weather conditions, preparation time, courier experience, promised delivery time, and order timing details.

The application returns the predicted outcome, estimated late-delivery probability, risk category, and suggested operational guidance.

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Opens the operations dashboard |
| `/health` | GET | Checks whether the application is running |
| `/predict` | POST | Submits the delivery assessment form |
| `/api/predict` | POST | Accepts JSON input and returns a prediction |

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/qatar-delivery-operations-intelligence.git
cd qatar-delivery-operations-intelligence
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the data and model

If the cleaned dataset is not available, run:

```bash
python clean_delivery_data.py
```

If the saved model is not available, run:

```bash
python train_model.py
```

### 5. Start the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Deployment

The application can be deployed to Render using the connected GitHub repository.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
gunicorn app:app
```

The trained model and required data files must be included in the deployed application or made available through a configured storage location.

## Project Structure

```text
Qatar_Delivery_Operations_Intelligence/
├── app.py
├── clean_delivery_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── qatar_delivery_orders_raw.csv
│   └── qatar_delivery_orders_clean.csv
├── models/
│   └── late_delivery_model.joblib
├── notebooks/
│   └── delivery.ipynb
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Limitations and Future Improvements

The dataset is synthetic and does not represent actual operational data from a delivery company. Model performance on this dataset may not generalise to real delivery environments.

Potential improvements include:

- Testing on real, authorised delivery data.
- Improving late-delivery precision and recall.
- Analysing delay patterns by delivery zone, hour, traffic, and weather.
- Integrating Power BI reporting with a reliable data refresh process.
- Adding automated model monitoring and prediction logging.

## Project Links

- **GitHub repository:** [Add repository URL]
- **Live application:** [Add Render URL]
- **Power BI dashboard:** [Add dashboard link or screenshot]

---

**Note:** This is an independent portfolio project using synthetic data. It is not affiliated with, endorsed by, or connected to Snoonu, Talabat, or any other delivery company.