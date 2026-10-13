
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "qatar_delivery_orders_clean.csv"
MODEL_FILE = BASE_DIR / "models" / "late_delivery_model.joblib"

# Load the trained model
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py before starting the app."
    )

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run clean_delivery_data.py first."
    )

saved_model = joblib.load(MODEL_FILE)
pipeline = saved_model["pipeline"]
FEATURES = saved_model["features"]

# Load data for dashboard KPIs
df = pd.read_csv(DATA_FILE)

df["order_status"] = (
    df["order_status"].astype("string").str.strip().str.title()
)

for column in ["actual_delivery_min", "promised_delivery_min"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

delivered = df[df["order_status"] == "Delivered"].copy()
cancelled_count = int((df["order_status"] == "Cancelled").sum())

delivered["is_late"] = (
    delivered["actual_delivery_min"] > delivered["promised_delivery_min"]
)

total_orders = len(df)
delivered_count = len(delivered)
late_count = int(delivered["is_late"].sum())

on_time_rate = (
    100 * (delivered_count - late_count) / delivered_count
    if delivered_count else 0
)

average_time = delivered["actual_delivery_min"].mean()
average_time = float(average_time) if pd.notna(average_time) else 0

METRICS = {
    "total_orders": f"{total_orders:,}",
    "delivered_orders": f"{delivered_count:,}",
    "cancelled_orders": f"{cancelled_count:,}",
    "late_deliveries": f"{late_count:,}",
    "on_time_rate": f"{on_time_rate:.1f}%",
    "avg_delivery_time": f"{average_time:.1f} min",
}

# Build dropdown choices from the actual dataset
CATEGORY_FIELDS = [
    "delivery_zone",
    "order_type",
    "traffic_level",
    "weather_condition",
]

CHOICES = {
    field: sorted(
        df[field].dropna().astype(str).unique().tolist()
    )
    for field in CATEGORY_FIELDS
}

DEFAULTS = {
    "delivery_zone": CHOICES["delivery_zone"][0],
    "order_type": CHOICES["order_type"][0],
    "traffic_level": CHOICES["traffic_level"][0],
    "weather_condition": CHOICES["weather_condition"][0],
    "distance_km": "5",
    "prep_time_min": "15",
    "courier_experience_months": "12",
    "promised_delivery_min": "40",
    "order_hour": "19",
    "order_day_of_week": "2",
    "order_month": "10",
}


def make_prediction(values):
    input_data = pd.DataFrame(
        [{feature: values[feature] for feature in FEATURES}],
        columns=FEATURES,
    )

    probabilities = pipeline.predict_proba(input_data)[0]
    late_probability = float(probabilities[1])

    if late_probability >= 0.70:
        risk = "High"
    elif late_probability >= 0.40:
        risk = "Moderate"
    else:
        risk = "Low"

    prediction = "Late" if late_probability >= 0.50 else "On time"

    return {
        "prediction": prediction,
        "risk": risk,
        "late_probability": late_probability,
        "late_probability_pct": f"{late_probability * 100:.1f}%",
        "late_probability_number": round(late_probability * 100, 1),
    }


def render_dashboard(result=None, form_data=None, error=None, status=200):
    return render_template(
        "index.html",
        metrics=METRICS,
        choices=CHOICES,
        result=result,
        form_data=form_data or DEFAULTS,
        error=error,
    ), status


@app.get("/")
def home():
    response, status = render_dashboard()
    return response, status


@app.post("/predict")
def predict_form():
    form_data = request.form.to_dict()

    try:
        values = {
            "delivery_zone": form_data["delivery_zone"],
            "order_type": form_data["order_type"],
            "distance_km": float(form_data["distance_km"]),
            "traffic_level": form_data["traffic_level"],
            "weather_condition": form_data["weather_condition"],
            "prep_time_min": float(form_data["prep_time_min"]),
            "courier_experience_months": float(
                form_data["courier_experience_months"]
            ),
            "promised_delivery_min": float(
                form_data["promised_delivery_min"]
            ),
            "order_hour": int(form_data["order_hour"]),
            "order_day_of_week": int(form_data["order_day_of_week"]),
            "order_month": int(form_data["order_month"]),
        }

        if not 0 <= values["order_hour"] <= 23:
            raise ValueError("Order hour must be between 0 and 23.")

        if not 0 <= values["order_day_of_week"] <= 6:
            raise ValueError("Day of week must be between 0 and 6.")

        if not 1 <= values["order_month"] <= 12:
            raise ValueError("Month must be between 1 and 12.")

        for field in [
            "distance_km",
            "prep_time_min",
            "courier_experience_months",
        ]:
            if values[field] < 0:
                raise ValueError(f"{field} cannot be negative.")

        if values["promised_delivery_min"] <= 0:
            raise ValueError("Promised delivery time must be greater than zero.")

        result = make_prediction(values)

        if result["prediction"] == "Late":
            result["recommendation"] = (
                "Review traffic, preparation time, and courier availability. "
                "Consider operational intervention where possible."
            )
        else:
            result["recommendation"] = (
                "The model estimates a lower late-delivery likelihood. "
                "Continue monitoring the order as it progresses."
            )

        return render_dashboard(result=result, form_data=form_data)

    except (KeyError, TypeError, ValueError):
        return render_dashboard(
            form_data=form_data,
            error="Please check all fields and enter valid values.",
            status=400,
        )


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing,
        }), 400

    try:
        result = make_prediction(payload)
        return jsonify({
            "prediction": result["prediction"],
            "risk": result["risk"],
            "late_probability": result["late_probability"],
            "note": "Portfolio prototype trained on synthetic data.",
        })
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input values."}), 400


if __name__ == "__main__":
    app.run(debug=True)
