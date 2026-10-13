
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "qatar_delivery_orders_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "late_delivery_model.joblib"


# 2. Load data
df = pd.read_csv(DATA_FILE)

# 3. Keep delivered orders with valid timing information
df["actual_delivery_min"] = pd.to_numeric(
    df["actual_delivery_min"], errors="coerce"
)
df["promised_delivery_min"] = pd.to_numeric(
    df["promised_delivery_min"], errors="coerce"
)

df = df[df["order_status"].eq("Delivered")].copy()

df = df.dropna(
    subset=["actual_delivery_min", "promised_delivery_min"]
)

# Target: 1 means late; 0 means on time
df["target"] = (
    df["actual_delivery_min"] > df["promised_delivery_min"]
).astype(int)

# 4. Create useful features available before delivery
df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

parsed_time = pd.to_datetime(
    df["order_time"].astype("string"),
    format="mixed",
    errors="coerce",
)

df["order_hour"] = parsed_time.dt.hour

df["order_day_of_week"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month

# Do NOT use actual delivery time or the original late label
features = [
    "delivery_zone",
    "order_type",
    "distance_km",
    "traffic_level",
    "weather_condition",
    "prep_time_min",
    "courier_experience_months",
    "promised_delivery_min",
    "order_hour",
    "order_day_of_week",
    "order_month",
]

X = df[features].copy()
y = df["target"]

categorical_features = [
    "delivery_zone",
    "order_type",
    "traffic_level",
    "weather_condition",
]

numerical_features = [
    col for col in features if col not in categorical_features
]

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 6. Model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 8. Train and evaluate
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "On time", 1: "Late"}))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["On time", "Late"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))

# 9. Save model and metadata
joblib.dump({
    "pipeline": pipeline,
    "features": features,
    "target_definition": (
        "Late if actual_delivery_min > promised_delivery_min"
    ),
}, MODEL_FILE)

print("\nModel saved to:", MODEL_FILE)
