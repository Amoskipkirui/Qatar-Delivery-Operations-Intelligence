
from pathlib import Path
import pandas as pd

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_FILE = DATA_DIR / "qatar_delivery_orders_raw.csv"
CLEAN_FILE = DATA_DIR / "qatar_delivery_orders_clean.csv"
DUPLICATES_FILE = DATA_DIR / "duplicate_order_ids.csv"

# 2. Load raw data
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print("Original dataset shape:", df.shape)

# 3. Standardize column names and text
df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 4. Remove completely identical rows
exact_duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Exact duplicate rows removed:", exact_duplicates)

# 5. Standardize categorical values
categorical_columns = [
    "delivery_zone",
    "order_type",
    "traffic_level",
    "weather_condition",
    "order_status",
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].str.title()

# Standardize the target labels
if "late_delivery" in df.columns:
    df["late_delivery"] = (
        df["late_delivery"]
        .str.strip()
        .str.lower()
        .map({
            "yes": "Yes",
            "no": "No",
            "true": "Yes",
            "false": "No",
            "1": "Yes",
            "0": "No",
        })
    )

# 6. Convert columns to appropriate data types
if "order_date" in df.columns:
    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce"
    )

if "order_time" in df.columns:
    df["order_time"] = df["order_time"].astype("string").str.strip()

numeric_columns = [
    "distance_km",
    "prep_time_min",
    "courier_experience_months",
    "promised_delivery_min",
    "actual_delivery_min",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 7. Investigate repeated order IDs
if "order_id" in df.columns:
    repeated_ids = df[
        df["order_id"].duplicated(keep=False)
    ].sort_values("order_id")

    repeated_ids.to_csv(DUPLICATES_FILE, index=False)
    print("Rows with repeated order IDs:", len(repeated_ids))
    print("Duplicate ID report:", DUPLICATES_FILE)

# 8. Report missing values without blindly filling them
print("\nMissing values before saving:")
print(df.isna().sum().sort_values(ascending=False))

# 9. Save cleaned data
df.to_csv(CLEAN_FILE, index=False, date_format="%Y-%m-%d")

print("\nCleaning completed successfully!")
print("Cleaned dataset shape:", df.shape)
print("Saved to:", CLEAN_FILE)
print("\nOrder status distribution:")
print(df["order_status"].value_counts(dropna=False))

if "late_delivery" in df.columns:
    print("\nLate delivery distribution:")
    print(df["late_delivery"].value_counts(dropna=False))
