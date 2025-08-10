import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import math

# Paths
MODEL_PATH = "model.pkl"
LOG_PATH = "model_performance_log.csv"
TEST_DATA_PATH = "uploads/kochi_data.csv"  # use your default or uploaded CSV

def evaluate_model():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load test dataset
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)

    required = {"location", "area_sqft", "bhk", "age", "furnishing", "parking", "price"}
    if not required.issubset(df.columns):
        raise ValueError("Test CSV must contain required columns.")

    X = df[["location", "area_sqft", "bhk", "age", "furnishing", "parking"]]
    y_true = df["price"]

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Log entry
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4)
    }])

    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

    print(f"✅ Model Evaluation Complete: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

if __name__ == "__main__":
    evaluate_model()
