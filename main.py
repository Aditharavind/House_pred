import nest_asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import requests
import shutil

# 🔄 Enable FastAPI to run in notebook
nest_asyncio.apply()

# 📁 Paths
MODEL_PATH = "model.pkl"
LOG_PATH = "prediction_log.csv"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🚀 Initialize app
app = FastAPI()

# 🌐 Real-time web scraping
def get_avg_price_per_sqft(location: str) -> float | None:
    try:
        search_location = location.lower().replace(" ", "-")
        url = f"https://www.magicbricks.com/property-for-sale/residential-real-estate?keyword={search_location}%20kochi"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        price_elements = soup.select("div.m-srp-card__price")
        prices = []

        for el in price_elements:
            text = el.get_text()
            if "₹" in text:
                try:
                    raw = text.replace("₹", "").replace(",", "").strip().split()[0]
                    price = int(float(raw))
                    prices.append(price)
                except ValueError:
                    continue

        return round(sum(prices) / len(prices), 2) if prices else None

    except Exception as e:
        print(f"Scraping error: {e}")
        return None

# 🧠 Training from CSV
def train_model_from_csv(csv_path: str, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)

    required = {"location", "area_sqft", "bhk", "age", "furnishing", "parking", "price"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain required columns.")

    X = df[["location", "area_sqft", "bhk", "age", "furnishing", "parking"]]
    y = df["price"]

    cat_cols = ["location", "furnishing"]
    num_cols = ["area_sqft", "bhk", "age", "parking"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, model_path)
    print("✅ Model trained and saved")

# ✅ Create default training dataset (Kochi)
default_csv_data = """location,area_sqft,bhk,age,furnishing,parking,price
Kadavanthra,1200,2,5,Semi-Furnished,1,8500000
Panampilly Nagar,1500,3,2,Fully-Furnished,1,12500000
Edappally,1000,2,10,Unfurnished,0,6200000
Thrikkakara,1300,3,4,Semi-Furnished,1,9000000
Vyttila,1100,2,8,Fully-Furnished,0,8000000
Aluva,1600,4,3,Unfurnished,2,10000000
Kakkanad,1400,3,5,Semi-Furnished,1,9700000
Palarivattom,1200,2,6,Unfurnished,0,7500000
Thevara,1700,4,2,Fully-Furnished,2,13000000
Kaloor,1000,2,9,Semi-Furnished,1,7200000
"""

default_csv_path = os.path.join(UPLOAD_DIR, "kochi_data.csv")
with open(default_csv_path, "w") as f:
    f.write(default_csv_data)

# 🧠 Initial training
train_model_from_csv(default_csv_path)
model = joblib.load(MODEL_PATH)

# 📊 Logging
def log_prediction(data: dict, predicted_price: float):
    entry = data.copy()
    entry["predicted_price"] = predicted_price
    df = pd.DataFrame([entry])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", index=False, header=False)
    else:
        df.to_csv(LOG_PATH, index=False)

# 🛰️ WebSocket
@app.websocket("/ws")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
            required_keys = ["location", "area_sqft", "bhk", "age", "furnishing", "parking"]
            if not all(k in data for k in required_keys):
                await websocket.send_json({"error": "Missing required fields."})
                continue

            trend_price = get_avg_price_per_sqft(data["location"])
            if trend_price:
                data["area_sqft"] *= trend_price / 100  # Optional trend-based weighting

            input_df = pd.DataFrame([{
                "location": data["location"],
                "area_sqft": data["area_sqft"],
                "bhk": data["bhk"],
                "age": data["age"],
                "furnishing": data["furnishing"],
                "parking": data["parking"]
            }])

            prediction = model.predict(input_df)[0]
            log_prediction(data, prediction)
            await websocket.send_json({"predicted_price": round(prediction, 2)})

        except Exception as e:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
            break

# 📤 Upload CSV
@app.post("/upload-data")
async def upload_csv(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"File uploaded as {file.filename}"}

# 🔁 Retrain from uploaded CSV
@app.post("/retrain")
async def retrain_model(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found."}
    try:
        train_model_from_csv(file_path)
        global model
        model = joblib.load(MODEL_PATH)
        return {"message": "Model retrained successfully."}
    except Exception as e:
        return {"error": str(e)}

# 📥 Prediction via POST
class PredictionInput(BaseModel):
    location: str
    area_sqft: float
    bhk: int
    age: int
    furnishing: str
    parking: int

@app.post("/predict")
def predict_price(data: PredictionInput):
    trend_price = get_avg_price_per_sqft(data.location)
    input_data = data.dict()

    if trend_price:
        input_data["area_sqft"] *= trend_price / 100

    input_df = pd.DataFrame([{
        "location": input_data["location"],
        "area_sqft": input_data["area_sqft"],
        "bhk": input_data["bhk"],
        "age": input_data["age"],
        "furnishing": input_data["furnishing"],
        "parking": input_data["parking"]
    }])

    prediction = model.predict(input_df)[0]
    log_prediction(data.dict(), prediction)
    return {"predicted_price": round(prediction, 2)}

# Root endpoint
@app.get("/")
def root():
    return {"message": "🏠 Kochi House Price Prediction API is live"}

# 🚀 Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
