# 📈 Price Prediction API

A FastAPI-based machine learning API for predicting house prices (Kochi sample).  
Dockerized and CI/CD friendly — ready to run locally with Docker Compose or deploy to AWS / ECS / ECR.

---

## 🔍 Project Summary

This project exposes a FastAPI backend that:
- Loads a trained scikit-learn pipeline (`model.pkl`) to make price predictions.
- Accepts **real-time** prediction requests via WebSocket (`/ws`).
- Accepts CSV uploads for retraining the model (`/upload-data`) and retraining endpoint (`/retrain`).
- Logs predictions to `prediction_log.csv`.
- Includes a basic web-scraping helper to fetch a trend price (best-effort, may return `None` if site blocks scraping).

---

## 📂 Repository structure

Price_pred/
├── main.py # FastAPI application (server)
├── Dockerfile # Dockerfile for the service
├── docker-compose.yml # Docker Compose for local dev
├── requirements.txt # Python dependencies
├── model.pkl # Trained ML model (generated if missing)
├── prediction_log.csv # Prediction logs (created by app)
├── uploads/ # Uploaded CSVs (persisted by volume)
└── tests/
└── test_model_performance.py


---

## ⚙️ Requirements

Example `requirements.txt`:

