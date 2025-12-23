
# 🏈 NFL Predictor

A full-stack NFL game prediction platform that generates **weekly picks with confidence scores** using a machine-learning ensemble model, served through a **FastAPI backend** and rendered in a **modern React frontend**.

This project focuses on real-world ML deployment: feature engineering, model inference, API design, and a sleek UI for end users.

---

## 🚀 Features

- 🔮 Weekly NFL game predictions
- 📊 Confidence percentages per pick
- ⚡ FastAPI backend for real-time inference
- 🧠 Ensemble ML model trained on historical NFL data
- 🖥️ Modern React + TypeScript frontend
- 🎨 Minimalist black UI with custom typography
- 🔄 Live season/week selection
- 📱 Responsive layout

---

## 🧠 Model Overview

The prediction engine is an **ensemble classifier** trained on multiple seasons of NFL data with extensive feature engineering.

### Feature categories include:
- Team performance (rolling averages)
- Offensive and defensive efficiency
- Passing yards, rushing yards, turnovers
- Elo ratings and momentum
- Rest days and scheduling effects
- Market and contextual signals (when available)

The model outputs **win probabilities per team**, which are converted into:
- Predicted winner
- Confidence percentage
- Ranked weekly picks

---

## 🏗️ Architecture

```

nfl_predictor/
├── api/
│   ├── main.py        # FastAPI routes
│   └── utils.py       # Prediction + post-processing logic
├── src/
│   ├── data.py        # Dataset construction and model loading
│   └── model files    # Trained ensemble artifacts
├── web/
│   ├── App.tsx        # React UI
│   └── styles         # Custom styling
└── README.md

```

### Request Flow

```

React UI
↓ POST /predict
FastAPI API
↓
ML Model Inference
↓
JSON Response
↓
Frontend Rendering

````

---

## 🔌 API Example

**POST** `/predict`

```json
{
  "season": 2025,
  "week": 1
}
````

**Response (simplified):**

```json
{
  "picks": [
    {
      "game_id": "2025_01_DAL_WAS",
      "home_team": "DAL",
      "away_team": "WAS",
      "predicted_winner": "DAL",
      "confidence": 0.62
    }
  ]
}
```

---

## 🖥️ Frontend

* Built with **React + TypeScript**
* Fully client-side rendering of predictions
* Sleek, sportsbook-inspired minimalist design
* Team logos and visual confidence indicators
* Live API-driven data (no CSV exports)

---

## 🛠️ Tech Stack

**Backend**

* Python
* FastAPI
* Pandas
* NumPy
* scikit-learn
* Joblib

**Frontend**

* React
* TypeScript
* CSS
* Google Fonts (Source Code Pro, Saira Condensed)

---

## ▶️ Running Locally

### Backend

```bash
cd nfl_predictor
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd web
npm install
npm run dev
```

---

## 📌 Design Notes

* Designed for **live inference**, not batch CSV generation
* Offline scripts were intentionally removed to reflect real deployment workflows
* Modular structure allows easy extension:

  * Backtesting
  * Odds comparison
  * Betting simulations
  * Model calibration


## 👤 Author

**Nicholas Toptchi**


