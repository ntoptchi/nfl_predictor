from __future__ import annotations

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .utils import compute_week_picks, split_top_and_flips


app = FastAPI(title="NFL Predictor API", version="1.0.0")

# Dev-friendly CORS (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    season: int = Field(..., ge=1990, description="NFL season year (e.g., 2024)")
    week: int = Field(..., ge=1, le=25, description="NFL week number")
    top_k: int = Field(5, ge=1, le=50)
    flip_band: float = Field(0.03, ge=0.0, le=0.25)


class UIPickRow(BaseModel):
    matchup: str
    pick: str
    confidence: float


class PredictResponse(BaseModel):
    top: List[UIPickRow]
    all: List[UIPickRow]
    flips: List[UIPickRow]


@app.get("/")
def root():
    return {"ok": True, "message": "NFL Predictor API. Use POST /predict."}


@app.get("/predict")
def predict_help():
    return {
        "ok": True,
        "message": "POST /predict with JSON: {season, week, top_k?, flip_band?}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = compute_week_picks(req.season, req.week)

    top, flips = split_top_and_flips(df, top_k=req.top_k, flip_band=req.flip_band)

    # 'all' is your full slate of picks
    all_rows = [
        {
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "pick": str(r["predicted_winner"]),
            "confidence": float(r["confidence"]),
        }
        for _, r in df.iterrows()
    ]

    return {"top": top, "all": all_rows, "flips": flips}
