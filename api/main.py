import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (PredictRequest, PredictResponse, GamePick,
                      BacktestRequest, BacktestResponse, BacktestRow,
                      BettingRequest, BettingResponse, BetRow)
from .utils import compute_week_picks, split_top_and_flips, run_backtest, run_betting_sim

app = FastAPI(title="NFL Predictor API", version="1.0.0")

# CORS (adjust for your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = compute_week_picks(req.season, req.week)
    top, flips = split_top_and_flips(df, req.top_k, req.flip_band)

    def to_models(frame):
        return [
            GamePick(
                game_id=str(r.game_id),
                season=int(r.season),
                week=int(r.week),
                gameday=str(r.gameday),
                home_team=str(r.home_team),
                away_team=str(r.away_team),
                predicted_winner=str(r.predicted_winner),
                confidence=float(r.confidence),
            ) for _, r in frame.iterrows()
        ]

    return PredictResponse(
        picks=to_models(df),
        top_confidence=to_models(top),
        coin_flips=to_models(flips)
    )

@app.post("/eval/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    res = run_backtest(req.seasons_start, req.seasons_end, req.rolling_n)
    rows = [
        BacktestRow(
            season=str(r.season),
            games=int(r.games),
            accuracy=float(r.accuracy),
            log_loss=float(r.log_loss),
            brier=float(r.brier),
        ) for _, r in res.iterrows()
    ]
    return BacktestResponse(rows=rows)

@app.post("/eval/betting", response_model=BettingResponse)
def betting(req: BettingRequest):
    bets_df, roi, staked, hit, mdd = run_betting_sim(req.season, req.min_edge, req.kelly, req.flat)
    bets = []
    for _, r in bets_df.iterrows():
        bets.append(BetRow(
            season=int(r.get("season", 0)),
            week=int(r.get("week", 0)),
            gameday=str(r.get("gameday", "")),
            game_id=str(r.get("game_id", "")),
            home_team=str(r.get("home_team", "")),
            away_team=str(r.get("away_team", "")),
            chosen_team=str(r.get("chosen_team", "")),
            is_home_pick=int(r.get("is_home_pick", 0)),
            team_prob_win=float(r.get("team_prob_win", 0.0)),
            ml_pick=float(r.get("ml_pick")) if r.get("ml_pick") is not None else None,
            p_imp=float(r.get("p_imp")) if r.get("p_imp") is not None else None,
            edge=float(r.get("edge")) if r.get("edge") is not None else None,
            stake=float(r.get("stake", 0.0)),
            won=int(r.get("won", 0)),
            profit=float(r.get("profit", 0.0)),
        ))
    return BettingResponse(
        roi=float(roi),
        total_staked=float(staked),
        hit_rate=float(hit),
        max_drawdown=float(mdd),
        bets=bets
    )
