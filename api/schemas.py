from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    season: int = Field(..., ge=2000, le=2100)
    week: int = Field(..., ge=1, le=22)   # may include playoffs later
    top_k: int = 5
    flip_band: float = 0.02

class GamePick(BaseModel):
    game_id: str
    season: int
    week: int
    gameday: str
    home_team: str
    away_team: str
    predicted_winner: str
    confidence: float

class PredictResponse(BaseModel):
    picks: List[GamePick]
    top_confidence: List[GamePick]
    coin_flips: List[GamePick]

class BacktestRequest(BaseModel):
    seasons_start: int
    seasons_end: int
    rolling_n: int = 5

class BacktestRow(BaseModel):
    season: str
    games: int
    accuracy: float
    log_loss: float
    brier: float

class BacktestResponse(BaseModel):
    rows: List[BacktestRow]

class BettingRequest(BaseModel):
    season: int
    min_edge: float = 0.02
    kelly: float = 0.0
    flat: float = 1.0

class BetRow(BaseModel):
    season: int
    week: int
    gameday: str
    game_id: str
    home_team: str
    away_team: str
    chosen_team: str
    is_home_pick: int
    team_prob_win: float
    ml_pick: Optional[float] = None
    p_imp: Optional[float] = None
    edge: Optional[float] = None
    stake: float
    won: int
    profit: float

class BettingResponse(BaseModel):
    roi: float
    total_staked: float
    hit_rate: float
    max_drawdown: float
    bets: List[BetRow]
