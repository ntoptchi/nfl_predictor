import os
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from .config import Config

# helpers 
def _csv_path(name):
    return os.path.join(Config.DATA_DIR, name)

def _load_optional_csv(path, dtype=None, parse_dates=None):
    try:
        return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()

def _american_to_implied_prob(odds):
    # American odds → raw implied (no vig removal)
    s = pd.to_numeric(odds, errors="coerce")
    pos = s.where(s > 0)
    neg = s.where(s < 0)
    p = np.where(
        pd.notnull(pos),
        100.0 / (pos + 100.0),
        np.where(pd.notnull(neg), (-neg) / ((-neg) + 100.0), np.nan),
    )
    return pd.Series(p)

#  SCHEDULE 
def load_games(seasons):
    sched = nfl.import_schedules(seasons).copy()
    sched = sched[sched.get("game_type", "REG") == "REG"].copy()
    must = ["game_id","season","week","home_team","away_team","home_score","away_score","gameday"]
    for m in must:
        if m not in sched.columns:
            raise ValueError(f"Schedules missing column: {m}. Columns: {sorted(sched.columns)}")
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    return sched[[
        "game_id","season","week",
        "home_team","away_team",
        "home_score","away_score",
        "gameday","home_win"
    ]]

#  WEEKLY -> TEAM TOTALS 
def team_game_stats(seasons):
    frames = []
    for yr in sorted(set(seasons)):
        try:
            wk = nfl.import_weekly_data([yr]).copy()
        except Exception as e:
            print(f"[team_game_stats] Skipping season {yr}: {e}")
            continue

        cols = set(wk.columns)

        def pick(*cands):
            for c in cands:
                if c in cols:
                    return c
            return None

        team_col   = pick("recent_team", "team")
        week_col   = pick("week")
        season_col = pick("season")
        pass_col   = pick("pass_yds", "passing_yards")
        rush_col   = pick("rush_yds", "rushing_yards")
        int_col    = pick("interceptions", "int")
        fum_col    = pick("fumbles_lost", "fum_lost")

        if any(x is None for x in [team_col, week_col, season_col]):
            print(f"[team_game_stats] Missing identifiers for {yr}; columns={sorted(wk.columns)} — skipping")
            continue

        use = [season_col, week_col, team_col]
        for c in [pass_col, rush_col, int_col, fum_col]:
            if c:
                use.append(c)
        df = wk[use].copy()

        for c in [pass_col, rush_col, int_col, fum_col]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df = df.rename(columns={season_col:"season", week_col:"week", team_col:"team"})
        df["passing_yards"] = df[pass_col] if pass_col in df.columns else 0
        df["rushing_yards"] = df[rush_col] if rush_col in df.columns else 0
        df["turnovers"]     = (
            (df[int_col] if int_col in df.columns else 0) +
            (df[fum_col] if fum_col in df.columns else 0)
        )

        agg = df.groupby(
            ["season","week","team"], as_index=False
        )[["passing_yards","rushing_yards","turnovers"]].sum()
        frames.append(agg)

    if not frames:
        raise RuntimeError("No weekly data could be loaded for the requested seasons.")
    return pd.concat(frames, ignore_index=True)

#  ROLLING FEATURES
def add_rolling_features(df, n):
    df = df.sort_values(["team","season","week"]).copy()
    grp = df.groupby("team", group_keys=False)

    def roll(col, out):
        df[out] = grp[col].apply(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    roll("points_for",     "team_pf_roll")
    roll("points_against", "team_pa_roll")
    roll("passing_yards",  "team_pass_y_roll")
    roll("rushing_yards",  "team_rush_y_roll")
    roll("turnovers",      "team_to_roll")

    df["team_elo_momentum"] = (
        df["team_pf_roll"] - df["team_pa_roll"]
    ) / (df["team_pf_roll"].abs() + 1)

    df["prev_game_date"] = grp["game_date"].shift(1)
    df["team_rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(10)
    df.drop(columns=["prev_game_date"], inplace=True)
    return df

#  MATCHUPS (STACKED) 
def make_matchups(team_side, sched):
    df = team_side.copy()
    df["home"] = (df["home_away"] == "HOME").astype(int)

    home = df[df["home"] == 1].copy()
    away = df[df["home"] == 0].copy()

    home = home.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id"], keep="first")
    away = away.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id"], keep="first")

    merged = home.merge(
        away,
        on=["game_id","season","week"],
        suffixes=("", "_opp"),
        validate="one_to_one",
    )

    sched_small = sched[["game_id","home_win","home_team","away_team"]]
    merged = merged.merge(sched_small, on="game_id", how="left")

    def build_row(src, team_from="", opp_from="_opp", home_flag=1, team_win_expr=None):
        out = pd.DataFrame({
            "game_id": src["game_id"],
            "season":  src["season"],
            "week":    src["week"],
            "home":    home_flag,

            # team-side stats
            "points_for":         src[f"points_for{team_from}"],
            "points_against":     src[f"points_against{team_from}"],
            "passing_yards":      src[f"passing_yards{team_from}"],
            "rushing_yards":      src[f"rushing_yards{team_from}"],
            "turnovers":          src[f"turnovers{team_from}"],
            "team_pf_roll":       src[f"team_pf_roll{team_from}"],
            "team_pa_roll":       src[f"team_pa_roll{team_from}"],
            "team_pass_y_roll":   src[f"team_pass_y_roll{team_from}"],
            "team_rush_y_roll":   src[f"team_rush_y_roll{team_from}"],
            "team_to_roll":       src[f"team_to_roll{team_from}"],
            "team_elo_momentum":  src[f"team_elo_momentum{team_from}"],
            "team_rest_days":     src[f"team_rest_days{team_from}"],
            "team_elo":           src[f"team_elo{team_from}"],
            "elo_diff":           src[f"elo_diff{team_from}"],

            # market features (team perspective)
            "team_spread":        src.get(f"team_spread{team_from}", np.nan),
            "team_ml_implied":    src.get(f"team_ml_implied{team_from}", np.nan),
            "total_points":       src.get(f"total_points{team_from}", np.nan),

            # weather (game-level; same either side)
            "wind_mph":           src.get("wind_mph", np.nan),
            "temp_f":             src.get("temp_f", np.nan),
            "is_precip":          src.get("is_precip", np.nan),
            "is_outdoor":         src.get("is_outdoor", np.nan),

            # QB / injuries (team perspective)
            "qb_changed":         src.get(f"qb_changed{team_from}", np.nan),
            "starters_out":       src.get(f"starters_out{team_from}", np.nan),

            # EPA / SR (team perspective; rolling values expected)
            "off_epa_roll":       src.get(f"off_epa_roll{team_from}", np.nan),
            "def_epa_roll":       src.get(f"def_epa_roll{team_from}", np.nan),
            "sr_off_roll":        src.get(f"sr_off_roll{team_from}", np.nan),
            "sr_def_roll":        src.get(f"sr_def_roll{team_from}", np.nan),

            # opponent-side features (renamed with opp_*)
            "opp_points_for":     src[f"points_for{opp_from}"],
            "opp_points_against": src[f"points_against{opp_from}"],
            "opp_passing_yards":  src[f"passing_yards{opp_from}"],
            "opp_rushing_yards":  src[f"rushing_yards{opp_from}"],
            "opp_turnovers":      src[f"turnovers{opp_from}"],
            "opp_pf_roll":        src[f"team_pf_roll{opp_from}"],
            "opp_pa_roll":        src[f"team_pa_roll{opp_from}"],
            "opp_pass_y_roll":    src[f"team_pass_y_roll{opp_from}"],
            "opp_rush_y_roll":    src[f"team_rush_y_roll{opp_from}"],
            "opp_to_roll":        src[f"team_to_roll{opp_from}"],
            "opp_elo_momentum":   src[f"team_elo_momentum{opp_from}"],
            "opp_rest_days":      src[f"team_rest_days{opp_from}"],
            "opp_elo":            src[f"team_elo{opp_from}"],

            # market features (opponent perspective)
            "opp_spread":         src.get(f"team_spread{opp_from}", np.nan),
            "opp_ml_implied":     src.get(f"team_ml_implied{opp_from}", np.nan),

            # EPA/SR opponent
            "opp_off_epa_roll":   src.get(f"off_epa_roll{opp_from}", np.nan),
            "opp_def_epa_roll":   src.get(f"def_epa_roll{opp_from}", np.nan),
            "opp_sr_off_roll":    src.get(f"sr_off_roll{opp_from}", np.nan),
            "opp_sr_def_roll":    src.get(f"sr_def_roll{opp_from}", np.nan),
        })
        out["team_win"] = team_win_expr(src).astype(int)
        return out

    home_view = build_row(
        merged, team_from="",     opp_from="_opp", home_flag=1,
        team_win_expr=lambda s: s["home_win"]
    )
    away_view = build_row(
        merged, team_from="_opp", opp_from="",     home_flag=0,
        team_win_expr=lambda s: 1 - s["home_win"]
    )

    stacked = pd.concat([home_view, away_view], ignore_index=True)
    keep_id = ["game_id","home_team","away_team"]
    stacked = stacked.merge(sched[keep_id].drop_duplicates(), on="game_id", how="left")
    return stacked

#  BUILD DATASET
def build_dataset(seasons=None, rolling_n=None):
    seasons = seasons or Config.SEASONS
    rolling_n = rolling_n or Config.ROLLING_N

    # schedule + base stats
    sched = load_games(seasons)
    tstats = team_game_stats(seasons)

    # schedule-derived per-team rows
    st = sched[[
        "game_id","season","week","gameday",
        "home_team","away_team","home_score","away_score"
    ]].copy()

    st_home = st.rename(columns={"home_team":"team","away_team":"opponent"})
    st_home["home_away"] = "HOME"
    st_home["points_for"] = st_home["home_score"]
    st_home["points_against"] = st_home["away_score"]

    st_away = st.rename(columns={"away_team":"team","home_team":"opponent"})
    st_away["home_away"] = "AWAY"
    st_away["points_for"] = st_away["away_score"]
    st_away["points_against"] = st_away["home_score"]

    sched_team_rows = pd.concat([st_home, st_away], ignore_index=True)
    sched_team_rows = sched_team_rows.rename(columns={"gameday":"game_date"})
    sched_team_rows["game_date"] = pd.to_datetime(
        sched_team_rows["game_date"], errors="coerce"
    )

    # join weekly aggregates
    team_side = sched_team_rows.merge(
        tstats, on=["season","week","team"],
        how="left", validate="many_to_one"
    )

    # numeric fill
    for c in ["passing_yards","rushing_yards","turnovers"]:
        if c in team_side.columns:
            team_side[c] = pd.to_numeric(team_side[c], errors="coerce").fillna(0)

    # market odds ( by game_id) 
    if getattr(Config, "USE_MARKET", False):
        market = _load_optional_csv(_csv_path(Config.FILE_MARKET))
        if not market.empty and "game_id" in market.columns:
            # expected columns: game_id, close_spread_home (fav negative), ml_home, ml_away, total_points
            m = market.copy()
            team_side = team_side.merge(
                m[["game_id","close_spread_home","ml_home","ml_away","total_points"]],
                on="game_id", how="left"
            )
            team_side["team_spread"] = np.where(
                team_side["home_away"]=="HOME",
                team_side["close_spread_home"],
                -team_side["close_spread_home"],
            )
            team_side["opp_spread_tmp"] = -team_side["team_spread"]
            # moneyline implied (raw)
            team_side["team_ml_implied"] = np.where(
                team_side["home_away"]=="HOME",
                _american_to_implied_prob(team_side["ml_home"]),
                _american_to_implied_prob(team_side["ml_away"]),
            )
            team_side["opp_ml_implied_tmp"] = np.where(
                team_side["home_away"]=="HOME",
                _american_to_implied_prob(team_side["ml_away"]),
                _american_to_implied_prob(team_side["ml_home"]),
            )
            team_side["total_points"] = pd.to_numeric(
                team_side["total_points"], errors="coerce"
            )
        else:
            print("[build_dataset] market file missing or empty; skipping.")

    #  weather (by game_id) 
    if getattr(Config, "USE_WEATHER", False):
        weather = _load_optional_csv(_csv_path(Config.FILE_WEATHER))
        if not weather.empty and "game_id" in weather.columns:
            w = weather.copy()
            # expected columns: game_id, wind_mph, temp_f, is_precip, is_outdoor
            team_side = team_side.merge(
                w[["game_id","wind_mph","temp_f","is_precip","is_outdoor"]],
                on="game_id", how="left"
            )
        else:
            print("[build_dataset] weather file missing or empty; skipping.")

    #  QB / injuries (by season/week/team)
    if getattr(Config, "USE_QB", False):
        qb = _load_optional_csv(_csv_path(Config.FILE_QB))
        if not qb.empty and set(["season","week","team"]).issubset(qb.columns):
            # expected: season, week, team, qb_changed (0/1), starters_out (int)
            team_side = team_side.merge(
                qb[["season","week","team","qb_changed","starters_out"]],
                on=["season","week","team"], how="left"
            )
        else:
            print("[build_dataset] qb_status file missing or empty; skipping.")

    # EPA / Success Rate (by season/week/team) 
    if getattr(Config, "USE_EPA", False):
        epa = _load_optional_csv(_csv_path(Config.FILE_EPA))
        if not epa.empty and set(["season","week","team"]).issubset(epa.columns):
            # expected: season, week, team, off_epa_roll, def_epa_roll, sr_off_roll, sr_def_roll
            team_side = team_side.merge(
                epa[["season","week","team","off_epa_roll","def_epa_roll","sr_off_roll","sr_def_roll"]],
                on=["season","week","team"], how="left"
            )
        else:
            print("[build_dataset] team_epa file missing or empty; skipping.")

    # Guarantee one row per (game_id, team)
    keys = ["game_id","team"]
    num_cols = [
        c for c in ["passing_yards","rushing_yards","turnovers","points_for","points_against"]
        if c in team_side.columns
    ]
    if num_cols:
        team_side = (
            team_side.sort_values(["season","week","team"])
                     .groupby(keys + ["season","week","opponent","home_away","game_date"], as_index=False)[num_cols]
                     .sum()
        )

    #  Simple Elo
    def _simple_elo(df):
        df = df.sort_values(["team","season","week"]).copy()
        K = 20
        base = 1500.0
        elo = {}
        ratings = []
        for _, r in df.iterrows():
            t, opp = r["team"], r["opponent"]
            Ra = elo.get(t, base)
            Rb = elo.get(opp, base)
            Ea = 1.0 / (1 + 10 ** ((Rb - Ra)/400))
            win = 1.0 if r["points_for"] > r["points_against"] else 0.0
            Ra_new = Ra + K * (win - Ea)
            elo[t] = Ra_new
            ratings.append(Ra_new)
        df["team_elo"] = ratings
        return df

    team_side = _simple_elo(team_side)
    team_side = team_side.merge(
        team_side[["game_id","team","team_elo"]].rename(
            columns={"team":"opponent","team_elo":"opp_elo"}
        ),
        on=["game_id","opponent"], how="left"
    )
    team_side["elo_diff"] = team_side["team_elo"] - team_side["opp_elo"]

    # Rolling features
    team_side = add_rolling_features(team_side, rolling_n)

    # Build stacked dataset
    dataset = make_matchups(team_side, sched)

    # Fill early-week NaNs in rolling/new cols (safe per-column fill)
    roll_cols = [c for c in dataset.columns if c.endswith("_roll")]
    fill_cols = roll_cols + [
        "team_spread","opp_spread","team_ml_implied","opp_ml_implied","total_points",
        "wind_mph","temp_f","is_precip","is_outdoor","qb_changed","starters_out",
        "off_epa_roll","def_epa_roll","sr_off_roll","sr_def_roll",
        "opp_off_epa_roll","opp_def_epa_roll","opp_sr_off_roll","opp_sr_def_roll",
    ]
    present = [c for c in fill_cols if c in dataset.columns]

    for col in present:
        if pd.api.types.is_numeric_dtype(dataset[col]):
            med = dataset[col].median()
            if pd.isna(med):
                continue
            dataset[col] = dataset[col].fillna(med)

    dataset = dataset.dropna(subset=["team_win"])
    return dataset
