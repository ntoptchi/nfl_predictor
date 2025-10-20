import pandas as pd
import numpy as np
import nfl_data_py as nfl
from .config import Config

#  SCHEDULE
def load_games(seasons):
    sched = nfl.import_schedules(seasons).copy()
    # keep regular season only
    sched = sched[sched.get("game_type", "REG") == "REG"].copy()
    # normalize fields
    must = ["game_id","season","week","home_team","away_team","home_score","away_score","gameday"]
    for m in must:
        if m not in sched.columns:
            raise ValueError(f"Schedules missing column: {m}. Columns: {sorted(sched.columns)}")
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    return sched[["game_id","season","week","home_team","away_team","home_score","away_score","gameday","home_win"]]

#  WEEKLY -> TEAM TOTALS 
def team_game_stats(seasons):
    """
    Aggregate player weekly rows into per-team, per-game totals.
    Skips seasons whose weekly parquet is not yet published (e.g., future/current).
    Returns one row per (season, week, team) with passing_yards, rushing_yards, turnovers.
    """
    import pandas as pd
    import nfl_data_py as nfl

    frames = []
    for yr in sorted(set(seasons)):
        try:
            wk = nfl.import_weekly_data([yr]).copy()  # fetch one season at a time to catch 404s
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

        # Required identifiers must exist
        if any(x is None for x in [team_col, week_col, season_col]):
            print(f"[team_game_stats] Missing identifiers for {yr}; columns={sorted(wk.columns)} — skipping")
            continue

        use = [season_col, week_col, team_col]
        for c in [pass_col, rush_col, int_col, fum_col]:
            if c:
                use.append(c)
        df = wk[use].copy()

        # numeric
        for c in [pass_col, rush_col, int_col, fum_col]:
            if c and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df = df.rename(columns={season_col: "season", week_col: "week", team_col: "team"})
        df["passing_yards"] = df[pass_col] if pass_col in df.columns else 0
        df["rushing_yards"] = df[rush_col] if rush_col in df.columns else 0
        df["turnovers"]     = (df[int_col] if int_col in df.columns else 0) + (df[fum_col] if fum_col in df.columns else 0)

        agg = (
            df.groupby(["season", "week", "team"], as_index=False)[["passing_yards", "rushing_yards", "turnovers"]]
              .sum()
        )
        frames.append(agg)

    if not frames:
        raise RuntimeError("No weekly data could be loaded for the requested seasons — check internet or seasons list.")

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

    df["team_elo_momentum"] = (df["team_pf_roll"] - df["team_pa_roll"]) / (df["team_pf_roll"].abs() + 1)

    df["prev_game_date"] = grp["game_date"].shift(1)
    df["team_rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(10)
    df.drop(columns=["prev_game_date"], inplace=True)
    return df

# MATCHUPS (STACKED) 
def make_matchups(team_side, sched):
    """
    Build stacked (team-vs-opp) rows with UNIQUE column names.
    Each game contributes two rows:
      - home_view: team = home side
      - away_view: team = away side
    Columns are prefixed for opponent as 'opp_*' to avoid duplicates.
    """
    df = team_side.copy()
    df["home"] = (df["home_away"] == "HOME").astype(int)

    home = df[df["home"] == 1].copy()
    away = df[df["home"] == 0].copy()

    # one row per side per game
    home = home.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id"], keep="first")
    away = away.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id"], keep="first")

    merged = home.merge(
        away,
        on=["game_id","season","week"],
        suffixes=("", "_opp"),
        validate="one_to_one",
    )

    # attach result labels
    sched_small = sched[["game_id","home_win","home_team","away_team"]]
    merged = merged.merge(sched_small, on="game_id", how="left")

    # helper to build a clean row with unique names
    def build_row(src, team_from="", opp_from="_opp", home_flag=1, team_win_expr=None):
        out = pd.DataFrame({
            "game_id": src["game_id"],
            "season":  src["season"],
            "week":    src["week"],
            "home":    home_flag,
            # team stats
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
            # opponent stats (unique names)
            "opp_points_for":         src[f"points_for{opp_from}"],
            "opp_points_against":     src[f"points_against{opp_from}"],
            "opp_passing_yards":      src[f"passing_yards{opp_from}"],
            "opp_rushing_yards":      src[f"rushing_yards{opp_from}"],
            "opp_turnovers":          src[f"turnovers{opp_from}"],
            "opp_pf_roll":            src[f"team_pf_roll{opp_from}"],
            "opp_pa_roll":            src[f"team_pa_roll{opp_from}"],
            "opp_pass_y_roll":        src[f"team_pass_y_roll{opp_from}"],
            "opp_rush_y_roll":        src[f"team_rush_y_roll{opp_from}"],
            "opp_to_roll":            src[f"team_to_roll{opp_from}"],
            "opp_elo_momentum":       src[f"team_elo_momentum{opp_from}"],
            "opp_rest_days":          src[f"team_rest_days{opp_from}"],
        })
        out["team_win"] = team_win_expr(src).astype(int)
        return out

    # home perspective: take team stats from '', opp from '_opp'
    home_view = build_row(
        merged,
        team_from="",
        opp_from="_opp",
        home_flag=1,
        team_win_expr=lambda s: s["home_win"],
    )

    # away perspective: take team stats from '_opp', opp from ''
    away_view = build_row(
        merged,
        team_from="_opp",
        opp_from="",
        home_flag=0,
        team_win_expr=lambda s: 1 - s["home_win"],
    )

    stacked = pd.concat([home_view, away_view], ignore_index=True)

    # bring team names for reference
    keep_id = ["game_id","home_team","away_team"]
    stacked = stacked.merge(sched[keep_id].drop_duplicates(), on="game_id", how="left")

    return stacked

#  BUILD DATASET 
def build_dataset(seasons=None, rolling_n=None):
    seasons = seasons or Config.SEASONS
    rolling_n = rolling_n or Config.ROLLING_N

    sched = load_games(seasons)
    tstats = team_game_stats(seasons)

    # Create schedule-derived per-team rows (HOME & AWAY) with game context + PF/PA
    side_cols = ["game_id","season","week","gameday","home_team","away_team","home_score","away_score"]
    st = sched[side_cols].copy()
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
    sched_team_rows["game_date"] = pd.to_datetime(sched_team_rows["game_date"], errors="coerce")

    # Join team numeric stats by (season, week, team)
    team_side = sched_team_rows.merge(
        tstats,
        on=["season","week","team"],
        how="left",
        validate="many_to_one"
    )

    # Fill numeric NaNs with 0 (early seasons or bye oddities)
    for c in ["passing_yards","rushing_yards","turnovers"]:
        if c in team_side.columns:
            team_side[c] = pd.to_numeric(team_side[c], errors="coerce").fillna(0)

    # Guarantee one row per (game_id, team)
    keys = ["game_id","team"]
    num_cols = [c for c in ["passing_yards","rushing_yards","turnovers","points_for","points_against"] if c in team_side.columns]
    if num_cols:
        team_side = (
            team_side
            .sort_values(["season","week","team"])
            .groupby(keys + ["season","week","opponent","home_away","game_date"], as_index=False)[num_cols]
            .sum()
        )

    # Add rolling features
    team_side = add_rolling_features(team_side, rolling_n)

    # Build stacked dataset (two rows per game)
    dataset = make_matchups(team_side, sched)

    # Clean NaNs in rolling cols (early weeks)
    roll_cols = [c for c in dataset.columns if c.endswith("_roll")]
    if roll_cols:
        dataset[roll_cols] = dataset[roll_cols].fillna(dataset[roll_cols].mean())

    # Target sanity
    dataset = dataset.dropna(subset=["team_win"])
    return dataset
