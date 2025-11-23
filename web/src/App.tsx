import { useState } from "react";

type Row = {
  matchup: string;
  pick: string;
  prob: number; // 0–1
};

type ApiPayload = Record<string, any>;

function coerceTables(payload: ApiPayload) {
  const top = (payload.top_picks ?? payload.top ?? payload.top_confidence ?? []) as any[];
  const all = (payload.all_picks ?? payload.all ?? payload.picks ?? []) as any[];
  const flips = (payload.coin_flips ?? payload.flips ?? []) as any[];

  const norm = (rows: any[]): Row[] =>
    Array.isArray(rows)
      ? rows.map((r) => ({
          matchup:
            r.matchup ??
            r.Matchup ??
            (r.away && r.home ? `${r.away} @ ${r.home}` : String(r.matchup ?? "")),
          pick: r.pick ?? r.Pick ?? r.gameday_pick ?? r.predicted_winner ?? "",
          prob: typeof r.confidence === "number"
            ? r.confidence
            : typeof r.probability === "number"
            ? r.probability
            : typeof r.Confidence === "number"
            ? r.Confidence
            : typeof r.prob === "number"
            ? r.prob
            : 0,
        }))
      : [];

  return {
    top: norm(top),
    all: norm(all),
    flips: norm(flips),
  };
}

export default function App() {
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [top, setTop] = useState<Row[]>([]);
  const [all, setAll] = useState<Row[]>([]);
  const [flips, setFlips] = useState<Row[]>([]);

  const getPicks = async () => {
    setLoading(true);
    setError(null);

    try {
      const url = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
      const res = await fetch(`${url}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season: Number(season), week: Number(week) }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API ${res.status} ${res.statusText}: ${txt.slice(0, 200)}`);
      }

      const data = (await res.json()) as ApiPayload;
      const tables = coerceTables(data);

      setTop(tables.top);
      setAll(tables.all);
      setFlips(tables.flips);
    } catch (e: any) {
      console.error(e);
      setError(
        "Failed to fetch picks. Make sure the API server is running on port 8000 and try again."
      );
      setTop([]);
      setAll([]);
      setFlips([]);
    } finally {
      setLoading(false);
    }
  };

  const Table = ({ title, rows }: { title: string; rows: Row[] }) => (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">{title}</h2>
      </div>
      {rows.length === 0 ? (
        <p className="muted">No games in this bucket.</p>
      ) : (
        <div className="table-wrapper">
          <table className="table">
            <thead>
              <tr>
                <th>Matchup</th>
                <th>Gameday Pick</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.matchup}</td>
                  <td>{r.pick}</td>
                  <td>{(r.prob * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  return (
    <div className="page">
      <header className="hero">
        <h1 className="hero-title">NFL Game Predictor</h1>
        <p className="hero-subtitle">
          Ensemble model using historical stats to pick weekly winners and confidence.
        </p>
      </header>

      <section className="card controls-card">
        <div className="card-header">
          <h2 className="card-title">Get Weekly Picks</h2>
          <p className="card-subtitle">
            Enter a season and week, then let the model crunch the numbers.
          </p>
        </div>

        <div className="form-row">
          <div className="field">
            <label htmlFor="season">Season</label>
            <input
              id="season"
              type="number"
              value={season}
              onChange={(e) => setSeason(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label htmlFor="week">Week</label>
            <input
              id="week"
              type="number"
              value={week}
              min={1}
              max={22}
              onChange={(e) => setWeek(Number(e.target.value))}
            />
          </div>
          <button className="primary-btn" onClick={getPicks} disabled={loading}>
            {loading ? "Loading..." : "Get Picks"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {!error && top.length === 0 && all.length === 0 && !loading && (
          <p className="muted small">
            Tip: start with a recent regular-season week (e.g. season 2024, week 5).
          </p>
        )}
      </section>

      <section className="grid">
        <Table title="Top Confidence Picks" rows={top} />
        <Table title="All Model Picks" rows={all} />
        <Table title="Coin Flips (Low Confidence)" rows={flips} />
      </section>
    </div>
  );
}
