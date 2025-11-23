import { useState } from "react";

type Row = {
  matchup: string;
  pick: string;
  prob: number; // 0–1
};

type ApiPayload = Record<string, any>;

function coerceTables(payload: ApiPayload) {
  // Accept a few possible shapes so we don't crash if the backend changes keys
  const top = (payload.top_picks ?? payload.top ?? payload.top_confidence ?? []) as any[];
  const all = (payload.all_picks ?? payload.all ?? payload.picks ?? []) as any[];
  const flips = (payload.coin_flips ?? payload.flips ?? []) as any[];

  // Normalize rows to (matchup, pick, prob)
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
      // If you added a Vite proxy (below), call /api/predict.
      // Otherwise, hit FastAPI directly.
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
        "Failed to fetch picks. Make sure the API server is running (uvicorn on :8000) and try again."
      );
      // Clear old tables so the screen doesn't look stale
      setTop([]);
      setAll([]);
      setFlips([]);
    } finally {
      setLoading(false);
    }
  };

  const Table = ({ title, rows }: { title: string; rows: Row[] }) => (
    <section className="section">
      <h2 className="h2">{title}</h2>
      {rows.length === 0 ? (
        <p className="muted">No rows.</p>
      ) : (
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
      )}
    </section>
  );

  return (
    <div className="wrap">
      <h1 className="h1">NFL Game Predictor</h1>

      <p className="muted">Enter a season + week to get model picks and confidence.</p>

      <div className="form">
        <label>
          Season
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
          />
        </label>
        <label>
          Week
          <input
            type="number"
            value={week}
            onChange={(e) => setWeek(Number(e.target.value))}
          />
        </label>
        <button onClick={getPicks} disabled={loading}>
          {loading ? "Loading..." : "Get Picks"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <Table title="Top Confidence Picks" rows={top} />
      <Table title="All Picks" rows={all} />
      <Table title="Coin Flips (low confidence)" rows={flips} />
    </div>
  );
}
