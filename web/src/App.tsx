import { useState } from "react";
import "./App.css";

type GamePick = {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  confidence: number;
};

type PredictResponse = {
  picks: GamePick[];
  top_confidence: GamePick[];
  coin_flips: GamePick[];
};

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PredictResponse | null>(null);

  async function fetchPicks() {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          season,
          week,
          top_k: 5,
          flip_band: 0.02,
        }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const json = (await res.json()) as PredictResponse;
      setData(json);
    } catch (err: any) {
      console.error(err);
      setError(err?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>NFL Game Predictor</h1>
        <p>Enter a season + week to get model picks and confidence.</p>
      </header>

      <section className="controls">
        <div className="field">
          <label>Season</label>
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(parseInt(e.target.value || "0"))}
          />
        </div>
        <div className="field">
          <label>Week</label>
          <input
            type="number"
            value={week}
            onChange={(e) => setWeek(parseInt(e.target.value || "0"))}
          />
        </div>
        <button onClick={fetchPicks} disabled={loading}>
          {loading ? "Loading…" : "Get Picks"}
        </button>
      </section>

      {error && <p className="error">{error}</p>}

      {data && (
        <main className="results">
          <Section title="Top Confidence Picks">
            <PicksTable rows={data.top_confidence} />
          </Section>

          <Section title="All Picks">
            <PicksTable rows={data.picks} />
          </Section>

          <Section title="Coin Flips (low confidence)">
            <PicksTable rows={data.coin_flips} />
          </Section>
        </main>
      )}
    </div>
  );
}

function Section(props: { title: string; children: React.ReactNode }) {
  return (
    <section className="section">
      <h2>{props.title}</h2>
      {props.children}
    </section>
  );
}

function PicksTable({ rows }: { rows: GamePick[] }) {
  if (!rows || rows.length === 0) {
    return <p className="muted">None</p>;
  }

  return (
    <table className="picks-table">
      <thead>
        <tr>
          <th>Matchup</th>
          <th>Gameday</th>
          <th>Pick</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.game_id}>
            <td>
              {r.away_team} @ {r.home_team}
            </td>
            <td>{r.gameday}</td>
            <td>{r.predicted_winner}</td>
            <td>{(r.confidence * 100).toFixed(2)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default App;
