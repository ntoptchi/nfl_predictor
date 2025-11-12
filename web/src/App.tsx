// web/src/App.tsx
import { useMemo, useState } from "react";

type PickRow = {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  confidence: number; // 0..1
};

const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

function clsConfidence(p: number) {
  if (p >= 0.66) return "conf high";
  if (p >= 0.55) return "conf mid";
  return "conf low";
}

function ConfidenceCell({
  p,
  showBars,
}: {
  p: number;
  showBars: boolean;
}) {
  const pct = (p * 100).toFixed(2) + "%";

  if (!showBars) {
    return <span className={clsConfidence(p)}>{pct}</span>;
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div className="bar-wrap">
        <div
          className="bar-fill"
          style={{ width: `${Math.max(2, Math.min(100, p * 100))}%` }}
        />
      </div>
      <span className={clsConfidence(p)} style={{ minWidth: 64 }}>
        {pct}
      </span>
    </div>
  );
}

export default function App() {
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(1);
  const [data, setData] = useState<PickRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [showBars, setShowBars] = useState(true); // toggle
  const [error, setError] = useState<string>("");

  const [top, flips, all] = useMemo(() => {
    const sorted = [...data].sort((a, b) => b.confidence - a.confidence);
    const top = sorted.slice(0, 5);
    const flips = sorted.filter(
      (r) => r.confidence >= 0.48 && r.confidence <= 0.52
    );
    return [top, flips, sorted];
  }, [data]);

  async function getPicks() {
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season, week }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as PickRow[];
      setData(json);
    } catch (e: any) {
      setError(`Failed to fetch picks: ${e?.message ?? e}`);
      setData([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>NFL Game Predictor</h1>

      <div className="card" style={{ marginTop: "1rem" }}>
        <div className="controls">
          <div>
            <label className="label">Season</label>
            <input
              className="input"
              type="number"
              value={season}
              onChange={(e) => setSeason(parseInt(e.target.value || "0", 10))}
              placeholder="2025"
            />
          </div>
          <div>
            <label className="label">Week</label>
            <input
              className="input"
              type="number"
              value={week}
              onChange={(e) => setWeek(parseInt(e.target.value || "0", 10))}
              placeholder="1"
            />
          </div>

          <button className="button" onClick={getPicks}>
            {loading ? "Fetching…" : "Get Picks"}
          </button>

          <label className="toggle">
            <input
              type="checkbox"
              checked={showBars}
              onChange={(e) => setShowBars(e.target.checked)}
            />
            Show probability bars
          </label>
        </div>

        {error && (
          <div className="mt-3" style={{ color: "#f87171" }}>
            {error}
          </div>
        )}

        {loading && <div className="spinner" />}

        {!loading && data.length === 0 && (
          <p className="mt-3 muted">
            Enter a season & week, then click “Get Picks”.
          </p>
        )}
      </div>

      {!loading && data.length > 0 && (
        <>
          <h2 className="mt-6">Top Confidence Picks</h2>
          <div className="card">
            <table className="table">
              <thead>
                <tr>
                  <th>Matchup</th>
                  <th>Gameday</th>
                  <th>Pick</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {top.map((r) => (
                  <tr key={`top_${r.game_id}`}>
                    <td>
                      {r.away_team} @ {r.home_team}
                    </td>
                    <td>{r.gameday || "—"}</td>
                    <td>{r.predicted_winner}</td>
                    <td>
                      <ConfidenceCell p={r.confidence} showBars={showBars} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="mt-6">All Picks</h2>
          <div className="card">
            <table className="table">
              <thead>
                <tr>
                  <th>Matchup</th>
                  <th>Gameday</th>
                  <th>Pick</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {all.map((r) => (
                  <tr key={r.game_id}>
                    <td>
                      {r.away_team} @ {r.home_team}
                    </td>
                    <td>{r.gameday || "—"}</td>
                    <td>{r.predicted_winner}</td>
                    <td>
                      <ConfidenceCell p={r.confidence} showBars={showBars} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="mt-6">Coin Flips (low confidence)</h2>
          <div className="card">
            {flips.length === 0 ? (
              <p className="muted">No coin flips this week.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>Matchup</th>
                    <th>Gameday</th>
                    <th>Pick</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {flips.map((r) => (
                    <tr key={`flip_${r.game_id}`}>
                      <td>
                        {r.away_team} @ {r.home_team}
                      </td>
                      <td>{r.gameday || "—"}</td>
                      <td>{r.predicted_winner}</td>
                      <td>
                        <ConfidenceCell p={r.confidence} showBars={showBars} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}