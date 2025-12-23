import { useMemo, useState } from "react";

type PickRow = {
  // existing fields
  matchup?: string; // "DAL @ PHI"
  pick?: string; // "DAL"
  confidence?: number; // 0..1

  home_team?: string; // "PHI"
  away_team?: string; // "DAL"
  home_win_prob?: number; // 0..1
  away_win_prob?: number; // 0..1

  // optional fields if present
  gameday?: string;
};

type ApiResponse = {
  top: PickRow[];
  all: PickRow[];
  flips: PickRow[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

function clamp01(x: number) {
  if (Number.isNaN(x)) return 0.5;
  return Math.max(0, Math.min(1, x));
}

function fmtPct(x: number) {
  return `${(clamp01(x) * 100).toFixed(2)}%`;
}

function parseMatchup(matchup?: string) {
  if (!matchup) return { away: "", home: "" };
  // supports "DAL @ PHI" or "DAL vs PHI"
  const at = matchup.split("@").map((s) => s.trim());
  if (at.length === 2) return { away: at[0], home: at[1] };

  const vs = matchup.split("vs").map((s) => s.trim());
  if (vs.length === 2) return { away: vs[0], home: vs[1] };

  return { away: matchup.trim(), home: "" };
}

function teamLogoUrl(abbr: string) {
  const a = (abbr || "").trim().toLowerCase();
  if (!a) return "";
  // ESPN CDN works well for NFL team abbreviations
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${a}.png`;
}

function deriveProbs(r: PickRow) {
  const matchupTeams = parseMatchup(r.matchup);
  const away = (r.away_team || matchupTeams.away || "").trim();
  const home = (r.home_team || matchupTeams.home || "").trim();

  const pick = (r.pick || "").trim();
  const conf = clamp01(typeof r.confidence === "number" ? r.confidence : 0.5);

  // Prefer explicit probs from API if available
  const hasExplicit =
    typeof r.home_win_prob === "number" || typeof r.away_win_prob === "number";

  let homeP = hasExplicit ? clamp01(r.home_win_prob ?? 1 - (r.away_win_prob ?? 0.5)) : 0.5;
  let awayP = hasExplicit ? clamp01(r.away_win_prob ?? 1 - homeP) : 0.5;

  // If no explicit probs, infer from pick + confidence
  if (!hasExplicit) {
    if (pick && away && pick.toUpperCase() === away.toUpperCase()) {
      awayP = conf;
      homeP = 1 - conf;
    } else if (pick && home && pick.toUpperCase() === home.toUpperCase()) {
      homeP = conf;
      awayP = 1 - conf;
    } else {
      // fallback: treat confidence as "winner prob" but we don't know side
      awayP = 0.5;
      homeP = 0.5;
    }
  }

  // Normalize just in case
  const sum = homeP + awayP;
  if (sum > 0) {
    homeP = homeP / sum;
    awayP = awayP / sum;
  }

  return { home, away, pick, homeP, awayP, conf };
}

function confidenceLabel(conf: number) {
  // you can tweak these thresholds
  if (conf >= 0.65) return { text: "High", tone: "high" as const };
  if (conf >= 0.55) return { text: "Medium", tone: "med" as const };
  return { text: "Low", tone: "low" as const };
}

function TeamRow(props: {
  abbr: string;
  prob: number;
  isPick: boolean;
}) {
  const { abbr, prob, isPick } = props;
  const logo = teamLogoUrl(abbr);
  return (
    <div className={`teamRow ${isPick ? "isPick" : ""}`}>
      <div className="teamLeft">
        {logo ? (
          <img className="logo" src={logo} alt={`${abbr} logo`} loading="lazy" />
        ) : (
          <div className="logoFallback" />
        )}
        <div className="teamName">{abbr || "—"}</div>
      </div>

      <div className="teamRight">
        <div className="bar">
          <div className="fill" style={{ width: `${clamp01(prob) * 100}%` }} />
        </div>
        <div className="pct">{fmtPct(prob)}</div>
      </div>
    </div>
  );
}

function PickCard({ row }: { row: PickRow }) {
  const { home, away, pick, homeP, awayP, conf } = deriveProbs(row);
  const badge = confidenceLabel(conf);
  const matchupText = row.matchup || (away && home ? `${away} @ ${home}` : "Matchup");

  const awayIsPick = !!pick && !!away && pick.toUpperCase() === away.toUpperCase();
  const homeIsPick = !!pick && !!home && pick.toUpperCase() === home.toUpperCase();

  return (
    <div className="pickCard">
      <div className="pickTop">
        <div className="leagueTag">
          <span className="dot" />
          NFL
        </div>

        <div className={`badge ${badge.tone}`}>{badge.text}</div>
      </div>

      <div className="matchup">{matchupText}</div>

      <div className="teams">
        <TeamRow abbr={away} prob={awayP} isPick={awayIsPick} />
        <TeamRow abbr={home} prob={homeP} isPick={homeIsPick} />
      </div>

      <div className="pickMeta">
        <div className="metaLine">
          <span className="metaKey">Pick</span>
          <span className="metaVal">{pick || "—"}</span>
        </div>
        {row.gameday ? (
          <div className="metaLine">
            <span className="metaKey">Game day</span>
            <span className="metaVal">{row.gameday}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function App() {
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const [tab, setTab] = useState<"top" | "all" | "flips">("top");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<"highToLow" | "lowToHigh">("highToLow");

  const [top, setTop] = useState<PickRow[]>([]);
  const [all, setAll] = useState<PickRow[]>([]);
  const [flips, setFlips] = useState<PickRow[]>([]);

  async function getPicks() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          season,
          week,
          top_k: 6,
          flip_band: 0.03,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data = (await res.json()) as ApiResponse;
      setTop(Array.isArray(data.top) ? data.top : []);
      setAll(Array.isArray(data.all) ? data.all : []);
      setFlips(Array.isArray(data.flips) ? data.flips : []);
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch picks.");
    } finally {
      setLoading(false);
    }
  }

  const rows = useMemo(() => {
    const base = tab === "top" ? top : tab === "flips" ? flips : all;

    const q = search.trim().toLowerCase();
    let filtered = base;

    if (q) {
      filtered = base.filter((r) => {
        const m = (r.matchup ?? "").toLowerCase();
        const p = (r.pick ?? "").toLowerCase();
        const ht = (r.home_team ?? "").toLowerCase();
        const at = (r.away_team ?? "").toLowerCase();
        return m.includes(q) || p.includes(q) || ht.includes(q) || at.includes(q);
      });
    }

    const sorted = [...filtered].sort((a, b) => {
      const ca = clamp01(typeof a.confidence === "number" ? a.confidence : 0.5);
      const cb = clamp01(typeof b.confidence === "number" ? b.confidence : 0.5);
      return sort === "highToLow" ? cb - ca : ca - cb;
    });

    return sorted;
  }, [tab, top, all, flips, search, sort]);

  return (
    <div className="app">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@300;400;600;700&family=Source+Code+Pro:wght@400;600;700&display=swap');

        :root{
          --bg0:#070a0a;
          --bg1:#0b0f10;
          --card:#0f1416;
          --card2:#0c1012;
          --stroke: rgba(255,255,255,.10);
          --stroke2: rgba(255,255,255,.08);
          --text: rgba(255,255,255,.92);
          --muted: rgba(255,255,255,.62);
          --muted2: rgba(255,255,255,.45);
          --green:#b6ff00;
          --green2:#7dff3a;
          --shadow: 0 24px 70px rgba(0,0,0,.55);
        }

        *{ box-sizing:border-box; }
        body{
              margin: 0;
              background: #000000;
              color: var(--text);
              font-family: "Saira Condensed", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            }


        .app{
          max-width: 1180px;
          margin: 0 auto;
          padding: 26px 18px 46px;
        }

        /* Header */
        .topBar{
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:16px;
          padding: 18px 18px;
          border: 1px solid var(--stroke2);
          border-radius: 18px;
          background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
          box-shadow: var(--shadow);
          backdrop-filter: blur(10px);
        }

        .brand{
          display:flex;
          gap:12px;
          align-items:center;
          min-width: 250px;
        }
        .mark{
          width:44px;height:44px;
          border-radius:14px;
          border:1px solid var(--stroke);
          display:grid;
          place-items:center;
          background: radial-gradient(14px 14px at 35% 30%, rgba(182,255,0,.22), transparent 60%),
                      rgba(255,255,255,.05);
          font-weight:700;
          letter-spacing:.5px;
        }
        .titleWrap{ display:flex; flex-direction:column; line-height:1; }
        .title{
          font-family:"Source Code Pro", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 20px;
          font-weight: 700;
          letter-spacing: .2px;
          margin:0;
        }
        .subtitle{
          margin:6px 0 0;
          color: var(--muted);
          font-size: 14px;
          letter-spacing:.2px;
        }

        .controls{
          display:flex;
          align-items:flex-end;
          gap:12px;
          flex-wrap:wrap;
          justify-content:flex-end;
        }
        .field{
          display:flex;
          flex-direction:column;
          gap:6px;
        }
        .field label{
          color: var(--muted);
          font-size: 13px;
          letter-spacing: .3px;
        }
        .field input{
          width: 140px;
          height: 38px;
          padding: 8px 10px;
          border-radius: 12px;
          border:1px solid var(--stroke2);
          background: rgba(0,0,0,.25);
          color: var(--text);
          outline:none;
        }
        .field input:focus{
          border-color: rgba(182,255,0,.35);
          box-shadow: 0 0 0 3px rgba(182,255,0,.10);
        }

        .btn{
          height: 38px;
          padding: 0 14px;
          border-radius: 12px;
          border: 1px solid var(--stroke2);
          background: rgba(255,255,255,.06);
          color: var(--text);
          font-weight: 600;
          cursor:pointer;
          transition: transform .08s ease, border-color .12s ease;
        }
        .btn:hover{ border-color: rgba(255,255,255,.16); }
        .btn:active{ transform: translateY(1px); }
        .btnPrimary{
          border-color: rgba(182,255,0,.35);
          background: linear-gradient(180deg, rgba(182,255,0,.18), rgba(182,255,0,.10));
        }
        .btn[disabled]{ opacity:.55; cursor:not-allowed; }

        /* Toolbar */
        .toolbar{
          margin-top: 16px;
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:12px;
          flex-wrap:wrap;
        }
        .tabs{
          display:flex;
          gap:10px;
          padding: 6px;
          border-radius: 14px;
          border:1px solid var(--stroke2);
          background: rgba(255,255,255,.04);
        }
        .tab{
          border:0;
          background: transparent;
          color: var(--muted);
          font-weight: 600;
          padding: 8px 12px;
          border-radius: 12px;
          cursor:pointer;
          letter-spacing:.2px;
        }
        .tabActive{
          color: var(--text);
          background: rgba(255,255,255,.06);
          border: 1px solid var(--stroke2);
        }

        .tools{
          display:flex;
          gap:10px;
          align-items:center;
        }
        .search{
          width: 260px;
          height: 38px;
          padding: 8px 12px;
          border-radius: 12px;
          border:1px solid var(--stroke2);
          background: rgba(0,0,0,.25);
          color: var(--text);
          outline:none;
        }
        .select{
          height: 38px;
          padding: 0 10px;
          border-radius: 12px;
          border:1px solid var(--stroke2);
          background: rgba(0,0,0,.25);
          color: var(--text);
          outline:none;
        }

        .error{
          margin-top: 14px;
          padding: 12px 14px;
          border-radius: 14px;
          border:1px solid rgba(255,90,90,.28);
          background: rgba(255,90,90,.10);
          color: rgba(255,220,220,.95);
          font-size: 14px;
        }

        /* Grid */
        .grid{
          margin-top: 14px;
          display:grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
        }
        @media (max-width: 980px){
          .grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .brand{ min-width: 0; }
        }
        @media (max-width: 640px){
          .grid{ grid-template-columns: 1fr; }
          .search{ width: 100%; }
          .tools{ width: 100%; }
        }

        /* Card */
        .pickCard{
          border-radius: 18px;
          border: 1px solid var(--stroke2);
          background:
            radial-gradient(900px 220px at 35% -40%, rgba(182,255,0,.12), transparent 55%),
            linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
          box-shadow: 0 18px 55px rgba(0,0,0,.45);
          padding: 14px 14px 12px;
          overflow:hidden;
          position:relative;
        }

        .pickTop{
          display:flex;
          align-items:center;
          justify-content:space-between;
          margin-bottom: 10px;
        }
        .leagueTag{
          display:flex;
          align-items:center;
          gap:8px;
          color: var(--muted);
          font-weight:600;
          letter-spacing:.25px;
        }
        .dot{
          width:10px;height:10px;border-radius:999px;
          background: rgba(255,255,255,.18);
          border: 1px solid rgba(255,255,255,.20);
        }

        .badge{
          font-size: 12px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid var(--stroke2);
          color: var(--muted);
          background: rgba(0,0,0,.20);
          font-weight: 700;
          letter-spacing:.3px;
        }
        .badge.high{
          color: rgba(210,255,190,.95);
          border-color: rgba(182,255,0,.28);
          background: rgba(182,255,0,.10);
        }
        .badge.med{
          color: rgba(220,235,255,.95);
          border-color: rgba(160,200,255,.20);
          background: rgba(160,200,255,.08);
        }
        .badge.low{
          color: rgba(255,255,255,.70);
        }

        .matchup{
          font-size: 18px;
          font-weight: 700;
          letter-spacing: .3px;
          margin-bottom: 12px;
        }

        .teams{
          display:flex;
          flex-direction:column;
          gap:10px;
          margin-bottom: 12px;
        }

        .teamRow{
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:12px;
          padding: 10px 10px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,.06);
          background: rgba(0,0,0,.18);
        }
        .teamRow.isPick{
          border-color: rgba(182,255,0,.26);
          background: linear-gradient(180deg, rgba(182,255,0,.10), rgba(0,0,0,.18));
        }

        .teamLeft{
          display:flex;
          align-items:center;
          gap:10px;
          min-width: 120px;
        }
        .logo{
          width:34px;height:34px;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(255,255,255,.06);
          object-fit: cover;
        }
        .logoFallback{
          width:34px;height:34px;border-radius:10px;
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(255,255,255,.06);
        }
        .teamName{
          font-size: 16px;
          font-weight: 700;
          letter-spacing:.3px;
        }

        .teamRight{
          display:flex;
          align-items:center;
          gap:10px;
          min-width: 150px;
          justify-content:flex-end;
        }
        .bar{
          width: 120px;
          height: 8px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,.10);
          background: rgba(255,255,255,.06);
          overflow:hidden;
        }
        .fill{
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, rgba(182,255,0,.80), rgba(182,255,0,.35));
        }
        .teamRow:not(.isPick) .fill{
          background: linear-gradient(90deg, rgba(255,255,255,.30), rgba(255,255,255,.12));
        }

        .pct{
          width: 70px;
          text-align:right;
          font-weight: 800;
          letter-spacing: .2px;
          color: var(--green);
        }
        .teamRow:not(.isPick) .pct{
          color: rgba(255,255,255,.60);
        }

        .pickMeta{
          border-top: 1px solid rgba(255,255,255,.08);
          padding-top: 10px;
          display:flex;
          flex-direction:column;
          gap:6px;
        }
        .metaLine{
          display:flex;
          align-items:center;
          justify-content:space-between;
        }
        .metaKey{
          color: var(--muted2);
          font-size: 13px;
          letter-spacing:.25px;
        }
        .metaVal{
          font-weight: 700;
          letter-spacing:.25px;
        }

        .empty{
          margin-top: 18px;
          color: var(--muted);
          font-size: 14px;
        }
      `}</style>

      <div className="topBar">
        <div className="brand">
        
          <div className="titleWrap">
            <h1 className="title">NFL Predictor</h1>
            <div className="subtitle">Weekly picks with confidence</div>
          </div>
        </div>

        <div className="controls">
          <div className="field">
            <label>Season</label>
            <input
              type="number"
              value={season}
              onChange={(e) => setSeason(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>Week</label>
            <input
              type="number"
              value={week}
              min={1}
              max={22}
              onChange={(e) => setWeek(Number(e.target.value))}
            />
          </div>
          <button className="btn btnPrimary" onClick={getPicks} disabled={loading}>
            {loading ? "Loading..." : "Get Picks"}
          </button>
        </div>
      </div>

      <div className="toolbar">
        <div className="tabs">
          <button
            className={`tab ${tab === "top" ? "tabActive" : ""}`}
            onClick={() => setTab("top")}
          >
            Top Picks
          </button>
          <button
            className={`tab ${tab === "all" ? "tabActive" : ""}`}
            onClick={() => setTab("all")}
          >
            All Picks
          </button>
          <button
            className={`tab ${tab === "flips" ? "tabActive" : ""}`}
            onClick={() => setTab("flips")}
          >
            Coin Flips
          </button>
        </div>

        <div className="tools">
          <input
            className="search"
            placeholder="Search team or matchup..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select className="select" value={sort} onChange={(e) => setSort(e.target.value as any)}>
            <option value="highToLow">Confidence: High → Low</option>
            <option value="lowToHigh">Confidence: Low → High</option>
          </select>
        </div>
      </div>

      {error ? <div className="error">{error}</div> : null}

      {rows.length === 0 && !loading && !error ? (
        <div className="empty">No picks yet — try hitting “Get Picks”.</div>
      ) : null}

      <div className="grid">
        {rows.map((r, idx) => (
          <PickCard key={`${r.matchup ?? "row"}-${idx}`} row={r} />
        ))}
      </div>
    </div>
  );
}
