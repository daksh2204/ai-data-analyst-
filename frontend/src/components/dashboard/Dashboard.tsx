import { useContext, useEffect, useState } from "react";
import { api, ApiError } from "../../api/client";
import { SessionContext } from "../../context";
import type { DashboardData, Dataset } from "../../types";
import { compact, percent } from "../../lib/format";
import { Plot } from "../charts/Plot";
import { QualityIssues, ScoreMeter } from "../chat/ArtifactView";

interface Props {
  datasets: Dataset[];
  version: number;
  onAsk: (q: string) => void;
}

export function Dashboard({ datasets, version, onAsk }: Props) {
  const { sessionId } = useContext(SessionContext);
  const candidates = datasets.filter((d) => d.column_summary.some((c) => c.role === "measure"));
  const [table, setTable] = useState<string | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const active = table && datasets.some((d) => d.name === table) ? table : null;

  useEffect(() => {
    if (!sessionId || !datasets.length) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .dashboard(sessionId, active ?? undefined)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e instanceof ApiError ? e.message : "Couldn't build the dashboard."))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [sessionId, active, version, datasets.length]);

  if (!datasets.length) {
    return (
      <div className="pane-empty">
        <h2>No data yet</h2>
        <p>Upload a CSV file to generate a dashboard of its key measures.</p>
      </div>
    );
  }

  return (
    <div className="dashboard" aria-busy={loading}>
      <div className="pane-head">
        <div>
          <h1 className="pane-title">Dashboard</h1>
          <p className="pane-lede">Built automatically from the column profile. No model calls, so it is instant and reproducible.</p>
        </div>
        {candidates.length > 1 && (
          <label className="select">
            <span className="sr-only">Table</span>
            <select value={data?.table ?? ""} onChange={(e) => setTable(e.target.value)}>
              {candidates.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.filename}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {error && <p className="notice notice--error">{error}</p>}
      {data && (
        <>
          <div className="kpis">
            {data.kpis.map((k) => (
              <div key={k.label} className="kpi">
                <span className="kpi__label">{k.label}</span>
                <span className="kpi__value">
                  {k.format === "percent" ? percent(k.value) : k.format === "compact" ? compact(k.value) : k.value.toLocaleString("en-GB")}
                </span>
                {k.detail && <span className="kpi__detail">{k.detail}</span>}
              </div>
            ))}
          </div>
          <div className="dash-grid">
            {data.charts.map((c) => (
              <figure key={c.id} className={`artifact chart-card ${c.span === 2 ? "span-2" : ""}`}>
                <figcaption className="artifact__head">
                  <span className="artifact__title">{c.title}</span>
                </figcaption>
                <Plot figure={c.figure} height={c.span === 2 ? 300 : 280} label={c.title} />
              </figure>
            ))}
            <section className="artifact">
              <div className="artifact__head">
                <span className="artifact__title">Data quality</span>
                <ScoreMeter score={data.quality_score} />
              </div>
              <QualityIssues issues={data.issues} />
              <button type="button" className="link-button" onClick={() => onAsk(`Detect anomalies in ${data.table}.`)}>
                Scan {data.table} for anomalies
              </button>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
