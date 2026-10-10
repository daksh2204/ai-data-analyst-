import { useMemo, useState } from "react";
import { AlertTriangle, CircleAlert, Info, TableProperties } from "lucide-react";
import type { Anomaly, Artifact, PlotlyFigure, QualityIssue, TablePayload } from "../../types";
import { Plot } from "../charts/Plot";
import { DataTable } from "../common/DataTable";
import { compact, formatNumber, pretty } from "../../lib/format";

/** Flatten a figure's traces into a table so every chart has an accessible data view. */
export function figureToTable(fig: PlotlyFigure): TablePayload | null {
  const rows: unknown[][] = [];
  let columns: string[] = [];
  for (const t of fig.data) {
    const name = (t.name as string) ?? "";
    if (t.type === "pie") {
      columns = ["Label", "Value"];
      const labels = (t.labels as unknown[]) ?? [];
      const values = (t.values as unknown[]) ?? [];
      labels.forEach((l, i) => rows.push([l, values[i]]));
    } else if (Array.isArray(t.x) && Array.isArray(t.y) && !(t.meta as { role?: string })?.role?.startsWith("band")) {
      columns = ["Series", "X", "Y"];
      const horizontal = t.orientation === "h";
      (t.x as unknown[]).forEach((x, i) => {
        const y = (t.y as unknown[])[i];
        rows.push(horizontal ? [name, y, x] : [name, x, y]);
      });
    }
  }
  if (!rows.length) return null;
  return { columns, rows: rows.slice(0, 500), row_count: rows.length, truncated: rows.length > 500 };
}

function ChartCard({ title, figure, notes, height = 320 }: { title: string; figure: PlotlyFigure; notes?: string[]; height?: number }) {
  const [showData, setShowData] = useState(false);
  const table = useMemo(() => (showData ? figureToTable(figure) : null), [showData, figure]);
  return (
    <figure className="artifact chart-card">
      <figcaption className="artifact__head">
        <span className="artifact__title">{title}</span>
        <button type="button" className="ghost-button small" aria-pressed={showData} onClick={() => setShowData((v) => !v)}>
          <TableProperties size={14} aria-hidden /> {showData ? "Hide data" : "Data"}
        </button>
      </figcaption>
      <Plot figure={figure} height={height} label={title} />
      {notes && notes.length > 0 && <p className="artifact__note">{notes.join(" ")}</p>}
      {table && <DataTable table={table} dense initialRows={10} rawHeaders filename={title} />}
    </figure>
  );
}

const KIND_LABEL: Record<Anomaly["kind"], string> = { outlier: "Unusual value", rule: "Rule break", timeseries: "Unusual period" };

function AnomalyItem({ a }: { a: Anomaly }) {
  const record = Object.entries(a.record ?? {}).slice(0, 6);
  return (
    <li className={`anomaly anomaly--${a.severity}`}>
      <div className="anomaly__head">
        <span className={`status-chip status-chip--${a.severity === "high" ? "critical" : "warning"}`}>
          {a.severity === "high" ? <CircleAlert size={13} aria-hidden /> : <AlertTriangle size={13} aria-hidden />}
          {a.severity === "high" ? "High" : "Medium"}
        </span>
        <span className="anomaly__kind">{KIND_LABEL[a.kind]}</span>
        <span className="anomaly__where">{a.date ?? pretty(a.column)}</span>
      </div>
      <p className="anomaly__reason">{a.reason}</p>
      {record.length > 0 && (
        <dl className="anomaly__record">
          {record.map(([k, v]) => (
            <div key={k}>
              <dt>{pretty(k)}</dt>
              <dd>{formatNumber(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </li>
  );
}

function AnomalyCard({ art }: { art: Extract<Artifact, { kind: "anomalies" }> }) {
  const [all, setAll] = useState(false);
  const [filter, setFilter] = useState<"all" | Anomaly["kind"]>("all");
  const items = art.items.filter((a) => filter === "all" || a.kind === filter);
  const shown = all ? items : items.slice(0, 6);
  const filters: { key: "all" | Anomaly["kind"]; label: string; count: number }[] = [
    { key: "all", label: "All", count: art.items.length },
    { key: "outlier", label: "Unusual values", count: art.counts.outlier ?? 0 },
    { key: "timeseries", label: "Unusual periods", count: art.counts.timeseries ?? 0 },
    { key: "rule", label: "Rule breaks", count: art.counts.rule ?? 0 },
  ];
  return (
    <section className="artifact anomaly-card" aria-label={art.title}>
      <div className="artifact__head">
        <span className="artifact__title">{art.title}</span>
      </div>
      <p className="artifact__lede">{art.summary}</p>
      {art.figure && <Plot figure={art.figure} height={280} label={`${art.title} over time`} />}
      <div className="segmented" role="group" aria-label="Filter anomalies">
        {filters
          .filter((f) => f.key === "all" || f.count > 0)
          .map((f) => (
            <button key={f.key} type="button" aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>
              {f.label} <span className="count">{f.count}</span>
            </button>
          ))}
      </div>
      <ul className="anomaly-list">
        {shown.map((a, i) => (
          <AnomalyItem key={`${a.kind}-${a.row_index}-${a.date}-${i}`} a={a} />
        ))}
      </ul>
      {items.length > 6 && (
        <button type="button" className="link-button" onClick={() => setAll((v) => !v)}>
          {all ? "Show fewer" : `Show all ${items.length}`}
        </button>
      )}
      <p className="artifact__note">
        <Info size={13} aria-hidden /> {art.method}
      </p>
    </section>
  );
}

function ForecastCard({ art }: { art: Extract<Artifact, { kind: "forecast" }> }) {
  const [showTable, setShowTable] = useState(false);
  const s = art.summary;
  return (
    <section className="artifact" aria-label={art.title}>
      <div className="artifact__head">
        <span className="artifact__title">{art.title}</span>
        <button type="button" className="ghost-button small" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}>
          <TableProperties size={14} aria-hidden /> {showTable ? "Hide values" : "Values"}
        </button>
      </div>
      <Plot figure={art.figure} height={300} label={art.title} />
      <p className="artifact__note">
        {s.method}. Forecast total {compact(s.forecast_total)}
        {s.backtest_mape_pct !== null ? `; backtest error ${s.backtest_mape_pct.toFixed(1)}% MAPE` : ""}.
        {s.notes.filter((n) => !n.startsWith("Backtest")).map((n) => ` ${n}`)}
      </p>
      {showTable && <DataTable table={art.table} dense initialRows={12} filename="forecast" />}
    </section>
  );
}

function scoreTone(score: number): "good" | "warning" | "critical" {
  return score >= 85 ? "good" : score >= 60 ? "warning" : "critical";
}

export function QualityIssues({ issues }: { issues: QualityIssue[] }) {
  if (!issues.length) return <p className="muted">No issues found.</p>;
  return (
    <ul className="issue-list">
      {issues.map((i, idx) => (
        <li key={idx} className={`issue issue--${i.severity}`}>
          <span className={`status-chip status-chip--${i.severity === "info" ? "info" : i.severity}`}>
            {i.severity === "critical" ? <CircleAlert size={13} aria-hidden /> : i.severity === "warning" ? <AlertTriangle size={13} aria-hidden /> : <Info size={13} aria-hidden />}
            {pretty(i.severity)}
          </span>
          <span>
            {i.column && <code className="inline-code">{i.column}</code>} {i.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function ScoreMeter({ score }: { score: number }) {
  const tone = scoreTone(score);
  return (
    <div className={`meter meter--${tone}`} role="meter" aria-valuenow={score} aria-valuemin={0} aria-valuemax={100} aria-label="Data quality score">
      <div className="meter__track">
        <div className="meter__fill" style={{ width: `${score}%` }} />
      </div>
      <span className="meter__value">{score}/100</span>
    </div>
  );
}

function QualityCard({ art }: { art: Extract<Artifact, { kind: "quality" }> }) {
  return (
    <section className="artifact" aria-label={art.title}>
      <div className="artifact__head">
        <span className="artifact__title">{art.title}</span>
        <ScoreMeter score={art.score} />
      </div>
      <QualityIssues issues={art.issues} />
    </section>
  );
}

export function ArtifactView({ artifact }: { artifact: Artifact }) {
  switch (artifact.kind) {
    case "chart":
      return <ChartCard title={artifact.title} figure={artifact.figure} notes={artifact.notes} />;
    case "table":
      return (
        <section className="artifact" aria-label={artifact.title}>
          <div className="artifact__head">
            <span className="artifact__title">{artifact.title}</span>
          </div>
          <DataTable table={artifact} filename={artifact.title} />
        </section>
      );
    case "anomalies":
      return <AnomalyCard art={artifact} />;
    case "forecast":
      return <ForecastCard art={artifact} />;
    case "quality":
      return <QualityCard art={artifact} />;
    default:
      return null;
  }
}

const ORDER: Record<Artifact["kind"], number> = { anomalies: 0, forecast: 0, chart: 1, table: 2, quality: 3 };

export function sortArtifacts(list: Artifact[]): Artifact[] {
  return [...list].sort((a, b) => ORDER[a.kind] - ORDER[b.kind]);
}
