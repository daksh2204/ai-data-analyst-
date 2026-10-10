import { useContext, useEffect, useState } from "react";
import { Wand2 } from "lucide-react";
import { api, ApiError } from "../../api/client";
import { SessionContext } from "../../context";
import type { Dataset, QualityIssue, SessionState, TablePayload, TableProfile } from "../../types";
import { bytes, formatNumber } from "../../lib/format";
import { DataTable } from "../common/DataTable";
import { ScoreMeter } from "../chat/ArtifactView";

interface Props {
  datasets: Dataset[];
  table: string | null;
  onSelect: (name: string) => void;
  onSession: (s: SessionState) => void;
}

const ROLE_LABEL: Record<string, string> = {
  measure: "Measure",
  dimension: "Category",
  time: "Date",
  identifier: "Key",
  text: "Text",
  flag: "Yes/no",
};

function statSummary(c: TableProfile["columns"][number]): string {
  const s = c.stats as Record<string, unknown>;
  if (c.dtype === "integer" || c.dtype === "float") {
    return `${formatNumber(s.min)} to ${formatNumber(s.max)}, median ${formatNumber(s.median)}`;
  }
  if (c.dtype === "date" || c.dtype === "datetime") return `${s.min} to ${s.max} (${s.granularity})`;
  if (c.top_values.length) return c.top_values.slice(0, 3).map((t) => `${t.value} (${t.count})`).join(", ");
  return c.sample_values.slice(0, 3).map(String).join(", ");
}

export function DataView({ datasets, table, onSelect, onSession }: Props) {
  const { sessionId, notify } = useContext(SessionContext);
  const name = table && datasets.some((d) => d.name === table) ? table : datasets[0]?.name ?? null;
  const [detail, setDetail] = useState<{ dataset: Dataset; profile: TableProfile; preview: TablePayload } | null>(null);
  const [busyFix, setBusyFix] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId || !name) return;
    let cancelled = false;
    api
      .dataset(sessionId, name)
      .then((d) => !cancelled && setDetail(d))
      .catch((e) => notify(e instanceof ApiError ? e.message : "Couldn't load the table.", "error"));
    return () => {
      cancelled = true;
    };
  }, [sessionId, name, notify]);

  async function applyFix(issue: QualityIssue) {
    if (!sessionId || !name || !issue.fix) return;
    setBusyFix(issue.kind + issue.column);
    try {
      const res = await api.fix(sessionId, name, issue.fix, issue.column);
      notify(res.message, "success");
      onSession(res.session);
      const fresh = await api.dataset(sessionId, name);
      setDetail(fresh);
    } catch (e) {
      notify(e instanceof ApiError ? e.message : "The fix couldn't be applied.", "error");
    } finally {
      setBusyFix(null);
    }
  }

  if (!datasets.length) {
    return (
      <div className="pane-empty">
        <h2>No data yet</h2>
        <p>Upload a CSV file to see its columns, quality checks and a preview.</p>
      </div>
    );
  }

  const p = detail?.profile;
  return (
    <div className="data-view">
      <div className="pane-head">
        <div>
          <h1 className="pane-title">{detail?.dataset.filename ?? name}</h1>
          {detail && (
            <p className="pane-lede">
              Table <code className="inline-code">{detail.dataset.name}</code>: {detail.dataset.rows.toLocaleString("en-GB")} rows,{" "}
              {detail.dataset.columns} columns, {bytes(detail.dataset.size_bytes)}, {detail.dataset.encoding}
            </p>
          )}
        </div>
        {datasets.length > 1 && (
          <label className="select">
            <span className="sr-only">Table</span>
            <select value={name ?? ""} onChange={(e) => onSelect(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.filename}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {p && (
        <div className="data-grid">
          <section className="artifact">
            <div className="artifact__head">
              <span className="artifact__title">Quality checks</span>
              <ScoreMeter score={p.quality_score} />
            </div>
            {p.issues.length === 0 ? (
              <p className="muted">No issues found.</p>
            ) : (
              <ul className="issue-list">
                {p.issues.map((i, idx) => (
                  <li key={idx} className={`issue issue--${i.severity}`}>
                    <span className={`status-chip status-chip--${i.severity === "info" ? "info" : i.severity}`}>{i.severity}</span>
                    <span className="issue__text">
                      {i.column && <code className="inline-code">{i.column}</code>} {i.message}
                    </span>
                    {i.fix && (
                      <button type="button" className="ghost-button small" disabled={busyFix !== null} onClick={() => applyFix(i)}>
                        <Wand2 size={13} aria-hidden /> {i.fix === "drop_duplicates" ? "Remove duplicates" : "Standardise labels"}
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            {detail && detail.dataset.notes.length > 0 && (
              <>
                <h3 className="subhead">What changed on import</h3>
                <ul className="note-list">
                  {detail.dataset.notes.map((n, i) => (
                    <li key={i}>
                      {n.column && <code className="inline-code">{n.column}</code>} {n.message}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </section>

          <section className="artifact span-2">
            <div className="artifact__head">
              <span className="artifact__title">Columns</span>
            </div>
            <div className="data-table">
              <div className="data-table__scroll">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Column</th>
                      <th scope="col">Type</th>
                      <th scope="col">Used as</th>
                      <th scope="col" className="num">Missing</th>
                      <th scope="col" className="num">Distinct</th>
                      <th scope="col">Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.columns.map((c) => (
                      <tr key={c.name}>
                        <td>
                          <code className="inline-code">{c.name}</code>
                          {c.original_name !== c.name && <span className="muted"> {c.original_name}</span>}
                        </td>
                        <td>{c.dtype}</td>
                        <td>
                          <span className={`role role--${c.role}`}>{ROLE_LABEL[c.role] ?? c.role}</span>
                        </td>
                        <td className="num">{c.null_pct ? `${c.null_pct}%` : "0"}</td>
                        <td className="num">{c.unique_count.toLocaleString("en-GB")}</td>
                        <td className="truncate">{statSummary(c)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section className="artifact span-2">
            <div className="artifact__head">
              <span className="artifact__title">Preview</span>
            </div>
            {detail && <DataTable table={detail.preview} initialRows={15} rawHeaders filename={detail.dataset.name} />}
          </section>
        </div>
      )}
    </div>
  );
}
