import { useRef } from "react";
import { Database, FilePlus2, Link2, Monitor, Moon, RotateCcw, Sun, X } from "lucide-react";
import type { AppConfig, Dataset, Relationship } from "../types";
import type { ThemePreference } from "../hooks/useTheme";

interface Props {
  config: AppConfig | null;
  datasets: Dataset[];
  relationships: Relationship[];
  activeTable: string | null;
  uploading: boolean;
  themePref: ThemePreference;
  onTheme: (p: ThemePreference) => void;
  onUpload: (files: File[]) => void;
  onSamples: () => void;
  onOpenDataset: (name: string) => void;
  onRemoveDataset: (name: string) => void;
  onNewSession: () => void;
}

function qualityTone(score: number) {
  return score >= 85 ? "good" : score >= 60 ? "warning" : "critical";
}

export function Sidebar(props: Props) {
  const { config, datasets, relationships, activeTable, uploading, themePref } = props;
  const input = useRef<HTMLInputElement>(null);
  const manyToOne = relationships.filter((r) => r.cardinality === "many-to-one");

  return (
    <aside className="sidebar">
      <div className="brand">
        <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden>
          <rect width="32" height="32" rx="7" className="brand__tile" />
          <path d="M9 8v16h14" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M13 19l3.5-4 3 2.5L23 12" fill="none" stroke="#9FE3CF" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div>
          <div className="brand__name">Ledger</div>
          <div className="brand__tag">AI data analyst</div>
        </div>
      </div>

      <div className="sidebar__section">
        <div className="sidebar__heading">
          <span>Files</span>
          {datasets.length > 0 && <span className="muted">{datasets.length}</span>}
        </div>
        {datasets.length === 0 && <p className="sidebar__empty">No files yet.</p>}
        <ul className="dataset-list">
          {datasets.map((d) => (
            <li key={d.name} className={activeTable === d.name ? "is-active" : undefined}>
              <button type="button" className="dataset" onClick={() => props.onOpenDataset(d.name)} title={`Open ${d.filename}`}>
                <Database size={15} aria-hidden className="dataset__icon" />
                <span className="dataset__text">
                  <span className="dataset__name">{d.filename}</span>
                  <span className="dataset__meta">
                    {d.rows.toLocaleString("en-GB")} rows, {d.columns} columns
                  </span>
                </span>
                <span className={`score score--${qualityTone(d.quality_score)}`} title="Data quality score">
                  {d.quality_score}
                </span>
              </button>
              <button
                type="button"
                className="icon-button dataset__remove"
                aria-label={`Remove ${d.filename}`}
                onClick={() => props.onRemoveDataset(d.name)}
              >
                <X size={14} aria-hidden />
              </button>
            </li>
          ))}
        </ul>
        <input
          ref={input}
          type="file"
          accept=".csv,.tsv,.txt,text/csv"
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            if (files.length) props.onUpload(files);
            e.target.value = "";
          }}
        />
        <div className="sidebar__actions">
          <button type="button" className="button button--primary" onClick={() => input.current?.click()} disabled={uploading}>
            <FilePlus2 size={16} aria-hidden /> {uploading ? "Uploading" : "Add CSV files"}
          </button>
          {config?.samples_available && !datasets.some((d) => d.name === "sales_orders") && (
            <button type="button" className="button" onClick={props.onSamples} disabled={uploading}>
              Load sample data
            </button>
          )}
        </div>
        <p className="sidebar__fine">Up to {config?.limits.max_upload_mb ?? 50} MB per file. Drop files anywhere.</p>
      </div>

      {manyToOne.length > 0 && (
        <div className="sidebar__section">
          <div className="sidebar__heading">
            <span>Connections</span>
          </div>
          <ul className="relations">
            {manyToOne.slice(0, 6).map((r, i) => (
              <li key={i}>
                <Link2 size={13} aria-hidden />
                <span>
                  {r.left_table}.<b>{r.left_column}</b> to {r.right_table}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="sidebar__footer">
        <div className="engine" title={config ? `${config.provider} ${config.model}` : ""}>
          <span className={`engine__dot ${config?.offline ? "is-offline" : ""}`} aria-hidden />
          <span className="engine__text">
            {config ? (config.offline ? "Offline engine" : `${config.provider}: ${config.model}`) : "Connecting"}
          </span>
        </div>
        <div className="sidebar__footer-row">
          <div className="theme-switch" role="group" aria-label="Colour theme">
            {(
              [
                ["light", Sun, "Light"],
                ["system", Monitor, "System"],
                ["dark", Moon, "Dark"],
              ] as const
            ).map(([key, Icon, label]) => (
              <button key={key} type="button" aria-pressed={themePref === key} aria-label={`${label} theme`} title={`${label} theme`} onClick={() => props.onTheme(key)}>
                <Icon size={14} aria-hidden />
              </button>
            ))}
          </div>
          <button type="button" className="ghost-button small" onClick={props.onNewSession} title="Start over with a new session">
            <RotateCcw size={13} aria-hidden /> New session
          </button>
        </div>
      </div>
    </aside>
  );
}
