import { useState } from "react";
import { ChevronRight, Loader2, TriangleAlert } from "lucide-react";
import type { CodeBlock as Code, Step } from "../../types";
import { CodeBlock } from "../common/CodeBlock";
import { DataTable } from "../common/DataTable";

const TOOL_LABEL: Record<string, string> = {
  run_sql: "SQL query",
  run_pandas: "pandas",
  create_chart: "Chart",
  detect_anomalies: "Anomaly scan",
  forecast: "Forecast",
  profile_table: "Data profile",
};

function StepRow({ step, index }: { step: Step; index: number }) {
  const [open, setOpen] = useState(false);
  if (step.kind === "note") {
    return (
      <li className="ledger__step ledger__step--note">
        <span className="ledger__marker" aria-hidden />
        <p className="ledger__note">{step.text}</p>
      </li>
    );
  }
  const running = step.status === "running";
  const failed = step.status === "error";
  const hasDetail = !!(step.code || step.preview || (step.input && Object.keys(step.input).length));
  return (
    <li className={`ledger__step ${failed ? "is-error" : ""}`}>
      <span className="ledger__marker" aria-hidden>
        {running ? <Loader2 size={14} className="spin" /> : failed ? <TriangleAlert size={13} /> : index}
      </span>
      <div className="ledger__body">
        <button
          type="button"
          className="ledger__row"
          aria-expanded={open}
          onClick={() => hasDetail && setOpen((v) => !v)}
          disabled={!hasDetail}
        >
          <span className="ledger__purpose">{step.purpose || TOOL_LABEL[step.tool ?? ""] || step.tool}</span>
          <span className="ledger__meta">
            <span className="ledger__tool">{TOOL_LABEL[step.tool ?? ""] ?? step.tool}</span>
            {step.elapsed_ms !== undefined && <span>{Math.round(step.elapsed_ms)} ms</span>}
            {hasDetail && <ChevronRight size={14} className={`chev ${open ? "is-open" : ""}`} aria-hidden />}
          </span>
        </button>
        {step.summary && <p className={`ledger__summary ${failed ? "is-error" : ""}`}>{step.summary}</p>}
        {open && (
          <div className="ledger__detail">
            {step.code && <CodeBlock code={step.code.code} language={step.code.language} />}
            {!step.code && step.input && (
              <CodeBlock code={JSON.stringify(step.input, null, 2)} language="json" runnable={false} />
            )}
            {step.preview && <DataTable table={step.preview} dense initialRows={8} rawHeaders filename="step-result" />}
          </div>
        )}
      </div>
    </li>
  );
}

interface Props {
  steps: Step[];
  code: Code[];
  live?: boolean;
  defaultOpen?: boolean;
  elapsedMs?: number;
}

export function Working({ steps, code, live = false, defaultOpen = false, elapsedMs }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [tab, setTab] = useState<"steps" | "code">("steps");
  const toolSteps = steps.filter((s) => s.kind === "tool");
  if (!steps.length && !live) return null;
  const isOpen = open || live;
  const sqlCount = code.filter((c) => c.language === "sql").length;
  const pyCount = code.length - sqlCount;
  let n = 0;

  return (
    <section className={`ledger ${isOpen ? "is-open" : ""}`} aria-label="How this answer was worked out">
      <button type="button" className="ledger__toggle" aria-expanded={isOpen} onClick={() => setOpen((v) => !v)} disabled={live}>
        <ChevronRight size={15} className={`chev ${isOpen ? "is-open" : ""}`} aria-hidden />
        <span>{live ? "Working" : "Show working"}</span>
        <span className="ledger__count">
          {toolSteps.length} step{toolSteps.length === 1 ? "" : "s"}
          {elapsedMs !== undefined && !live ? `, ${(elapsedMs / 1000).toFixed(1)} s` : ""}
        </span>
      </button>
      {isOpen && (
        <div className="ledger__panel">
          {!live && code.length > 0 && (
            <div className="segmented" role="tablist" aria-label="Working view">
              <button type="button" role="tab" aria-selected={tab === "steps"} aria-pressed={tab === "steps"} onClick={() => setTab("steps")}>
                Steps <span className="count">{toolSteps.length}</span>
              </button>
              <button type="button" role="tab" aria-selected={tab === "code"} aria-pressed={tab === "code"} onClick={() => setTab("code")}>
                Code{" "}
                <span className="count">
                  {sqlCount ? `${sqlCount} SQL` : ""}
                  {sqlCount && pyCount ? ", " : ""}
                  {pyCount ? `${pyCount} pandas` : ""}
                </span>
              </button>
            </div>
          )}
          {tab === "steps" || live ? (
            <ol className="ledger__steps">
              {steps.map((s) => {
                if (s.kind === "tool") n += 1;
                return <StepRow key={s.id} step={s} index={n} />;
              })}
            </ol>
          ) : (
            <div className="ledger__code">
              {code.map((c, i) => (
                <CodeBlock key={i} code={c.code} language={c.language} />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
