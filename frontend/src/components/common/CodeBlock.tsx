import { useContext, useMemo, useState } from "react";
import hljs from "highlight.js/lib/core";
import sql from "highlight.js/lib/languages/sql";
import python from "highlight.js/lib/languages/python";
import { Check, Copy, Play } from "lucide-react";
import { api, ApiError } from "../../api/client";
import { SessionContext } from "../../context";
import type { TablePayload } from "../../types";
import { DataTable } from "./DataTable";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("python", python);

interface Props {
  code: string;
  language?: string;
  runnable?: boolean;
}

export function CodeBlock({ code, language = "", runnable }: Props) {
  const { sessionId } = useContext(SessionContext);
  const [copied, setCopied] = useState(false);
  const [result, setResult] = useState<(TablePayload & { elapsed_ms: number }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const lang = language === "py" ? "python" : language;
  const html = useMemo(() => {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  }, [code, lang]);
  const canRun = (runnable ?? lang === "sql") && !!sessionId;

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  async function run() {
    if (!sessionId) return;
    setRunning(true);
    setError(null);
    try {
      setResult(await api.query(sessionId, code));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "The query failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="code-block">
      <div className="code-block__bar">
        <span className="code-block__lang">{lang === "python" ? "pandas" : lang ? lang.toUpperCase() : "Code"}</span>
        <span className="code-block__actions">
          {canRun && (
            <button type="button" className="ghost-button small" onClick={run} disabled={running}>
              <Play size={13} aria-hidden /> {running ? "Running" : "Run"}
            </button>
          )}
          <button type="button" className="ghost-button small" onClick={copy}>
            {copied ? <Check size={13} aria-hidden /> : <Copy size={13} aria-hidden />} {copied ? "Copied" : "Copy"}
          </button>
        </span>
      </div>
      <pre>
        <code className={`hljs language-${lang}`} dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
      {error && <p className="code-block__error">{error}</p>}
      {result && (
        <div className="code-block__result">
          <DataTable table={result} dense initialRows={8} rawHeaders filename="query-result" />
        </div>
      )}
    </div>
  );
}
