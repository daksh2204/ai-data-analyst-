import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, Download, Eraser, LayoutDashboard, Menu, MessageSquareText, Table2, Upload, X } from "lucide-react";
import { api, ApiError, saveToken } from "./api/client";
import { SessionContext, ThemeContext } from "./context";
import { useTheme } from "./hooks/useTheme";
import { useChat } from "./hooks/useChat";
import type { AppConfig, AssistantMessage, SessionState } from "./types";
import { Sidebar } from "./components/Sidebar";
import { Conversation } from "./components/chat/Conversation";
import { Composer } from "./components/chat/Composer";
import { Dashboard } from "./components/dashboard/Dashboard";
import { DataView } from "./components/data/DataView";

const SESSION_KEY = "ledger.sessionId";
type Tab = "ask" | "dashboard" | "data";
interface Toast {
  id: number;
  message: string;
  tone: "info" | "error" | "success";
}

function storedSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}
function storeSessionId(id: string | null) {
  try {
    if (id) localStorage.setItem(SESSION_KEY, id);
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}

export default function App() {
  const { pref, mode, setPref } = useTheme();
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [needToken, setNeedToken] = useState(false);
  const [tokenInput, setTokenInput] = useState("");
  const [tab, setTab] = useState<Tab>("ask");
  const [activeTable, setActiveTable] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const [dataVersion, setDataVersion] = useState(0);
  const [bootError, setBootError] = useState<string | null>(null);
  const dragDepth = useRef(0);

  const notify = useCallback((message: string, tone: Toast["tone"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-3), { id, message, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), tone === "error" ? 7000 : 4000);
  }, []);

  const handleError = useCallback(
    (e: unknown, fallback: string) => {
      if (e instanceof ApiError && e.status === 401) {
        setNeedToken(true);
        return;
      }
      notify(e instanceof ApiError ? e.message : fallback, "error");
    },
    [notify],
  );

  const boot = useCallback(async () => {
    setBootError(null);
    try {
      const cfg = await api.config();
      setConfig(cfg);
      const existing = storedSessionId();
      let s: SessionState | null = null;
      if (existing) {
        try {
          s = await api.getSession(existing);
        } catch (e) {
          if (e instanceof ApiError && e.status === 401) throw e;
          s = null;
        }
      }
      s ??= await api.createSession();
      storeSessionId(s.id);
      setSession(s);
      setNeedToken(false);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setNeedToken(true);
      else setBootError(e instanceof ApiError ? e.message : "Couldn't start the app.");
    }
  }, []);

  useEffect(() => {
    void boot();
  }, [boot]);

  const onFinal = useCallback((question: string, message: AssistantMessage) => {
    setSession((s) =>
      s
        ? {
            ...s,
            messages: [...s.messages, { id: `u_${message.id}`, role: "user", content: question, created_at: message.created_at }, message],
          }
        : s,
    );
  }, []);
  const { live, ask, stop, busy, clearLive } = useChat(session?.id ?? null, onFinal);

  const askAndShow = useCallback(
    (q: string) => {
      setTab("ask");
      void ask(q);
    },
    [ask],
  );

  async function upload(files: File[]) {
    if (!session) return;
    const csv = files.filter((f) => /\.(csv|tsv|txt)$/i.test(f.name));
    if (!csv.length) {
      notify("Only CSV files can be uploaded (.csv, .tsv or .txt).", "error");
      return;
    }
    setUploading(true);
    try {
      const res = await api.upload(session.id, csv);
      setSession(res.session);
      setDataVersion((v) => v + 1);
      const ok = res.results.filter((r) => r.ok);
      const failed = res.results.filter((r) => !r.ok);
      if (ok.length) notify(`Added ${ok.map((r) => r.filename).join(", ")}.`, "success");
      failed.forEach((r) => notify(r.error?.message ?? `${r.filename} couldn't be read.`, "error"));
    } catch (e) {
      handleError(e, "The upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function loadSamples() {
    if (!session) return;
    setUploading(true);
    try {
      const res = await api.loadSamples(session.id);
      setSession(res.session);
      setDataVersion((v) => v + 1);
      notify("Loaded the sample sales data: orders, customers, products and regional targets.", "success");
    } catch (e) {
      handleError(e, "Couldn't load the sample data.");
    } finally {
      setUploading(false);
    }
  }

  async function removeDataset(name: string) {
    if (!session) return;
    try {
      const s = await api.removeDataset(session.id, name);
      setSession(s);
      setDataVersion((v) => v + 1);
      notify(`Removed ${name}.`);
    } catch (e) {
      handleError(e, "Couldn't remove the file.");
    }
  }

  async function newSession() {
    stop();
    clearLive();
    storeSessionId(null);
    setSession(null);
    setActiveTable(null);
    setTab("ask");
    await boot();
  }

  async function clearConversation() {
    if (!session) return;
    try {
      setSession(await api.resetConversation(session.id));
      clearLive();
    } catch (e) {
      handleError(e, "Couldn't clear the conversation.");
    }
  }

  // Drag-and-drop anywhere on the window.
  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    };
    const over = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const leave = () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length) void upload(files);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  });

  const ctx = useMemo(() => ({ sessionId: session?.id ?? null, notify }), [session?.id, notify]);
  const datasets = session?.datasets ?? [];
  const messages = session?.messages ?? [];
  const hasAnswers = messages.some((m) => m.role === "assistant");

  const tabs: { key: Tab; label: string; Icon: typeof MessageSquareText }[] = [
    { key: "ask", label: "Ask", Icon: MessageSquareText },
    { key: "dashboard", label: "Dashboard", Icon: LayoutDashboard },
    { key: "data", label: "Data", Icon: Table2 },
  ];

  return (
    <ThemeContext.Provider value={mode}>
      <SessionContext.Provider value={ctx}>
        <div className={`shell ${drawer ? "drawer-open" : ""}`}>
          <a href="#main" className="skip-link">
            Skip to content
          </a>
          <Sidebar
            config={config}
            datasets={datasets}
            relationships={session?.relationships ?? []}
            activeTable={tab === "data" ? activeTable ?? datasets[0]?.name ?? null : null}
            uploading={uploading}
            themePref={pref}
            onTheme={setPref}
            onUpload={(f) => void upload(f)}
            onSamples={() => void loadSamples()}
            onOpenDataset={(n) => {
              setActiveTable(n);
              setTab("data");
              setDrawer(false);
            }}
            onRemoveDataset={(n) => void removeDataset(n)}
            onNewSession={() => void newSession()}
          />
          <button type="button" className="scrim" aria-label="Close menu" tabIndex={drawer ? 0 : -1} onClick={() => setDrawer(false)} />

          <main id="main" className="main">
            <header className="topbar">
              <button type="button" className="icon-button menu-button" aria-label="Open menu" onClick={() => setDrawer(true)}>
                <Menu size={18} aria-hidden />
              </button>
              <nav className="tabs" aria-label="Views">
                {tabs.map(({ key, label, Icon }) => (
                  <button key={key} type="button" className="tab" aria-current={tab === key ? "page" : undefined} onClick={() => setTab(key)}>
                    <Icon size={15} aria-hidden /> {label}
                  </button>
                ))}
              </nav>
              <div className="topbar__actions">
                {tab === "ask" && hasAnswers && (
                  <button type="button" className="ghost-button" onClick={() => void clearConversation()} disabled={busy}>
                    <Eraser size={15} aria-hidden /> <span className="hide-sm">Clear</span>
                  </button>
                )}
                {session && hasAnswers && (
                  <details className="menu">
                    <summary className="ghost-button">
                      <Download size={15} aria-hidden /> <span className="hide-sm">Export</span>
                    </summary>
                    <div className="menu__list">
                      <a href={api.reportUrl(session.id, "html")} download>
                        HTML report with charts
                      </a>
                      <a href={api.reportUrl(session.id, "md")} download>
                        Markdown
                      </a>
                    </div>
                  </details>
                )}
              </div>
            </header>

            {bootError && (
              <div className="pane-empty">
                <h2>Can't reach the server</h2>
                <p>{bootError}</p>
                <button type="button" className="button button--primary" onClick={() => void boot()}>
                  Try again
                </button>
              </div>
            )}

            {!bootError && tab === "ask" && (
              <div className="ask">
                {datasets.length === 0 && session ? (
                  <div className="welcome">
                    <div className="welcome__inner">
                      <BarChart3 size={28} aria-hidden className="welcome__icon" />
                      <h1 className="welcome__title">Ask your spreadsheets questions</h1>
                      <p className="welcome__lede">
                        Drop in CSV exports and ask things like "which region generated the highest revenue?" You get the answer, a chart, the SQL behind
                        it and any anomalies worth a look.
                      </p>
                      <div className="welcome__actions">
                        <label className="button button--primary button--lg">
                          <Upload size={17} aria-hidden /> Upload CSV files
                          <input
                            type="file"
                            accept=".csv,.tsv,.txt,text/csv"
                            multiple
                            className="sr-only"
                            onChange={(e) => {
                              const f = Array.from(e.target.files ?? []);
                              if (f.length) void upload(f);
                              e.target.value = "";
                            }}
                          />
                        </label>
                        {config?.samples_available && (
                          <button type="button" className="button button--lg" onClick={() => void loadSamples()} disabled={uploading}>
                            Try the sample sales data
                          </button>
                        )}
                      </div>
                      <p className="welcome__fine">Files stay in this session on the server and are discarded when it expires.</p>
                    </div>
                  </div>
                ) : (
                  <Conversation
                    messages={messages}
                    live={live}
                    suggestions={session?.suggestions ?? []}
                    hasData={datasets.length > 0}
                    offline={!!config?.offline}
                    onPick={askAndShow}
                  />
                )}
                {datasets.length > 0 && (
                  <div className="composer-wrap">
                    <Composer
                      disabled={!session}
                      busy={busy}
                      placeholder={hasAnswers ? "Ask a follow-up, e.g. now break that down by month" : "Ask a question about your data"}
                      onSend={askAndShow}
                      onStop={stop}
                      prefill={prefill}
                    />
                    {hasAnswers && !busy && (session?.suggestions.length ?? 0) > 0 && (
                      <div className="quick-asks" aria-label="More questions">
                        {session!.suggestions.slice(0, 4).map((s) => (
                          <button key={s} type="button" onClick={() => setPrefill({ text: s, nonce: Date.now() })}>
                            {s}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            {!bootError && tab === "dashboard" && <Dashboard datasets={datasets} version={dataVersion} onAsk={askAndShow} />}
            {!bootError && tab === "data" && (
              <DataView
                datasets={datasets}
                table={activeTable}
                onSelect={setActiveTable}
                onSession={(s) => {
                  setSession(s);
                  setDataVersion((v) => v + 1);
                }}
              />
            )}
          </main>

          {dragging && (
            <div className="drop-overlay" aria-hidden>
              <div className="drop-overlay__card">
                <Upload size={26} />
                <span>Drop CSV files to add them</span>
              </div>
            </div>
          )}

          <div className="toasts" role="status" aria-live="polite">
            {toasts.map((t) => (
              <div key={t.id} className={`toast toast--${t.tone}`}>
                <span>{t.message}</span>
                <button type="button" className="icon-button" aria-label="Dismiss" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>
                  <X size={14} aria-hidden />
                </button>
              </div>
            ))}
          </div>

          {needToken && (
            <div className="modal-backdrop">
              <form
                className="modal"
                onSubmit={(e) => {
                  e.preventDefault();
                  saveToken(tokenInput.trim() || null);
                  void boot();
                }}
              >
                <h2>Access token required</h2>
                <p>This workspace is protected. Enter the access token configured on the server (APP_ACCESS_TOKEN).</p>
                <label htmlFor="token" className="sr-only">
                  Access token
                </label>
                <input id="token" type="password" autoFocus value={tokenInput} onChange={(e) => setTokenInput(e.target.value)} placeholder="Access token" />
                <button type="submit" className="button button--primary">
                  Continue
                </button>
              </form>
            </div>
          )}
        </div>
      </SessionContext.Provider>
    </ThemeContext.Provider>
  );
}
