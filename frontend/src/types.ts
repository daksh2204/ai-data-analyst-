// Shapes returned by the API. Kept in one place so components agree on them.

export interface ColumnSummary {
  name: string;
  dtype: string;
  role: string;
}

export interface IngestNote {
  level: "info" | "warning";
  message: string;
  column: string | null;
}

export interface Dataset {
  name: string;
  filename: string;
  rows: number;
  columns: number;
  size_bytes: number;
  encoding: string;
  delimiter: string;
  quality_score: number;
  issue_count: number;
  notes: IngestNote[];
  column_summary: ColumnSummary[];
}

export interface Relationship {
  left_table: string;
  left_column: string;
  right_table: string;
  right_column: string;
  overlap: number;
  cardinality: string;
}

export interface TablePayload {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
}

export interface PlotlyFigure {
  data: Array<Record<string, unknown>>;
  layout: Record<string, unknown>;
}

export interface Anomaly {
  kind: "outlier" | "rule" | "timeseries";
  column: string;
  value: number | null;
  expected: number | null;
  score: number;
  severity: "high" | "medium";
  reason: string;
  row_index: number | null;
  date: string | null;
  record: Record<string, unknown>;
}

export interface QualityIssue {
  severity: "info" | "warning" | "critical";
  kind: string;
  message: string;
  column: string | null;
  count: number | null;
  fix: string | null;
}

export type Artifact =
  | ({ kind: "chart"; id: string; title: string; chart_type: string; figure: PlotlyFigure; notes: string[]; result_id?: string })
  | ({ kind: "table"; id: string; title: string; result_id: string } & TablePayload)
  | {
      kind: "anomalies";
      id: string;
      title: string;
      summary: string;
      method: string;
      counts: Record<string, number>;
      items: Anomaly[];
      figure: PlotlyFigure | null;
    }
  | {
      kind: "forecast";
      id: string;
      title: string;
      summary: {
        method: string;
        frequency: string;
        backtest_mape_pct: number | null;
        notes: string[];
        forecast_total: number;
      };
      figure: PlotlyFigure;
      table: TablePayload;
    }
  | { kind: "quality"; id: string; title: string; table: string; score: number; issues: QualityIssue[]; rows: number; columns: number };

export interface CodeBlock {
  language: "sql" | "python";
  code: string;
}

export interface Step {
  id: string;
  kind: "tool" | "note";
  tool?: string;
  purpose?: string;
  text?: string;
  input?: Record<string, unknown>;
  status?: "running" | "ok" | "error";
  summary?: string;
  preview?: TablePayload | null;
  code?: CodeBlock | null;
  elapsed_ms?: number;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  question: string;
  answer: string;
  artifacts: Artifact[];
  steps: Step[];
  code: CodeBlock[];
  provider: string;
  model: string;
  usage: { input_tokens: number; output_tokens: number; cached_input_tokens: number; llm_calls: number };
  elapsed_ms: number;
  cached: boolean;
  created_at: number;
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  created_at: number;
}

export type ChatMessage = AssistantMessage | UserMessage;

export interface SessionState {
  id: string;
  created_at: number;
  datasets: Dataset[];
  relationships: Relationship[];
  messages: ChatMessage[];
  suggestions: string[];
  provider: string;
  model: string;
}

export interface AppConfig {
  app_name: string;
  provider: string;
  model: string;
  offline: boolean;
  auth_required: boolean;
  limits: { max_upload_mb: number; max_files_per_session: number; max_rows_per_file: number };
  samples_available: boolean;
}

export interface ColumnProfile {
  name: string;
  original_name: string;
  dtype: string;
  role: string;
  null_count: number;
  null_pct: number;
  unique_count: number;
  unique_pct: number;
  sample_values: unknown[];
  top_values: { value: string; count: number }[];
  stats: Record<string, unknown>;
}

export interface TableProfile {
  name: string;
  row_count: number;
  column_count: number;
  columns: ColumnProfile[];
  issues: QualityIssue[];
  quality_score: number;
  duplicate_rows: number;
  memory_mb: number;
}

export interface DashboardData {
  table: string;
  kpis: { label: string; value: number; format: "integer" | "compact" | "percent"; detail?: string }[];
  charts: { id: string; title: string; chart_type: string; figure: PlotlyFigure; sql: string; span: number }[];
  quality_score: number;
  issues: QualityIssue[];
}

/** Events streamed by POST /chat. */
export type ChatEvent =
  | { type: "start"; turn_id: string; provider: string; model: string }
  | { type: "notice"; level: string; message: string }
  | { type: "token"; text: string }
  | { type: "thought"; text: string }
  | { type: "step_start"; step: Step }
  | { type: "step_end"; step: Step }
  | { type: "artifact"; artifact: Artifact }
  | { type: "final"; message: AssistantMessage }
  | { type: "error"; code: string; message: string };
