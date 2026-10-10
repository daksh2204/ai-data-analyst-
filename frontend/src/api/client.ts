import type { AppConfig, ChatEvent, DashboardData, Dataset, SessionState, TablePayload, TableProfile } from "../types";

const TOKEN_KEY = "ledger.accessToken";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function saveToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: the token lives for this page only */
  }
  memoryToken = token;
}

let memoryToken: string | null = readToken();

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  if (memoryToken) h.Authorization = `Bearer ${memoryToken}`;
  return h;
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const err = body?.error ?? {};
    return new ApiError(res.status, err.code ?? "error", err.message ?? res.statusText);
  } catch {
    return new ApiError(res.status, "error", `Request failed (${res.status}).`);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: headers((init.headers as Record<string, string>) ?? {}) });
  } catch {
    throw new ApiError(0, "network", "Can't reach the server. Check that the API is running.");
  }
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(body),
  headers: { "Content-Type": "application/json" },
});

export interface UploadResult {
  results: { filename: string; ok: boolean; dataset?: Dataset; error?: { code: string; message: string } }[];
  session: SessionState;
}

export const api = {
  config: () => request<AppConfig>("/api/config"),
  createSession: () => request<SessionState>("/api/sessions", { method: "POST" }),
  getSession: (id: string) => request<SessionState>(`/api/sessions/${id}`),
  resetConversation: (id: string) => request<SessionState>(`/api/sessions/${id}/reset`, { method: "POST" }),
  upload: (id: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    return request<UploadResult>(`/api/sessions/${id}/datasets`, { method: "POST", body: form });
  },
  loadSamples: (id: string) => request<UploadResult>(`/api/sessions/${id}/datasets/samples`, { method: "POST" }),
  dataset: (id: string, table: string) =>
    request<{ dataset: Dataset; profile: TableProfile; preview: TablePayload }>(`/api/sessions/${id}/datasets/${table}`),
  removeDataset: (id: string, table: string) =>
    request<SessionState>(`/api/sessions/${id}/datasets/${table}`, { method: "DELETE" }),
  fix: (id: string, table: string, fix: string, column?: string | null) =>
    request<{ message: string; profile: TableProfile; session: SessionState }>(
      `/api/sessions/${id}/datasets/${table}/fix`,
      json({ fix, column: column ?? null }),
    ),
  dashboard: (id: string, table?: string) =>
    request<DashboardData>(`/api/sessions/${id}/dashboard${table ? `?table=${encodeURIComponent(table)}` : ""}`),
  query: (id: string, sql: string) =>
    request<TablePayload & { sql: string; elapsed_ms: number }>(`/api/sessions/${id}/query`, json({ sql })),
  reportUrl: (id: string, format: "html" | "md") =>
    `/api/sessions/${id}/report?format=${format}${memoryToken ? `&token=${encodeURIComponent(memoryToken)}` : ""}`,
};

/** Split a text/event-stream buffer into complete `data:` payloads, returning the unparsed remainder. */
export function parseSseChunk(buffer: string): { events: string[]; rest: string } {
  const events: string[] = [];
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() ?? "";
  for (const block of blocks) {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, ""))
      .join("\n");
    if (data) events.push(data);
  }
  return { events, rest };
}

/** POST a question and stream the agent's events. */
export async function streamChat(
  sessionId: string,
  message: string,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`/api/sessions/${sessionId}/chat`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({ message }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    throw new ApiError(0, "network", "Can't reach the server. Check that the API is running.");
  }
  if (!res.ok || !res.body) throw await parseError(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSseChunk(buffer);
    buffer = rest;
    for (const raw of events) {
      try {
        onEvent(JSON.parse(raw) as ChatEvent);
      } catch {
        /* ignore a malformed event rather than abort the stream */
      }
    }
  }
}
