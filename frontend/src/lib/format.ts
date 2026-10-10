export function formatNumber(v: unknown, digits = 2): string {
  if (v === null || v === undefined || v === "") return "";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "";
    if (Number.isInteger(v)) return v.toLocaleString("en-GB");
    return v.toLocaleString("en-GB", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

export function compact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}K`;
  if (a >= 100) return Math.round(v).toLocaleString("en-GB");
  return v.toLocaleString("en-GB", { maximumFractionDigits: 2 });
}

export function percent(v: number, signed = true): string {
  const s = `${(v * 100).toFixed(1)}%`;
  return signed && v > 0 ? `+${s}` : s;
}

export function bytes(n: number): string {
  if (n >= 1_048_576) return `${(n / 1_048_576).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} B`;
}

export function pretty(name: string): string {
  const t = name.replace(/_+/g, " ").trim();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function isNumericColumn(rows: unknown[][], index: number): boolean {
  let seen = 0;
  for (const r of rows.slice(0, 30)) {
    const v = r[index];
    if (v === null || v === undefined) continue;
    if (typeof v !== "number") return false;
    seen++;
  }
  return seen > 0;
}

export function toCsv(columns: string[], rows: unknown[][]): string {
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.map(esc).join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
}

export function download(filename: string, content: string, type = "text/csv"): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
