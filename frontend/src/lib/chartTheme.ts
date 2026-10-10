import type { PlotlyFigure } from "../types";

// Categorical slots in fixed order (validated for colour-vision deficiency, light and dark
// stepped separately). Server traces carry `meta.role`; colour is applied here per theme.
const SERIES = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
const STATUS_CRITICAL = "#d03b3b";

export type ThemeMode = "light" | "dark";

interface Tokens {
  surface: string;
  text: string;
  textMuted: string;
  grid: string;
  axis: string;
}

const TOKENS: Record<ThemeMode, Tokens> = {
  light: { surface: "#ffffff", text: "#15212b", textMuted: "#5b6872", grid: "#e9edf0", axis: "#cfd7dd" },
  dark: { surface: "#141e26", text: "#e3eaef", textMuted: "#9aa8b3", grid: "#22303a", axis: "#33434f" },
};

const SEQUENTIAL = {
  light: [[0, "#cde2fb"], [0.35, "#86b6ef"], [0.7, "#2a78d6"], [1, "#0d366b"]],
  dark: [[0, "#184f95"], [0.4, "#256abf"], [0.75, "#5598e7"], [1, "#b7d3f6"]],
};
const DIVERGING = {
  light: [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
  dark: [[0, "#e66767"], [0.5, "#383835"], [1, "#3987e5"]],
};

export function currentTheme(): ThemeMode {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function themeFigure(figure: PlotlyFigure, mode: ThemeMode, height?: number): PlotlyFigure {
  const t = TOKENS[mode];
  const series = SERIES[mode];
  const data = figure.data.map((raw) => {
    const trace: Record<string, unknown> = structuredClone(raw);
    const meta = (trace.meta as { role?: string; index?: number }) ?? {};
    const role = meta.role ?? "series";
    const type = trace.type as string;
    const color =
      role === "series"
        ? series[(meta.index ?? 0) % series.length]
        : role === "anomaly"
          ? STATUS_CRITICAL
          : role === "baseline"
            ? t.textMuted
            : role === "trend"
              ? t.text
              : role === "forecast" || role === "band" || role === "band-inner"
                ? series[0]
                : series[0];
    if (type === "pie") {
      trace.marker = { ...(trace.marker as object), colors: series, line: { color: t.surface, width: 2 } };
      trace.textfont = { color: "#ffffff" };
      trace.outsidetextfont = { color: t.text };
    } else if (type === "heatmap") {
      trace.colorscale = role === "diverging" ? DIVERGING[mode] : SEQUENTIAL[mode];
      trace.colorbar = { outlinewidth: 0, tickfont: { color: t.textMuted }, thickness: 10 };
      trace.textfont = { color: mode === "dark" ? "#f0f4f7" : "#15212b" };
    } else if (type === "bar" || type === "histogram") {
      trace.marker = { ...(trace.marker as object), color, line: { width: 0 } };
    } else if (type === "box") {
      trace.marker = { ...(trace.marker as object), color };
      trace.line = { color, width: 1.5 };
      trace.fillcolor = withAlpha(color, 0.12);
    } else {
      const marker = (trace.marker as Record<string, unknown>) ?? {};
      trace.line = { ...(trace.line as object), color };
      trace.marker = { ...marker, color, line: { color: t.surface, width: 2 } };
      if (role === "band") trace.fillcolor = withAlpha(series[0], 0.1);
      if (role === "band-inner") trace.fillcolor = withAlpha(series[0], 0.2);
      if (trace.fill && role === "series") trace.fillcolor = withAlpha(color, 0.12);
    }
    return trace;
  });

  const layout: Record<string, unknown> = structuredClone(figure.layout ?? {});
  const axis = (a: unknown) => ({
    ...(a as object),
    gridcolor: t.grid,
    linecolor: t.axis,
    zerolinecolor: t.axis,
    tickfont: { color: t.textMuted, size: 11 },
    title: { ...((a as { title?: object })?.title ?? {}), font: { color: t.textMuted, size: 12 } },
    showline: false,
    ticks: "",
  });
  Object.assign(layout, {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "'Instrument Sans', system-ui, sans-serif", color: t.text, size: 12 },
    xaxis: axis(layout.xaxis),
    yaxis: axis(layout.yaxis),
    legend: { orientation: "h", y: -0.18, x: 0, font: { color: t.textMuted, size: 12 }, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.text, family: "'Instrument Sans', system-ui" } },
    separators: ".,",
    autosize: true,
  });
  if (height) layout.height = height;
  return { data, layout };
}
