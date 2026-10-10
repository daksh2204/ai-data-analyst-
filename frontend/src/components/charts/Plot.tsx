import { memo, useContext, useEffect, useRef } from "react";
import type { PlotlyFigure } from "../../types";
import { themeFigure } from "../../lib/chartTheme";
import { ThemeContext } from "../../context";

type PlotlyModule = {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
  Plots: { resize: (el: HTMLElement) => void };
};

let plotlyPromise: Promise<PlotlyModule> | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  // Plotly is large; load it on first use so the shell paints immediately.
  plotlyPromise ??= import("plotly.js-cartesian-dist-min").then((m) => (m.default ?? m) as unknown as PlotlyModule);
  return plotlyPromise;
}

interface Props {
  figure: PlotlyFigure;
  height?: number;
  label: string;
}

function PlotImpl({ figure, height = 320, label }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mode = useContext(ThemeContext);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    loadPlotly().then((Plotly) => {
      if (cancelled || !el) return;
      const themed = themeFigure(figure, mode, height);
      Plotly.react(el, themed.data, themed.layout, {
        displayModeBar: false,
        responsive: true,
        scrollZoom: false,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [figure, mode, height]);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      loadPlotly().then((Plotly) => el.isConnected && Plotly.Plots.resize(el));
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      loadPlotly().then((Plotly) => Plotly.purge(el));
    };
  }, []);

  return <div ref={ref} className="plot" style={{ height }} role="img" aria-label={label} />;
}

export const Plot = memo(PlotImpl);
