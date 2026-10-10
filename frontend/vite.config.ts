/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000 and Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.VITE_API_TARGET ?? "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    // Plotly (cartesian bundle) is ~1.3 MB and is loaded lazily on the first chart.
    chunkSizeWarningLimit: 1600,
  },
  test: { environment: "node" },
});
