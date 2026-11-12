import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the explorer runs on Vite (5173) and proxies /api to the FastAPI
// server (`python -m ffx serve`, port 8000). In production FastAPI serves dist/ itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
