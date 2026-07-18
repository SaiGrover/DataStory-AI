import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("plotly.js") || id.includes("react-plotly.js")) return "plotly";
          if (id.includes("react") || id.includes("react-dom") || id.includes("react-router-dom") || id.includes("react-query")) return "react";
          if (id.includes("framer-motion")) return "motion";
          if (id.includes("lucide-react")) return "icons";
          return "vendor";
        },
      },
    },
  },
  server: {
    port: 8501,
    strictPort: false,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
