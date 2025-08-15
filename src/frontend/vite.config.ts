import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false, // Disable sourcemaps in production for smaller bundle
    minify: "esbuild", // Use esbuild for fast minification
    rollupOptions: {
      output: {
        // Manual chunking for optimal loading
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          mui: ["@mui/material", "@mui/icons-material", "@emotion/react", "@emotion/styled"],
          charts: ["recharts"],
        },
        // Optimize chunk size
        chunkFileNames: "assets/[name]-[hash].js",
        entryFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
    // Optimize for ARM64/Pi 5
    target: "es2020", // Modern target for Pi 5's browser
    cssCodeSplit: true, // Split CSS for better caching
    assetsInlineLimit: 4096, // Inline assets < 4kb
    reportCompressedSize: false, // Disable gzip size reporting (slow on Pi)
    chunkSizeWarningLimit: 1000, // Warn if chunk > 1MB (target < 5MB total)
  },
  optimizeDeps: {
    // Pre-bundle dependencies for faster cold starts on Pi 5
    include: ["react", "react-dom", "@mui/material", "@emotion/react", "@emotion/styled"],
  },
});
