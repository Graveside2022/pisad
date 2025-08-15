import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { visualizer } from "rollup-plugin-visualizer";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Bundle analyzer - generates stats.html after build
    visualizer({
      filename: "./dist/stats.html",
      open: false,
      gzipSize: true,
      brotliSize: true,
    }) as any,
  ],
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
    minify: "esbuild", // Fast minification for Pi 5
    rollupOptions: {
      output: {
        // Optimized chunking strategy for <1.5MB total
        manualChunks(id) {
          // Split MUI icons into separate chunks by usage
          if (id.includes("@mui/icons-material")) {
            // Group icons by category to allow selective loading
            if (id.includes("CheckCircle") || id.includes("Error") || id.includes("Warning")) {
              return "icons-status";
            }
            if (id.includes("PlayArrow") || id.includes("Stop") || id.includes("Pause")) {
              return "icons-control";
            }
            if (id.includes("Settings") || id.includes("Save") || id.includes("Cancel")) {
              return "icons-config";
            }
            return "icons-misc";
          }

          // Core vendor chunk (React ecosystem)
          if (id.includes("node_modules")) {
            if (id.includes("react") || id.includes("react-dom") || id.includes("react-router")) {
              return "react-vendor";
            }

            // MUI core (without icons)
            if (id.includes("@mui/material") || id.includes("@emotion")) {
              return "mui-core";
            }

            // Charts as separate chunk (loaded on-demand)
            if (id.includes("recharts") || id.includes("d3")) {
              return "charts";
            }

            // Utilities
            if (id.includes("axios") || id.includes("lodash")) {
              return "utils";
            }
          }
        },
        // Optimize chunk size
        chunkFileNames: "assets/[name]-[hash:8].js",
        entryFileNames: "assets/[name]-[hash:8].js",
        assetFileNames: "assets/[name]-[hash:8][extname]",
      },
      // Tree-shaking optimizations
      treeshake: {
        preset: "recommended",
        moduleSideEffects: false,
        propertyReadSideEffects: false,
        tryCatchDeoptimization: false,
      },
    },
    // Optimize for ARM64/Pi 5
    target: "es2020", // Modern target for Pi 5's browser
    cssCodeSplit: true, // Split CSS for better caching
    cssMinify: true, // Enable CSS minification
    assetsInlineLimit: 4096, // Inline assets < 4kb
    reportCompressedSize: false, // Disable gzip size reporting (slow on Pi)
    chunkSizeWarningLimit: 500, // Warn if chunk > 500KB (target < 1.5MB total)
  },
  optimizeDeps: {
    // Pre-bundle dependencies for faster cold starts on Pi 5
    include: ["react", "react-dom", "@mui/material", "@emotion/react", "@emotion/styled"],
    // Exclude large libraries from pre-bundling
    exclude: ["@mui/icons-material"],
    // Force optimization of ESM packages
    esbuildOptions: {
      target: "es2020",
      // Enable tree-shaking for optimized deps
      treeShaking: true,
    },
  },
  // Additional optimizations
  esbuild: {
    legalComments: "none", // Remove all comments
    minifyIdentifiers: true,
    minifySyntax: true,
    minifyWhitespace: true,
  },
});
