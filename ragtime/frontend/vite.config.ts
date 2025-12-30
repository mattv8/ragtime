import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

const indexerPort = parseInt(process.env.INDEXER_PORT || '8001', 10);
const apiPort = parseInt(process.env.API_PORT || '8000', 10);

export default defineConfig({
  plugins: [react()],
  base: '/',
  appType: 'spa',
  cacheDir: resolve(__dirname, '.vite'),
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: '0.0.0.0',
    port: indexerPort,
    strictPort: true,
    proxy: {
      // Proxy all API calls to Python backend
      '^/(indexes|auth|health|docs|redoc|openapi.json|v1)': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
});
