import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import fs from 'fs';

const apiPort = parseInt(process.env.API_PORT || '8001', 10);
const backendPort = parseInt(process.env.PORT || '8000', 10);
const enableHttps = process.env.ENABLE_HTTPS === 'true';
const sslCertFile = process.env.SSL_CERT_FILE || '/data/ssl/server.crt';
const sslKeyFile = process.env.SSL_KEY_FILE || '/data/ssl/server.key';

// Configure HTTPS if enabled and certs exist
const httpsConfig = enableHttps && fs.existsSync(sslCertFile) && fs.existsSync(sslKeyFile)
  ? {
      key: fs.readFileSync(sslKeyFile),
      cert: fs.readFileSync(sslCertFile),
    }
  : undefined;

const backendProtocol = enableHttps ? 'https' : 'http';

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
    port: apiPort,
    strictPort: true,
    https: httpsConfig,
    proxy: {
      // Proxy all API calls to Python backend
      '^/(indexes|auth|authorize|token|health|docs|redoc|openapi.json|v1|mcp-routes|mcp-debug|mcp)': {
        target: `${backendProtocol}://127.0.0.1:${backendPort}`,
        changeOrigin: true,
        secure: false, // Allow self-signed certs
      },
    },
  },
});
