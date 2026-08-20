import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import fs from 'fs';

const apiPort = parseInt(process.env.API_PORT || '8001', 10);
const backendPort = parseInt(process.env.PORT || '8000', 10);
const enableHttps = process.env.ENABLE_HTTPS === 'true';
const sslCertFile = process.env.SSL_CERT_FILE || '/data/ssl/server.crt';
const sslKeyFile = process.env.SSL_KEY_FILE || '/data/ssl/server.key';

// Configure HTTPS if enabled and certs exist
const httpsConfig =
  enableHttps && fs.existsSync(sslCertFile) && fs.existsSync(sslKeyFile)
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
  test: {
    environment: 'jsdom',
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 3500,
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'index.html'),
        'share-theme': resolve(__dirname, 'src/styles/share-theme-entry.css'),
      },
      output: {
        assetFileNames: (assetInfo) => {
          const names = [
            ...(assetInfo.names ?? []),
            ...((assetInfo.originalFileNames as string[] | undefined) ?? []),
          ];
          if (
            names.some(
              (name) => name.endsWith('share-theme.css') || name.endsWith('share-theme-entry.css'),
            )
          ) {
            return 'assets/share-theme.css';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: apiPort,
    strictPort: true,
    https: httpsConfig,
    // Bind-mounted source on Docker/macOS does not deliver native FS events
    // reliably; enable polling so HMR detects edits made from the host.
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      // Proxy all API calls to Python backend
      '^/(indexes|auth|authorize|token|health|docs|redoc|openapi.json|v1|mcp-routes|mcp-debug|mcp)':
        {
          target: `${backendProtocol}://127.0.0.1:${backendPort}`,
          changeOrigin: true,
          ws: true,
          secure: false, // Allow self-signed certs
          configure: (proxy) => {
            // Suppress noisy WebSocket proxy errors (socket hang up, timeouts)
            // that occur during normal runtime startup/shutdown cycles
            proxy.on('error', (err, _req, res) => {
              const msg = (err as NodeJS.ErrnoException).message || '';
              if (
                msg.includes('socket hang up') ||
                msg.includes('ECONNRESET') ||
                msg.includes('ECONNREFUSED')
              ) {
                return; // Expected during runtime transitions
              }
              // For non-WebSocket responses, send a 502 if headers haven't been sent
              const httpRes = res as import('node:http').ServerResponse;
              if (res && 'writeHead' in res && !httpRes.headersSent) {
                httpRes.writeHead(502, { 'Content-Type': 'text/plain' });
                httpRes.end('Proxy error');
              }
            });
          },
        },
    },
  },
});
