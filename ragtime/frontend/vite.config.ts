import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import fs from 'fs';
import type { Plugin } from 'vite';

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

const proxiedPathPattern =
  /^\/(?:indexes|auth|authorize|token|health|docs|redoc|openapi\.json|v1|mcp-routes|mcp-debug|mcp)(?:\/|$)/;

function hasHost(host: string | undefined, hostname: string, port: number): boolean {
  const normalizedHost = host?.toLowerCase();
  return normalizedHost === hostname || normalizedHost === `${hostname}:${port}`;
}

const localPreviewOriginPlugin: Plugin = {
  name: 'local-preview-origin-redirect',
  apply: 'serve',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const requestPath = req.url || '/';
      if (!requestPath.startsWith('/') || requestPath.startsWith('//')) {
        next();
        return;
      }
      const pathname = new URL(requestPath, 'http://vite.local').pathname;
      const accept = req.headers.accept || '';
      const fetchDestination = req.headers['sec-fetch-dest'];
      const isLoopbackHost =
        hasHost(req.headers.host, 'localhost', apiPort) ||
        hasHost(req.headers.host, '127.0.0.1', apiPort) ||
        hasHost(req.headers.host, '[::1]', apiPort);

      if (
        req.method !== 'GET' ||
        !isLoopbackHost ||
        !accept.includes('text/html') ||
        (fetchDestination && fetchDestination !== 'document') ||
        proxiedPathPattern.test(pathname)
      ) {
        next();
        return;
      }

      const protocol = httpsConfig ? 'https' : 'http';
      const redirectUrl = new URL(`${protocol}://app.lvh.me:${apiPort}${requestPath}`);
      res.writeHead(307, { Location: redirectUrl.toString() });
      res.end();
    });
  },
};

const modelLibraryFiles: Record<string, string> = {
  'occt-import-js/occt-import-js-worker.js': 'occt-import-js/dist/occt-import-js-worker.js',
  'occt-import-js/occt-import-js.js': 'occt-import-js/dist/occt-import-js.js',
  'occt-import-js/occt-import-js.wasm': 'occt-import-js/dist/occt-import-js.wasm',
  'draco3d/draco_decoder_nodejs.min.js': 'draco3d/draco_decoder_nodejs.js',
  'draco3d/draco_decoder.wasm': 'draco3d/draco_decoder.wasm',
};
let modelLibraryPrefix = '/assets/vendor/';

const selfHostModelLibraries: Plugin = {
  name: 'self-host-model-libraries',
  configResolved(config) {
    modelLibraryPrefix = `${config.base.replace(/\/?$/, '/')}assets/vendor/`;
  },
  transform(code, id) {
    if (!id.split('?')[0].endsWith('/online-3d-viewer/build/engine/o3dv.module.js')) return;
    const modelLibraryUrls: Record<string, string> = {
      'https://cdn.jsdelivr.net/npm/occt-import-js@0.0.22/dist/': `${modelLibraryPrefix}occt-import-js/`,
      'https://cdn.jsdelivr.net/npm/draco3d@1.5.7/draco_decoder_nodejs.min.js': `${modelLibraryPrefix}draco3d/draco_decoder_nodejs.min.js`,
    };
    for (const [external, local] of Object.entries(modelLibraryUrls)) {
      if (!code.includes(external)) this.error(`Missing expected model library URL: ${external}`);
      if (external.includes('/occt-import-js@')) {
        if (!code.includes(`'${external}'`)) this.error('Unexpected OCCT library URL syntax');
        code = code.replaceAll(`'${external}'`, `new URL('${local}', window.location.origin).href`);
      } else {
        code = code.replaceAll(external, local);
      }
    }
    return { code, map: null };
  },
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const pathname = new URL(req.url || '/', 'http://vite.local').pathname;
      const prefix = new URL(modelLibraryPrefix, 'http://vite.local').pathname;
      const name = pathname.startsWith(prefix) ? pathname.slice(prefix.length) : '';
      const file = Object.hasOwn(modelLibraryFiles, name) ? modelLibraryFiles[name] : undefined;
      if (!file) return next();
      res.setHeader(
        'Content-Type',
        file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
      );
      fs.createReadStream(resolve(__dirname, 'node_modules', file))
        .on('error', next)
        .pipe(res);
    });
  },
  generateBundle(_options, bundle) {
    for (const output of Object.values(bundle)) {
      if (
        output.type === 'chunk' &&
        /cdn\.jsdelivr\.net\/npm\/(?:occt-import-js|draco3d)/.test(output.code)
      ) {
        this.error(`External model library URL found in ${output.fileName}`);
      }
    }
    for (const [name, file] of Object.entries(modelLibraryFiles)) {
      this.emitFile({
        type: 'asset',
        fileName: `assets/vendor/${name}`,
        source: fs.readFileSync(resolve(__dirname, 'node_modules', file)),
      });
    }
  },
};

export default defineConfig({
  plugins: [react(), localPreviewOriginPlugin, selfHostModelLibraries],
  optimizeDeps: { exclude: ['online-3d-viewer'] },
  base: '/',
  appType: 'spa',
  cacheDir: resolve(__dirname, '.vite'),
  test: {
    environment: 'jsdom',
    maxWorkers: 4,
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
    allowedHosts: ['app.lvh.me'],
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
            proxy.on('proxyReq', (proxyReq, req) => {
              if (hasHost(req.headers.host, 'app.lvh.me', apiPort)) {
                proxyReq.setHeader('host', `app.lvh.me:${backendPort}`);
              }
            });
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
