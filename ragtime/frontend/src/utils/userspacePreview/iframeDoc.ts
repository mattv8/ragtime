import type { UserSpaceLiveDataConnection } from '@/types';
import {
  USERSPACE_EXEC_BRIDGE,
  USERSPACE_EXEC_MESSAGE_TYPES,
  USERSPACE_EXECUTE_TIMEOUT_MS,
} from './constants';
import type { ModuleMap } from './moduleResolver';
import type { ThemeTokens } from './themeTokens';

export function buildIframeDoc(
  entryPath: string,
  transpiledModules: ModuleMap,
  themeTokens: ThemeTokens,
  liveDataConnections: UserSpaceLiveDataConnection[]
): string {
  const payload = encodeURIComponent(JSON.stringify({
    entryPath,
    modules: transpiledModules,
    themeTokens,
    liveDataConnections,
  }));
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob: https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data:; connect-src 'none';" />
    <style>
      :root { font-family: var(--font-sans, system-ui, sans-serif); color-scheme: dark; }
      *, *::before, *::after { box-sizing: border-box; }
      ::-webkit-scrollbar { width: 8px; height: 8px; background: transparent; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: var(--color-border, #334155); border-radius: var(--radius-full, 9999px); }
      ::-webkit-scrollbar-thumb:hover { background: var(--color-text-muted, #94a3b8); }
      ::-webkit-scrollbar-corner { background: transparent; }
      * { scrollbar-width: thin; scrollbar-color: var(--color-border, #334155) transparent; }
      html::-webkit-scrollbar, body::-webkit-scrollbar { width: 8px; height: 8px; background: transparent; }
      html::-webkit-scrollbar-track, body::-webkit-scrollbar-track { background: transparent; }
      html::-webkit-scrollbar-thumb, body::-webkit-scrollbar-thumb { background: var(--color-border, #334155); border-radius: var(--radius-full, 9999px); }
      html::-webkit-scrollbar-thumb:hover, body::-webkit-scrollbar-thumb:hover { background: var(--color-text-muted, #94a3b8); }
      html::-webkit-scrollbar-corner, body::-webkit-scrollbar-corner { background: transparent; }
      html, body { scrollbar-width: thin; scrollbar-color: var(--color-border, #334155) transparent; }
      html { height: 100%; }
      body { margin: 0; padding: 0; height: 100%; background: var(--color-bg-primary, #0f172a); color: var(--color-text-primary, #f1f5f9); font-family: var(--font-sans, system-ui, sans-serif); }
      #app { height: 100%; display: flex; flex-direction: column; }
      canvas { display: block; max-width: 100%; max-height: 100%; }
      a { color: var(--color-primary, #6366f1); }
      a:hover { color: var(--color-primary-hover, #4f46e5); }
      .userspace-render-error {
        margin: var(--space-md, 16px);
        padding: var(--space-md, 16px);
        border: 1px solid var(--color-error, #ef4444);
        border-radius: var(--radius-md, 8px);
        color: var(--color-text-primary, #f1f5f9);
        background: var(--color-error-light, rgba(239, 68, 68, 0.15));
        white-space: pre-wrap;
      }
      .userspace-runtime-error {
        margin: var(--space-sm, 8px);
        padding: var(--space-sm, 8px);
        border: 1px solid var(--color-error-border, #ef4444);
        border-radius: var(--radius-sm, 6px);
        background: var(--color-error-light, rgba(239, 68, 68, 0.15));
        color: var(--color-error, #ef4444);
        font-size: var(--text-sm, 14px);
        white-space: pre-wrap;
      }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  </head>
  <body>
    <div id="app" data-payload="${payload}"></div>
    <script type="module">
      const bridge = ${JSON.stringify(USERSPACE_EXEC_BRIDGE)};
      const messageTypes = ${JSON.stringify(USERSPACE_EXEC_MESSAGE_TYPES)};
      const executeTimeoutMs = ${USERSPACE_EXECUTE_TIMEOUT_MS};

      const container = document.getElementById('app');
      const renderFatal = (message) => {
        const mount = container || document.body;
        if (!mount) return;
        const el = document.createElement('div');
        el.className = 'userspace-render-error';
        el.textContent = message;
        mount.innerHTML = '';
        mount.appendChild(el);
      };

      const reportRuntimeError = (message) => {
        const mount = container || document.body;
        if (!mount) return;
        let panel = mount.querySelector('[data-userspace-runtime-error]');
        if (!panel) {
          panel = document.createElement('div');
          panel.className = 'userspace-runtime-error';
          panel.setAttribute('data-userspace-runtime-error', '1');
          mount.insertBefore(panel, mount.firstChild);
        }
        panel.textContent = message;
      };

      let entryPath = '';
      let modules = {};
      let themeTokens = {};
      let liveDataConnections = [];
      try {
        const encodedPayload = (container && container.getAttribute('data-payload')) || '{}';
        const payload = JSON.parse(decodeURIComponent(encodedPayload));
        entryPath = payload.entryPath || '';
        modules = payload.modules || {};
        themeTokens = payload.themeTokens || {};
        liveDataConnections = Array.isArray(payload.liveDataConnections) ? payload.liveDataConnections : [];
      } catch (error) {
        renderFatal(error instanceof Error ? error.stack || error.message : String(error));
      }

      const buildRuntimeComponents = (connections) => {
        const toAliasKey = (value) => String(value || '')
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '_')
          .replace(/^_+|_+$/g, '');

        const createMissingComponent = (requestedKey) => Object.freeze({
          component_id: null,
          component_kind: 'tool_config',
          component_name: null,
          component_type: null,
          request: {},
          refresh_interval_seconds: null,
          async execute() {
            const message = connections.length === 0
              ? 'No live data connections are configured for this workspace. Select tools in User Space settings and regenerate the module.'
              : 'Live data component "' + requestedKey + '" is not available. Use context.componentsList to inspect available components.';
            reportRuntimeError(message);
            return { rows: [], columns: [], row_count: 0, error: message };
          },
        });

        const byKey = {};
        const list = [];
        for (let index = 0; index < connections.length; index += 1) {
          const connection = connections[index] || {};
          const componentId = (typeof connection.component_id === 'string' && connection.component_id.trim())
            ? connection.component_id.trim()
            : 'component_' + index;

          const component = Object.freeze({
            component_id: componentId,
            component_kind: connection.component_kind || 'tool_config',
            component_name: connection.component_name || null,
            component_type: connection.component_type || null,
            request: connection.request ?? {},
            refresh_interval_seconds: connection.refresh_interval_seconds ?? null,
            async execute(requestOverride) {
              const effectiveRequest = requestOverride ?? connection.request ?? {};
              const callId = '__exec_' + Math.random().toString(36).slice(2) + '_' + Date.now();
              return new Promise((resolve) => {
                const timeout = setTimeout(() => {
                  window.removeEventListener('message', handler);
                  const timeoutMessage = 'Execute request timed out after 60s';
                  reportRuntimeError('Live data execution failed (' + componentId + '): ' + timeoutMessage);
                  window.parent.postMessage({
                    bridge,
                    type: messageTypes.ERROR,
                    callId,
                    component_id: componentId,
                    error: timeoutMessage,
                  }, '*');
                  resolve({ rows: [], columns: [], row_count: 0, error: timeoutMessage });
                }, executeTimeoutMs);
                const handler = (event) => {
                  if (event.source !== window.parent) return;
                  if (event.data && event.data.bridge === bridge && event.data.type === messageTypes.RESULT && event.data.callId === callId) {
                    window.removeEventListener('message', handler);
                    clearTimeout(timeout);
                    const result = event.data.result || { rows: [], columns: [], row_count: 0, error: 'Empty response' };
                    const resultError = typeof result.error === 'string' ? result.error.trim() : '';
                    if (resultError) {
                      reportRuntimeError('Live data execution failed (' + componentId + '): ' + resultError);
                      window.parent.postMessage({
                        bridge,
                        type: messageTypes.ERROR,
                        callId,
                        component_id: componentId,
                        error: resultError,
                      }, '*');
                      console.error('[UserSpacePreviewIframe] execute failed', {
                        component_id: componentId,
                        error: resultError,
                        request: effectiveRequest,
                      });
                    }
                    resolve(result);
                  }
                };
                window.addEventListener('message', handler);
                window.parent.postMessage({
                  bridge,
                  type: messageTypes.EXECUTE,
                  callId: callId,
                  component_id: componentId,
                  request: effectiveRequest,
                }, '*');
              });
            },
          });

          byKey[componentId] = component;
          byKey[index] = component;

          const aliasFromName = toAliasKey(connection.component_name);
          if (aliasFromName && !Object.prototype.hasOwnProperty.call(byKey, aliasFromName)) {
            byKey[aliasFromName] = component;
          }

          list.push(component);
        }

        const componentsProxy = new Proxy(byKey, {
          get(target, prop, receiver) {
            if (Reflect.has(target, prop)) {
              return Reflect.get(target, prop, receiver);
            }
            if (typeof prop === 'string' && list.length === 1) {
              return list[0];
            }
            if (typeof prop === 'string' || typeof prop === 'number') {
              return createMissingComponent(String(prop));
            }
            return undefined;
          },
          has(target, prop) {
            if (Reflect.has(target, prop)) return true;
            return typeof prop === 'string' && list.length === 1;
          },
        });

        return {
          byKey: Object.freeze(componentsProxy),
          list: Object.freeze(list),
        };
      };

      const runtimeComponents = buildRuntimeComponents(liveDataConnections);
      window.__RAGTIME_COMPONENTS__ = runtimeComponents.byKey;
      const rootStyle = document.documentElement.style;
      Object.keys(themeTokens).forEach((name) => {
        const value = themeTokens[name];
        rootStyle.setProperty(name, value);
      });

      const originalHeadAppendChild = document.head.appendChild.bind(document.head);
      document.head.appendChild = (node) => {
        if (
          node &&
          node.tagName === 'SCRIPT' &&
          typeof node.src === 'string' &&
          node.src.toLowerCase().includes('cdn.jsdelivr.net/npm/chart.js') &&
          window.Chart
        ) {
          if (typeof node.onload === 'function') {
            setTimeout(() => {
              node.onload(new Event('load'));
            }, 0);
          }
          return node;
        }

        return originalHeadAppendChild(node);
      };

      const applyChartDefaults = () => {
        const Chart = window.Chart;
        if (!Chart || !Chart.defaults) return;

        if (!window.__RAGTIME_CHART_IFRAME_PATCHED__) {
          const OriginalChart = Chart;
          const PatchedChart = function (item, config) {
            const nextConfig = config && typeof config === 'object'
              ? { ...config, options: { ...(config.options || {}) } }
              : config;

            const canvas = item && item.canvas
              ? item.canvas
              : (item instanceof HTMLCanvasElement ? item : null);

            if (canvas) {
              const parent = canvas.parentElement;
              if (parent) {
                if (!parent.style.position || parent.style.position === 'static') {
                  parent.style.position = 'relative';
                }
                parent.style.width = parent.style.width || '100%';
                parent.style.minHeight = parent.style.minHeight || '300px';
                if (!parent.style.height && !parent.style.flex) {
                  parent.style.flex = '1';
                }
              }
            }

            if (nextConfig && nextConfig.options) {
              nextConfig.options.responsive = true;
              nextConfig.options.maintainAspectRatio = true;
              nextConfig.options.aspectRatio = nextConfig.options.aspectRatio || 1.8;
              nextConfig.options.resizeDelay = 300;
              nextConfig.options.layout = nextConfig.options.layout || {};
              nextConfig.options.layout.padding = nextConfig.options.layout.padding ?? { top: 4, right: 8, bottom: 4, left: 4 };

              const cStyles = getComputedStyle(document.documentElement);
              const tColor = cStyles.getPropertyValue('--color-text-secondary').trim() || '#9ca3af';
              const gColor = cStyles.getPropertyValue('--color-border').trim() || '#374151';

              const scales = nextConfig.options.scales;
              if (scales && typeof scales === 'object') {
                for (const axisKey of Object.keys(scales)) {
                  const axis = scales[axisKey];
                  if (!axis || typeof axis !== 'object') continue;
                  axis.ticks = axis.ticks || {};
                  axis.ticks.color = tColor;
                  axis.grid = axis.grid || {};
                  axis.grid.color = gColor;
                  if (axis.title && typeof axis.title === 'object') {
                    axis.title.color = tColor;
                  }
                }
              }

              const plugins = nextConfig.options.plugins = nextConfig.options.plugins || {};
              if (plugins.legend) {
                plugins.legend.labels = plugins.legend.labels || {};
                plugins.legend.labels.color = tColor;
              }
              if (plugins.title) {
                plugins.title.color = tColor;
              }
              if (plugins.subtitle) {
                plugins.subtitle.color = tColor;
              }
              if (plugins.tooltip) {
                plugins.tooltip.titleColor = plugins.tooltip.titleColor || tColor;
                plugins.tooltip.bodyColor = plugins.tooltip.bodyColor || tColor;
              }

              nextConfig.options.color = tColor;
            }

            return new OriginalChart(item, nextConfig);
          };

          Object.assign(PatchedChart, OriginalChart);
          PatchedChart.prototype = OriginalChart.prototype;
          window.Chart = PatchedChart;
          window.__RAGTIME_CHART_IFRAME_PATCHED__ = true;
        }

        const styles = getComputedStyle(document.documentElement);
        const textColor = styles.getPropertyValue('--color-text-secondary').trim() || '#9ca3af';
        const gridColor = styles.getPropertyValue('--color-border').trim() || '#374151';

        Chart.defaults.color = textColor;
        Chart.defaults.borderColor = gridColor;
        Chart.defaults.plugins = Chart.defaults.plugins || {};
        Chart.defaults.plugins.legend = Chart.defaults.plugins.legend || {};
        Chart.defaults.plugins.legend.labels = Chart.defaults.plugins.legend.labels || {};
        Chart.defaults.plugins.legend.labels.color = textColor;
        Chart.defaults.plugins.title = Chart.defaults.plugins.title || {};
        Chart.defaults.plugins.title.color = textColor;

        Chart.defaults.scales = Chart.defaults.scales || {};
        Chart.defaults.scales.linear = Chart.defaults.scales.linear || {};
        Chart.defaults.scales.linear.ticks = Chart.defaults.scales.linear.ticks || {};
        Chart.defaults.scales.linear.ticks.color = textColor;
        Chart.defaults.scales.linear.grid = Chart.defaults.scales.linear.grid || {};
        Chart.defaults.scales.linear.grid.color = gridColor;

        Chart.defaults.scales.category = Chart.defaults.scales.category || {};
        Chart.defaults.scales.category.ticks = Chart.defaults.scales.category.ticks || {};
        Chart.defaults.scales.category.ticks.color = textColor;
        Chart.defaults.scales.category.grid = Chart.defaults.scales.category.grid || {};
        Chart.defaults.scales.category.grid.color = gridColor;

        Chart.defaults.scales.radialLinear = Chart.defaults.scales.radialLinear || {};
        Chart.defaults.scales.radialLinear.ticks = Chart.defaults.scales.radialLinear.ticks || {};
        Chart.defaults.scales.radialLinear.ticks.color = textColor;
        Chart.defaults.scales.radialLinear.grid = Chart.defaults.scales.radialLinear.grid || {};
        Chart.defaults.scales.radialLinear.grid.color = gridColor;
        Chart.defaults.scales.radialLinear.pointLabels = Chart.defaults.scales.radialLinear.pointLabels || {};
        Chart.defaults.scales.radialLinear.pointLabels.color = textColor;
      };

      applyChartDefaults();

      const extensions = ['.ts', '.tsx', '.js', '.jsx'];

      const normalizePath = (input) => {
        if (!input) return '';
        const withoutBackslashes = String(input).split('\\\\').join('/');
        const withoutLeadingSlash = withoutBackslashes.startsWith('/')
          ? withoutBackslashes.slice(1)
          : withoutBackslashes;
        const segments = withoutLeadingSlash.split('/');
        const normalized = [];

        for (const segment of segments) {
          if (!segment || segment === '.') continue;
          if (segment === '..') {
            if (normalized.length > 0) normalized.pop();
            continue;
          }
          normalized.push(segment);
        }

        return normalized.join('/');
      };

      const dirname = (path) => {
        const normalized = normalizePath(path);
        const index = normalized.lastIndexOf('/');
        return index === -1 ? '' : normalized.slice(0, index);
      };

      const hasExtension = (path) => extensions.some((ext) => path.endsWith(ext));

      const resolveImportTarget = (importerPath, specifier) => {
        if (!specifier) return '';
        if (specifier.startsWith('/')) {
          return normalizePath(specifier);
        }
        return normalizePath(dirname(importerPath) + '/' + specifier);
      };

      const resolveWorkspaceModulePath = (importerPath, specifier) => {
        if (!(specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/'))) {
          return null;
        }
        const basePath = resolveImportTarget(importerPath, specifier);
        if (!basePath) return null;

        const candidates = [basePath];
        if (!hasExtension(basePath)) {
          for (const ext of extensions) candidates.push(basePath + ext);
          for (const ext of extensions) candidates.push(basePath + '/index' + ext);
        } else if (basePath.endsWith('.js') || basePath.endsWith('.jsx')) {
          const withoutExtension = basePath.replace(/\.(js|jsx)$/i, '');
          candidates.push(withoutExtension + '.ts');
          candidates.push(withoutExtension + '.tsx');
        } else if (basePath.endsWith('.ts') || basePath.endsWith('.tsx')) {
          const withoutExtension = basePath.replace(/\.(ts|tsx)$/i, '');
          candidates.push(withoutExtension + '.js');
          candidates.push(withoutExtension + '.jsx');
        }

        for (const candidate of candidates) {
          if (Object.prototype.hasOwnProperty.call(modules, candidate)) {
            return candidate;
          }
        }

        return null;
      };

      const collectLocalSpecifiers = (source) => {
        const values = [];
        const staticImportPattern = /(?:import|export)\\s+(?:[^'\"]*?\\s*from\\s*)?['\"]([^'\"]+)['\"]/g;
        const dynamicImportPattern = /import\\(\\s*['\"]([^'\"]+)['\"]\\s*\\)/g;

        for (const pattern of [staticImportPattern, dynamicImportPattern]) {
          pattern.lastIndex = 0;
          let match;
          while ((match = pattern.exec(source)) !== null) {
            const specifier = match[1];
            if (specifier) values.push(specifier);
          }
        }

        return values;
      };

      const rewriteLocalSpecifiers = (source, importerPath, moduleUrlMap) => {
        const patterns = [
          /(\\bimport\\s*['\"])([^'\"]+)(['\"])/g,
          /(\\bfrom\\s*['\"])([^'\"]+)(['\"])/g,
          /(\\bimport\\(\\s*['\"])([^'\"]+)(['\"]\\s*\\))/g,
        ];

        const rewriteSpecifier = (specifier) => {
          const resolved = resolveWorkspaceModulePath(importerPath, specifier);
          if (!resolved) {
            if (specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/')) {
              throw new Error(importerPath + ": unresolved local import '" + specifier + "'.");
            }
            return specifier;
          }
          const moduleUrl = moduleUrlMap[resolved];
          if (!moduleUrl) {
            throw new Error(importerPath + ": missing module URL for '" + resolved + "'.");
          }
          return moduleUrl;
        };

        let output = source;
        for (const pattern of patterns) {
          output = output.replace(pattern, (_full, prefix, specifier, suffix) => {
            return prefix + rewriteSpecifier(specifier) + suffix;
          });
        }

        return output;
      };

      const showError = (message) => {
        renderFatal(message);
      };

      const runPreview = async () => {
        const normalizedEntry = normalizePath(entryPath);
        if (!normalizedEntry || !Object.prototype.hasOwnProperty.call(modules, normalizedEntry)) {
          throw new Error("Missing entry module '" + entryPath + "'.");
        }

        const extractLocalDeps = (source, importerPath) => {
          const deps = [];
          const staticRe = /(?:import|export)\\s+(?:[^'\"]*?\\s*from\\s*)?['\"]([^'\"]+)['\"]/g;
          const dynamicRe = /import\\(\\s*['\"]([^'\"]+)['\"]\\s*\\)/g;
          for (const re of [staticRe, dynamicRe]) {
            re.lastIndex = 0;
            let m;
            while ((m = re.exec(source)) !== null) {
              const spec = m[1];
              if (!spec) continue;
              if (!(spec.startsWith('./') || spec.startsWith('../') || spec.startsWith('/'))) continue;
              const resolved = resolveWorkspaceModulePath(importerPath, spec);
              if (resolved) deps.push(resolved);
            }
          }
          return deps;
        };

        const depMap = {};
        for (const modulePath of Object.keys(modules)) {
          depMap[modulePath] = extractLocalDeps(modules[modulePath], modulePath);
        }

        const visited = new Set();
        const inStack = new Set();
        const sorted = [];
        const visit = (mod) => {
          if (visited.has(mod)) return;
          if (inStack.has(mod)) return;
          inStack.add(mod);
          for (const dep of (depMap[mod] || [])) {
            visit(dep);
          }
          inStack.delete(mod);
          visited.add(mod);
          sorted.push(mod);
        };
        for (const mod of Object.keys(modules)) {
          visit(mod);
        }

        const moduleUrlMap = {};
        const blobUrls = [];
        for (const modulePath of sorted) {
          const rewritten = rewriteLocalSpecifiers(modules[modulePath], modulePath, moduleUrlMap);
          const remainingLocalSpecifiers = collectLocalSpecifiers(rewritten).filter((specifier) => {
            return specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/');
          });
          if (remainingLocalSpecifiers.length > 0) {
            throw new Error(
              modulePath + ": unresolved local module specifiers remained after rewrite: " + remainingLocalSpecifiers.join(', ')
            );
          }
          const blob = new Blob([rewritten], { type: 'text/javascript' });
          const url = URL.createObjectURL(blob);
          moduleUrlMap[modulePath] = url;
          blobUrls.push(url);
        }

        const loaded = await import(moduleUrlMap[normalizedEntry]);

        blobUrls.forEach((url) => URL.revokeObjectURL(url));

        if (typeof loaded.render !== 'function') {
          showError('Module must export render(container, context).');
        } else {
          const context = Object.freeze({
            components: Object.freeze(window.__RAGTIME_COMPONENTS__ ?? {}),
            componentsList: runtimeComponents.list,
            themeTokens: Object.freeze(themeTokens),
          });
          await loaded.render(container, context);
        }
      };

      runPreview().catch((error) => {
        showError(error instanceof Error ? error.stack || error.message : String(error));
      });
    </script>
  </body>
</html>`;
}
