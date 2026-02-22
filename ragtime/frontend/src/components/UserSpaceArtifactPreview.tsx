import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ts from 'typescript';
import type { UserSpaceLiveDataConnection } from '@/types';
import { api } from '@/api/client';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
  liveDataConnections?: UserSpaceLiveDataConnection[];
  previewInstanceKey?: string;
  workspaceId?: string;
  onExecutionStateChange?: (isExecuting: boolean) => void;
}

const THEME_TOKEN_NAMES = [
  // Base colors
  '--color-bg-primary',
  '--color-bg-secondary',
  '--color-bg-tertiary',
  // Surface colors
  '--color-surface',
  '--color-surface-hover',
  '--color-surface-active',
  // Text colors
  '--color-text-primary',
  '--color-text-secondary',
  '--color-text-muted',
  '--color-text-inverse',
  // Brand / accent
  '--color-primary',
  '--color-primary-hover',
  '--color-primary-light',
  '--color-primary-border',
  '--color-accent',
  '--color-accent-hover',
  '--color-accent-light',
  // Semantic colors
  '--color-success',
  '--color-success-light',
  '--color-success-border',
  '--color-error',
  '--color-error-light',
  '--color-error-border',
  '--color-warning',
  '--color-warning-light',
  '--color-warning-border',
  '--color-info',
  '--color-info-light',
  '--color-info-border',
  // Borders
  '--color-border',
  '--color-border-strong',
  // Input
  '--color-input-bg',
  '--color-input-border',
  '--color-input-focus',
  // Shadows
  '--shadow-sm',
  '--shadow-md',
  '--shadow-lg',
  '--shadow-xl',
  // Spacing
  '--space-xs',
  '--space-sm',
  '--space-md',
  '--space-lg',
  '--space-xl',
  '--space-2xl',
  // Border radius
  '--radius-sm',
  '--radius-md',
  '--radius-lg',
  '--radius-xl',
  '--radius-full',
  // Typography
  '--font-sans',
  '--font-mono',
  '--text-xs',
  '--text-sm',
  '--text-base',
  '--text-lg',
  '--text-xl',
  '--text-2xl',
  '--leading-tight',
  '--leading-normal',
  '--leading-relaxed',
  // Transitions
  '--transition-fast',
  '--transition-normal',
  '--transition-slow',
] as const;

type ThemeTokens = Record<string, string>;
type ModuleMap = Record<string, string>;

const DEFAULT_THEME_TOKENS: ThemeTokens = {
  '--color-bg-primary': '#0f172a',
  '--color-bg-secondary': '#111827',
  '--color-bg-tertiary': '#1f2937',
  '--color-surface': '#1e293b',
  '--color-surface-hover': '#334155',
  '--color-surface-active': '#1f2937',
  '--color-text-primary': '#f8fafc',
  '--color-text-secondary': '#cbd5e1',
  '--color-text-muted': '#94a3b8',
  '--color-border': '#334155',
  '--shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.35)',
  '--space-xs': '4px',
  '--space-sm': '8px',
  '--space-md': '12px',
  '--space-lg': '16px',
  '--space-xl': '24px',
  '--radius-sm': '4px',
  '--radius-md': '8px',
  '--radius-lg': '12px',
  '--font-sans': 'Inter, system-ui, -apple-system, Segoe UI, sans-serif',
  '--font-mono': 'ui-monospace, SFMono-Regular, Menlo, monospace',
  '--text-xs': '12px',
  '--text-sm': '14px',
  '--text-base': '16px',
  '--text-lg': '18px',
  '--text-xl': '20px',
  '--text-2xl': '30px',
  '--transition-fast': '150ms ease',
} as const;

const SUPPORTED_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx'] as const;

function normalizePath(input: string): string {
  const trimmed = (input || '').trim();
  if (!trimmed) return '';

  const withoutBackslashes = trimmed.replace(/\\/g, '/');
  const withoutLeadingSlash = withoutBackslashes.replace(/^\//, '');
  const segments = withoutLeadingSlash.split('/');
  const normalized: string[] = [];

  for (const segment of segments) {
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      if (normalized.length > 0) {
        normalized.pop();
      }
      continue;
    }
    normalized.push(segment);
  }

  return normalized.join('/');
}

function dirname(path: string): string {
  const normalized = normalizePath(path);
  const index = normalized.lastIndexOf('/');
  return index === -1 ? '' : normalized.slice(0, index);
}

function hasSupportedExtension(path: string): boolean {
  return SUPPORTED_EXTENSIONS.some((ext) => path.endsWith(ext));
}

function resolveImportTarget(importerPath: string, specifier: string): string {
  const trimmedSpecifier = (specifier || '').trim();
  if (!trimmedSpecifier) return '';

  if (trimmedSpecifier.startsWith('/')) {
    return normalizePath(trimmedSpecifier);
  }

  const importerDir = dirname(importerPath);
  return normalizePath(`${importerDir}/${trimmedSpecifier}`);
}

function resolveWorkspaceModulePath(
  importerPath: string,
  specifier: string,
  fileMap: ModuleMap
): string | null {
  if (!(specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/'))) {
    return null;
  }

  const basePath = resolveImportTarget(importerPath, specifier);
  if (!basePath) return null;

  const candidates: string[] = [basePath];
  if (!hasSupportedExtension(basePath)) {
    for (const extension of SUPPORTED_EXTENSIONS) {
      candidates.push(`${basePath}${extension}`);
    }
    for (const extension of SUPPORTED_EXTENSIONS) {
      candidates.push(`${basePath}/index${extension}`);
    }
  } else if (basePath.endsWith('.js') || basePath.endsWith('.jsx')) {
    const withoutExtension = basePath.replace(/\.(js|jsx)$/i, '');
    candidates.push(`${withoutExtension}.ts`);
    candidates.push(`${withoutExtension}.tsx`);
  } else if (basePath.endsWith('.ts') || basePath.endsWith('.tsx')) {
    const withoutExtension = basePath.replace(/\.(ts|tsx)$/i, '');
    candidates.push(`${withoutExtension}.js`);
    candidates.push(`${withoutExtension}.jsx`);
  }

  for (const candidate of candidates) {
    if (Object.prototype.hasOwnProperty.call(fileMap, candidate)) {
      return candidate;
    }
  }

  return null;
}

function collectLocalSpecifiers(source: string): string[] {
  const values: string[] = [];
  const staticImportPattern = /(?:import|export)\s+(?:[^'"`]*?\s*from\s*)?['"]([^'"]+)['"]/g;
  const dynamicImportPattern = /import\(\s*['"]([^'"]+)['"]\s*\)/g;

  for (const pattern of [staticImportPattern, dynamicImportPattern]) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(source)) !== null) {
      const specifier = match[1];
      if (specifier) values.push(specifier);
    }
  }

  return values;
}

function rewriteLocalSpecifiers(
  source: string,
  importerPath: string,
  moduleUrlMap: Record<string, string>,
  availableModules: ModuleMap
): { output: string; errors: string[] } {
  const errors: string[] = [];

  const rewrite = (specifier: string): string => {
    const resolved = resolveWorkspaceModulePath(importerPath, specifier, availableModules);
    if (!resolved) {
      if (specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/')) {
        errors.push(`${importerPath}: unresolved local import '${specifier}'.`);
      }
      return specifier;
    }
    const rewritten = moduleUrlMap[resolved];
    if (!rewritten) {
      errors.push(`${importerPath}: resolved import '${specifier}' to '${resolved}' but module URL is missing.`);
      return specifier;
    }
    return rewritten;
  };

  const patterns: RegExp[] = [
    /(\bimport\s*['"])([^'"]+)(['"])/g,
    /(\bfrom\s*['"])([^'"]+)(['"])/g,
    /(\bimport\(\s*['"])([^'"]+)(['"]\s*\))/g,
  ];

  let output = source;
  for (const pattern of patterns) {
    output = output.replace(pattern, (_full, prefix: string, specifier: string, suffix: string) => {
      return `${prefix}${rewrite(specifier)}${suffix}`;
    });
  }

  return { output, errors };
}

function readThemeTokens(): ThemeTokens {
  if (typeof window === 'undefined') return { ...DEFAULT_THEME_TOKENS };
  const rootStyles = window.getComputedStyle(document.documentElement);
  const bodyStyles = document.body ? window.getComputedStyle(document.body) : null;
  const tokens: ThemeTokens = { ...DEFAULT_THEME_TOKENS };
  for (const tokenName of THEME_TOKEN_NAMES) {
    const rootValue = rootStyles.getPropertyValue(tokenName).trim();
    const bodyValue = bodyStyles?.getPropertyValue(tokenName).trim() || '';
    const value = rootValue || bodyValue || tokens[tokenName];
    if (value) tokens[tokenName] = value;
  }
  return tokens;
}

function buildIframeDoc(
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
                    bridge: 'userspace-exec-v1',
                    type: 'ragtime-execute-error',
                    callId,
                    component_id: componentId,
                    error: timeoutMessage,
                  }, '*');
                  resolve({ rows: [], columns: [], row_count: 0, error: timeoutMessage });
                }, 60000);
                const handler = (event) => {
                  if (event.source !== window.parent) return;
                  if (event.data && event.data.bridge === 'userspace-exec-v1' && event.data.type === 'ragtime-execute-result' && event.data.callId === callId) {
                    window.removeEventListener('message', handler);
                    clearTimeout(timeout);
                    const result = event.data.result || { rows: [], columns: [], row_count: 0, error: 'Empty response' };
                    const resultError = typeof result.error === 'string' ? result.error.trim() : '';
                    if (resultError) {
                      reportRuntimeError('Live data execution failed (' + componentId + '): ' + resultError);
                      window.parent.postMessage({
                        bridge: 'userspace-exec-v1',
                        type: 'ragtime-execute-error',
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
                  bridge: 'userspace-exec-v1',
                  type: 'ragtime-execute',
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
              // Ensure parent constrains canvas so Chart.js responsive mode works
              const parent = canvas.parentElement;
              if (parent) {
                if (!parent.style.position || parent.style.position === 'static') {
                  parent.style.position = 'relative';
                }
                parent.style.width = parent.style.width || '100%';
                // Guarantee a readable minimum height for the chart area
                parent.style.minHeight = parent.style.minHeight || '300px';
                // If parent has no explicit height, let it grow to fill available space
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
              // Compact layout padding
              nextConfig.options.layout = nextConfig.options.layout || {};
              nextConfig.options.layout.padding = nextConfig.options.layout.padding ?? { top: 4, right: 8, bottom: 4, left: 4 };

              // Force theme-aware text/grid colors on every chart instance
              const cStyles = getComputedStyle(document.documentElement);
              const tColor = cStyles.getPropertyValue('--color-text-secondary').trim() || '#9ca3af';
              const gColor = cStyles.getPropertyValue('--color-border').trim() || '#374151';

              // Enforce scale tick + grid colors per named axis
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

              // Enforce legend + title plugin colors
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

              // Global font color fallback
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
        const staticImportPattern = /(?:import|export)\\s+(?:[^'"]*?\\s*from\\s*)?['"]([^'"]+)['"]/g;
        const dynamicImportPattern = /import\\(\\s*['"]([^'"]+)['"]\\s*\\)/g;

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
          /(\\bimport\\s*['"])([^'"]+)(['"])/g,
          /(\\bfrom\\s*['"])([^'"]+)(['"])/g,
          /(\\bimport\\(\\s*['"])([^'"]+)(['"]\\s*\\))/g,
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

        // Extract local dependency specifiers from module source text
        const extractLocalDeps = (source, importerPath) => {
          const deps = [];
          const staticRe = /(?:import|export)\\s+(?:[^'"]*?\\s*from\\s*)?['"]([^'"]+)['"]/g;
          const dynamicRe = /import\\(\\s*['"]([^'"]+)['"]\\s*\\)/g;
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

        // Build dependency graph
        const depMap = {};
        for (const modulePath of Object.keys(modules)) {
          depMap[modulePath] = extractLocalDeps(modules[modulePath], modulePath);
        }

        // Topological sort via DFS post-order (leaves first, entry last)
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

        // Process modules in topological order: each module's dependencies
        // already have real blob URLs when it is processed.
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

export function UserSpaceArtifactPreview({
  entryPath,
  workspaceFiles,
  liveDataConnections = [],
  previewInstanceKey,
  workspaceId,
  onExecutionStateChange,
}: UserSpaceArtifactPreviewProps) {
  const themeTokens = readThemeTokens();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [pendingExecutions, setPendingExecutions] = useState(0);

  const normalizeExecuteResult = useCallback((result: any) => {
    if (!result || typeof result !== 'object') {
      return { rows: [], columns: [], row_count: 0, error: 'Invalid execution response' };
    }

    const columns: string[] = Array.isArray(result.columns)
      ? result.columns.map((value: unknown) => String(value))
      : [];
    const rows = Array.isArray(result.rows) ? result.rows : [];

    const normalizedRows = rows.map((row: any) => {
      if (!row || typeof row !== 'object' || Array.isArray(row)) {
        return row;
      }

      const enrichedRow: Record<string, unknown> = { ...row };
      columns.forEach((columnName, index) => {
        if (!(index in enrichedRow)) {
          enrichedRow[index] = row[columnName] ?? null;
        }
      });
      return enrichedRow;
    });

    return {
      ...result,
      columns,
      rows: normalizedRows,
      row_count: typeof result.row_count === 'number' ? result.row_count : normalizedRows.length,
    };
  }, []);

  const handleIframeMessage = useCallback(
    async (event: MessageEvent) => {
      const frameWindow = iframeRef.current?.contentWindow;
      if (!frameWindow || event.source !== frameWindow) return;

      const isExpectedOrigin = event.origin === 'null' || event.origin === window.location.origin;
      if (!isExpectedOrigin) return;

      if (!event.data || event.data.bridge !== 'userspace-exec-v1') return;

      if (event.data.type === 'ragtime-execute-error') {
        const componentId = typeof event.data.component_id === 'string' ? event.data.component_id : 'unknown';
        const error = typeof event.data.error === 'string' ? event.data.error : 'Unknown execution error';
        const surfacedError = `Live data connection failed (${componentId}): ${error}`;
        setExecutionError(surfacedError);
        console.error('[UserSpacePreview] iframe execute error:', {
          component_id: componentId,
          error,
        });
        return;
      }

      if (event.data.type !== 'ragtime-execute') return;

      const { callId, component_id, request } = event.data;
      if (typeof callId !== 'string' || typeof component_id !== 'string') return;

      setPendingExecutions((current) => current + 1);
      setExecutionError(null);

      if (!workspaceId) {
        const errorMessage = 'No workspace context available';
        setExecutionError(errorMessage);
        console.error('[UserSpacePreview] execute-component failed:', errorMessage);
        frameWindow.postMessage(
          {
            bridge: 'userspace-exec-v1',
            type: 'ragtime-execute-result',
            callId,
            result: { rows: [], columns: [], row_count: 0, error: errorMessage },
          },
          '*'
        );
        setPendingExecutions((current) => Math.max(0, current - 1));
        return;
      }

      try {
        const result = await api.executeWorkspaceComponent(workspaceId, {
          component_id,
          request,
        });
        const normalizedResult = normalizeExecuteResult(result);
        const normalizedError = typeof normalizedResult.error === 'string' ? normalizedResult.error.trim() : '';
        if (normalizedError) {
          const surfacedError = `Live data connection failed (${component_id}): ${normalizedError}`;
          setExecutionError(surfacedError);
          console.error('[UserSpacePreview] execute-component returned error:', {
            component_id,
            error: normalizedError,
            request,
          });
        } else {
          setExecutionError(null);
        }
        frameWindow.postMessage(
          { bridge: 'userspace-exec-v1', type: 'ragtime-execute-result', callId, result: normalizedResult },
          '*'
        );
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        const surfacedError = `Live data connection failed (${component_id}): ${errorMessage}`;
        setExecutionError(surfacedError);
        console.error('[UserSpacePreview] execute-component request failed:', {
          component_id,
          error: errorMessage,
          request,
        });
        frameWindow.postMessage(
          {
            bridge: 'userspace-exec-v1',
            type: 'ragtime-execute-result',
            callId,
            result: {
              rows: [],
              columns: [],
              row_count: 0,
              error: errorMessage,
            },
          },
          '*'
        );
      } finally {
        setPendingExecutions((current) => Math.max(0, current - 1));
      }
    },
    [workspaceId, normalizeExecuteResult]
  );

  useEffect(() => {
    window.addEventListener('message', handleIframeMessage);
    return () => window.removeEventListener('message', handleIframeMessage);
  }, [handleIframeMessage]);

  useEffect(() => {
    setExecutionError(null);
  }, [workspaceId, entryPath, previewInstanceKey]);

  useEffect(() => {
    onExecutionStateChange?.(pendingExecutions > 0);
  }, [pendingExecutions, onExecutionStateChange]);

  useEffect(() => {
    setPendingExecutions(0);
  }, [workspaceId, entryPath, previewInstanceKey]);

  const transpileResult = useMemo(() => {
    const normalizedEntry = normalizePath(entryPath);
    const normalizedFiles: ModuleMap = {};
    for (const [path, source] of Object.entries(workspaceFiles || {})) {
      const normalizedPath = normalizePath(path);
      if (!normalizedPath) continue;
      normalizedFiles[normalizedPath] = source;
    }

    if (!normalizedEntry || !Object.prototype.hasOwnProperty.call(normalizedFiles, normalizedEntry)) {
      return {
        entryPath: normalizedEntry,
        modules: {} as ModuleMap,
        errors: `Entry module '${entryPath}' is missing. Create dashboard/main.ts to render this workspace frontend.`,
      };
    }

    const queue = [normalizedEntry];
    const visited = new Set<string>();
    const transpiledModules: ModuleMap = {};
    const allErrors: string[] = [];

    while (queue.length > 0) {
      const modulePath = queue.shift() as string;
      if (visited.has(modulePath)) continue;
      visited.add(modulePath);

      const source = normalizedFiles[modulePath];
      if (source === undefined) {
        allErrors.push(`Missing module source for '${modulePath}'.`);
        continue;
      }

      const transpiled = ts.transpileModule(source, {
        fileName: modulePath,
        reportDiagnostics: true,
        compilerOptions: {
          module: ts.ModuleKind.ES2020,
          target: ts.ScriptTarget.ES2020,
          isolatedModules: true,
          jsx: ts.JsxEmit.ReactJSX,
        },
      });

      const diagnostics = (transpiled.diagnostics ?? []).filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error);
      if (diagnostics.length > 0) {
        const messages = diagnostics.map((diagnostic) => {
          const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
          if (!diagnostic.file || diagnostic.start === undefined) return message;
          const position = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
          return `${diagnostic.file.fileName}:${position.line + 1}:${position.character + 1} ${message}`;
        });
        allErrors.push(...messages);
      }

      transpiledModules[modulePath] = transpiled.outputText;

      for (const specifier of collectLocalSpecifiers(source)) {
        if (!(specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/'))) {
          continue;
        }
        const resolvedPath = resolveWorkspaceModulePath(modulePath, specifier, normalizedFiles);
        if (!resolvedPath) {
          allErrors.push(`${modulePath}: unresolved local import '${specifier}'.`);
          continue;
        }
        if (!visited.has(resolvedPath)) {
          queue.push(resolvedPath);
        }
      }
    }

    const placeholderModuleUrls = Object.fromEntries(
      Object.keys(transpiledModules).map((modulePath) => [modulePath, `blob://placeholder/${encodeURIComponent(modulePath)}`])
    ) as Record<string, string>;

    for (const [modulePath, transpiledSource] of Object.entries(transpiledModules)) {
      const rewritten = rewriteLocalSpecifiers(transpiledSource, modulePath, placeholderModuleUrls, transpiledModules);
      if (rewritten.errors.length > 0) {
        allErrors.push(...rewritten.errors);
      }
    }

    if (allErrors.length > 0) {
      const uniqueErrors = Array.from(new Set(allErrors));
      return {
        entryPath: normalizedEntry,
        modules: {} as ModuleMap,
        errors: uniqueErrors.join('\n'),
      };
    }

    return {
      entryPath: normalizedEntry,
      modules: transpiledModules,
      errors: null as string | null,
    };
  }, [entryPath, workspaceFiles]);

  if (transpileResult.errors) {
    return (
      <div className="userspace-preview-card">
        <h4>TypeScript module preview</h4>
        <p>Fix TypeScript or module import errors to render this workspace app in the isolated runtime.</p>
        <pre>{transpileResult.errors}</pre>
      </div>
    );
  }

  return (
    <div className="userspace-preview-frame-wrap">
      {executionError ? (
        <div className="status-message error userspace-preview-exec-error" role="alert">
          {executionError}
        </div>
      ) : null}
      <iframe
        ref={iframeRef}
        key={`${previewInstanceKey ?? ''}:${transpileResult.entryPath}`}
        title="TypeScript module preview"
        className="userspace-preview-frame"
        sandbox="allow-scripts"
        srcDoc={buildIframeDoc(transpileResult.entryPath, transpileResult.modules, themeTokens, liveDataConnections)}
      />
    </div>
  );
}
