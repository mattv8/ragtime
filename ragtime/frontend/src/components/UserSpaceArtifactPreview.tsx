import { useMemo } from 'react';
import ts from 'typescript';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
  previewInstanceKey?: string;
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
  const staticImportPattern = /(?:import|export)\s+(?:[^'"`]*?\s+from\s+)?['"]([^'"]+)['"]/g;
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
    /(\bimport\s+['"])([^'"]+)(['"])/g,
    /(\bfrom\s+['"])([^'"]+)(['"])/g,
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
  if (typeof window === 'undefined') return {};
  const styles = window.getComputedStyle(document.documentElement);
  const tokens: ThemeTokens = {};
  for (const tokenName of THEME_TOKEN_NAMES) {
    const value = styles.getPropertyValue(tokenName).trim();
    if (value) tokens[tokenName] = value;
  }
  return tokens;
}

function buildIframeDoc(
  entryPath: string,
  transpiledModules: ModuleMap,
  themeTokens: ThemeTokens
): string {
  const payload = encodeURIComponent(JSON.stringify({
    entryPath,
    modules: transpiledModules,
    themeTokens,
  }));
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src data:; connect-src 'none';" />
    <style>
      :root { font-family: var(--font-sans, system-ui, sans-serif); color-scheme: dark; }
      *, *::before, *::after { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; min-height: 100%; background: var(--color-bg-primary, #0f172a); color: var(--color-text-primary, #f1f5f9); font-family: var(--font-sans, system-ui, sans-serif); }
      #app { min-height: 100%; padding: var(--space-md, 16px); }
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
    </style>
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

      let entryPath = '';
      let modules = {};
      let themeTokens = {};
      try {
        const encodedPayload = (container && container.getAttribute('data-payload')) || '{}';
        const payload = JSON.parse(decodeURIComponent(encodedPayload));
        entryPath = payload.entryPath || '';
        modules = payload.modules || {};
        themeTokens = payload.themeTokens || {};
      } catch (error) {
        renderFatal(error instanceof Error ? error.stack || error.message : String(error));
      }
      const rootStyle = document.documentElement.style;
      Object.keys(themeTokens).forEach((name) => {
        const value = themeTokens[name];
        rootStyle.setProperty(name, value);
      });

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

      const rewriteLocalSpecifiers = (source, importerPath, moduleUrlMap) => {
        const patterns = [
          /(\bimport\s+['"])([^'"]+)(['"])/g,
          /(\bfrom\s+['"])([^'"]+)(['"])/g,
          /(\bimport\(\s*['"])([^'"]+)(['"]\s*\))/g,
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
          const staticRe = /(?:import|export)\s+(?:[^'"\`]*?\s+from\s+)?['"]([^'"]+)['"]/g;
          const dynamicRe = /import\\(\s*['"]([^'"]+)['"]\s*\\)/g;
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
  previewInstanceKey,
}: UserSpaceArtifactPreviewProps) {
  const themeTokens = readThemeTokens();

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
    <div className="userspace-preview-card userspace-preview-frame-wrap">
      <iframe
        key={`${previewInstanceKey ?? ''}:${transpileResult.entryPath}`}
        title="TypeScript module preview"
        className="userspace-preview-frame"
        sandbox="allow-scripts"
        srcDoc={buildIframeDoc(transpileResult.entryPath, transpileResult.modules, themeTokens)}
      />
    </div>
  );
}
