import { useMemo } from 'react';
import ts from 'typescript';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
}

const THEME_TOKEN_NAMES = [
  '--color-bg-primary',
  '--color-bg-secondary',
  '--color-bg-tertiary',
  '--color-surface',
  '--color-surface-hover',
  '--color-surface-active',
  '--color-text-primary',
  '--color-text-secondary',
  '--color-text-muted',
  '--color-border',
  '--color-border-strong',
  '--color-primary',
  '--color-primary-hover',
  '--color-primary-light',
  '--color-primary-border',
  '--color-success',
  '--color-error',
  '--color-warning',
  '--space-xs',
  '--space-sm',
  '--space-md',
  '--space-lg',
  '--radius-sm',
  '--radius-md',
  '--radius-lg',
  '--font-sans',
  '--font-mono',
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

  const staticImportPattern = /((?:import|export)\s+(?:[^'"`]*?\s+from\s+)?['"])([^'"]+)(['"])/g;
  const dynamicImportPattern = /(import\(\s*['"])([^'"]+)(['"]\s*\))/g;

  let output = source.replace(staticImportPattern, (_full, prefix: string, specifier: string, suffix: string) => {
    return `${prefix}${rewrite(specifier)}${suffix}`;
  });

  output = output.replace(dynamicImportPattern, (_full, prefix: string, specifier: string, suffix: string) => {
    return `${prefix}${rewrite(specifier)}${suffix}`;
  });

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
  const encodedEntryPath = JSON.stringify(entryPath);
  const encodedModules = JSON.stringify(transpiledModules);
  const encodedThemeTokens = JSON.stringify(themeTokens);
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src data:; connect-src 'none';" />
    <style>
      :root { font-family: var(--font-sans, system-ui, sans-serif); }
      *, *::before, *::after { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; min-height: 100%; background: var(--color-bg-secondary, transparent); color: var(--color-text-primary, #e5e7eb); font-family: var(--font-sans, system-ui, sans-serif); }
      #app { min-height: 100%; padding: var(--space-md, 16px); }
      .userspace-render-error {
        margin: var(--space-md, 12px);
        padding: var(--space-md, 12px);
        border: 1px solid var(--color-error, #dc2626);
        border-radius: var(--radius-md, 8px);
        color: var(--color-text-primary, #fca5a5);
        background: var(--color-bg-tertiary, #2b1012);
        white-space: pre-wrap;
      }
    </style>
  </head>
  <body>
    <div id="app"></div>
    <script type="module">
      const entryPath = ${encodedEntryPath};
      const modules = ${encodedModules};
      const themeTokens = ${encodedThemeTokens};
      const container = document.getElementById('app');
      const rootStyle = document.documentElement.style;
      Object.entries(themeTokens).forEach(([name, value]) => rootStyle.setProperty(name, value));

      const extensions = ['.ts', '.tsx', '.js', '.jsx'];

      const normalizePath = (input) => {
        if (!input) return '';
        const withoutBackslashes = String(input).replace(/\\\\/g, '/');
        const withoutLeadingSlash = withoutBackslashes.replace(/^\//, '');
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
        const staticImportPattern = /((?:import|export)\s+(?:[^'"]*?\s+from\s+)?['"])([^'"]+)(['"])/g;
        const dynamicImportPattern = /(import\(\s*['"])([^'"]+)(['"]\s*\))/g;

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

        let output = source.replace(staticImportPattern, (_full, prefix, specifier, suffix) => {
          return prefix + rewriteSpecifier(specifier) + suffix;
        });

        output = output.replace(dynamicImportPattern, (_full, prefix, specifier, suffix) => {
          return prefix + rewriteSpecifier(specifier) + suffix;
        });

        return output;
      };

      const showError = (message) => {
        const el = document.createElement('div');
        el.className = 'userspace-render-error';
        el.textContent = message;
        container.innerHTML = '';
        container.appendChild(el);
      };

      try {
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
      } catch (error) {
        showError(error instanceof Error ? error.stack || error.message : String(error));
      }
    </script>
  </body>
</html>`;
}

export function UserSpaceArtifactPreview({
  entryPath,
  workspaceFiles,
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
        title="TypeScript module preview"
        className="userspace-preview-frame"
        sandbox="allow-scripts"
        srcDoc={buildIframeDoc(transpileResult.entryPath, transpileResult.modules, themeTokens)}
      />
    </div>
  );
}
