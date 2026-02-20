interface UserSpaceArtifactPreviewProps {
  filePath: string;
  content: string;
}

import { useMemo } from 'react';
import ts from 'typescript';

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

function buildIframeDoc(transpiledSource: string, themeTokens: ThemeTokens): string {
  const encodedSource = JSON.stringify(transpiledSource);
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
      const source = ${encodedSource};
      const themeTokens = ${encodedThemeTokens};
      const container = document.getElementById('app');
      const rootStyle = document.documentElement.style;
      Object.entries(themeTokens).forEach(([name, value]) => rootStyle.setProperty(name, value));

      const showError = (message) => {
        const el = document.createElement('div');
        el.className = 'userspace-render-error';
        el.textContent = message;
        container.innerHTML = '';
        container.appendChild(el);
      };

      try {
        const blob = new Blob([source], { type: 'text/javascript' });
        const moduleUrl = URL.createObjectURL(blob);
        const loaded = await import(moduleUrl);
        URL.revokeObjectURL(moduleUrl);

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
  filePath,
  content,
}: UserSpaceArtifactPreviewProps) {
  const themeTokens = readThemeTokens();

  const transpileResult = useMemo(() => {
    const result = ts.transpileModule(content, {
      fileName: filePath || 'module.ts',
      reportDiagnostics: true,
      compilerOptions: {
        module: ts.ModuleKind.ES2020,
        target: ts.ScriptTarget.ES2020,
        isolatedModules: true,
        jsx: ts.JsxEmit.ReactJSX,
      },
    });

    const diagnostics = (result.diagnostics ?? []).filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error);
    if (diagnostics.length === 0) {
      return { output: result.outputText, errors: null as string | null };
    }

    const errors = diagnostics
      .map((diagnostic) => {
        const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
        if (!diagnostic.file || diagnostic.start === undefined) return message;
        const position = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
        return `${diagnostic.file.fileName}:${position.line + 1}:${position.character + 1} ${message}`;
      })
      .join('\n');

    return { output: '', errors };
  }, [content, filePath]);

  if (transpileResult.errors) {
    return (
      <div className="userspace-preview-card">
        <h4>TypeScript module preview</h4>
        <p>Fix TypeScript errors to render this module in the isolated runtime.</p>
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
        srcDoc={buildIframeDoc(transpileResult.output, themeTokens)}
      />
    </div>
  );
}
