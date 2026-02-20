import { useEffect, useMemo, useState } from 'react';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import type { UserSpaceArtifactType } from '@/types';

interface UserSpaceArtifactPreviewProps {
  filePath: string;
  content: string;
  artifactType?: UserSpaceArtifactType | null;
  canEnableActivePreview?: boolean;
}

function buildHtmlPreviewSrcDoc(rawContent: string, allowScripts: boolean): string {
  const csp = allowScripts
    ? "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; font-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    : "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; font-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'";

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
  </head>
  <body>
${rawContent}
  </body>
</html>`;
}

function escapeModuleScriptContent(source: string): string {
  return source.replace(/<\/(script)/gi, '<\\/$1');
}

function detectArtifactType(filePath: string, artifactType?: UserSpaceArtifactType | null): UserSpaceArtifactType | 'unknown' {
  if (artifactType) return artifactType;
  const lower = filePath.toLowerCase();
  if (lower.endsWith('.json')) return 'dashboard_json';
  if (lower.endsWith('.md') || lower.endsWith('.markdown')) return 'report_markdown';
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'report_html';
  if (lower.endsWith('.ts')) return 'module_ts';
  if (lower.endsWith('.js') || lower.endsWith('.mjs')) return 'module_js';
  return 'unknown';
}

export function UserSpaceArtifactPreview({
  filePath,
  content,
  artifactType,
  canEnableActivePreview = true,
}: UserSpaceArtifactPreviewProps) {
  const resolvedType = detectArtifactType(filePath, artifactType);
  const [allowActivePreview, setAllowActivePreview] = useState(false);

  useEffect(() => {
    setAllowActivePreview(false);
  }, [filePath, resolvedType]);

  useEffect(() => {
    if (!canEnableActivePreview && allowActivePreview) {
      setAllowActivePreview(false);
    }
  }, [allowActivePreview, canEnableActivePreview]);

  const jsModuleSrcDoc = useMemo(
    () => `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; font-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'" />
    <style>
      body { font-family: sans-serif; margin: 12px; }
      pre { white-space: pre-wrap; color: #b91c1c; }
    </style>
  </head>
  <body>
    <div id="app"></div>
    <script type="module">
      try {
${escapeModuleScriptContent(content)}
      } catch (e) {
        document.body.innerHTML = '<pre>' + String(e) + '</pre>';
      }
    </script>
  </body>
</html>`,
    [content]
  );

  if (resolvedType === 'dashboard_json') {
    try {
      const parsed = JSON.parse(content);
      const title = parsed?.title || 'Dashboard';
      const panelCount = Array.isArray(parsed?.panels) ? parsed.panels.length : 0;
      return (
        <div className="userspace-preview-card">
          <h4>{title}</h4>
          <p>{panelCount} panel(s)</p>
          <pre>{JSON.stringify(parsed, null, 2)}</pre>
        </div>
      );
    } catch {
      return (
        <div className="userspace-preview-card">
          <h4>Dashboard JSON</h4>
          <p>Invalid JSON</p>
          <pre>{content}</pre>
        </div>
      );
    }
  }

  if (resolvedType === 'report_markdown') {
    return (
      <div className="userspace-preview-card markdown-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    );
  }

  if (resolvedType === 'report_html') {
    const sandboxValue = allowActivePreview ? 'allow-scripts' : '';
    const htmlPreviewSrcDoc = buildHtmlPreviewSrcDoc(content, allowActivePreview);
    return (
      <div className="userspace-preview-card userspace-preview-frame-wrap">
        <div className="userspace-preview-controls">
          <p className="userspace-preview-warning">
            HTML preview can execute scripts. Keep disabled unless you trust this file.
          </p>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={allowActivePreview}
              disabled={!canEnableActivePreview}
              onChange={(event) => setAllowActivePreview(event.target.checked)}
            />
            <span>Enable active HTML preview</span>
          </label>
          {!canEnableActivePreview ? (
            <p className="userspace-muted">Viewer access: active HTML execution is disabled.</p>
          ) : null}
        </div>
        <iframe
          key={`${filePath}:${resolvedType}:${allowActivePreview ? 'active' : 'safe'}`}
          className="userspace-preview-frame"
          sandbox={sandboxValue}
          srcDoc={htmlPreviewSrcDoc}
          referrerPolicy="no-referrer"
          loading="lazy"
          title="User Space HTML Preview"
        />
      </div>
    );
  }

  if (resolvedType === 'module_js') {
    if (!allowActivePreview) {
      return (
        <div className="userspace-preview-card">
          <h4>JavaScript module preview</h4>
          <p className="userspace-preview-warning">
            Active module execution is disabled by default. Enable only for trusted code.
          </p>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={allowActivePreview}
              disabled={!canEnableActivePreview}
              onChange={(event) => setAllowActivePreview(event.target.checked)}
            />
            <span>Enable JS module execution</span>
          </label>
          {!canEnableActivePreview ? (
            <p className="userspace-muted">Viewer access: JS execution is disabled.</p>
          ) : null}
          <pre>{content}</pre>
        </div>
      );
    }

    return (
      <div className="userspace-preview-card userspace-preview-frame-wrap">
        <div className="userspace-preview-controls">
          <p className="userspace-preview-warning">
            JS module is running in a restricted sandbox without same-origin or network access.
          </p>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={allowActivePreview}
              disabled={!canEnableActivePreview}
              onChange={(event) => setAllowActivePreview(event.target.checked)}
            />
            <span>Keep JS module execution enabled</span>
          </label>
        </div>
        <iframe
          key={`${filePath}:${resolvedType}:active`}
          className="userspace-preview-frame"
          sandbox="allow-scripts"
          srcDoc={jsModuleSrcDoc}
          referrerPolicy="no-referrer"
          loading="lazy"
          title="User Space JS Module Preview"
        />
      </div>
    );
  }

  if (resolvedType === 'module_ts') {
    return (
      <div className="userspace-preview-card">
        <h4>TypeScript module preview</h4>
        <p>Execution is disabled for TypeScript source in this bootstrap renderer.</p>
        <pre>{content}</pre>
      </div>
    );
  }

  return (
    <div className="userspace-preview-card">
      <h4>Raw Preview</h4>
      <pre>{content}</pre>
    </div>
  );
}
