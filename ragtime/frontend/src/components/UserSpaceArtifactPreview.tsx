import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ts from 'typescript';
import type { UserSpaceLiveDataConnection } from '@/types';
import { api } from '@/api/client';
import {
  buildIframeDoc,
  collectLocalSpecifiers,
  isLocalSpecifier,
  normalizePath,
  resolveWorkspaceModulePath,
  rewriteLocalSpecifiers,
  readThemeTokens,
  USERSPACE_EXEC_BRIDGE,
  USERSPACE_EXEC_MESSAGE_TYPES,
} from '@/utils/userspacePreview';
import type { ModuleMap } from '@/utils/userspacePreview';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
  liveDataConnections?: UserSpaceLiveDataConnection[];
  previewInstanceKey?: string;
  workspaceId?: string;
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
  sharePassword?: string;
  onExecutionStateChange?: (isExecuting: boolean) => void;
}

export function UserSpaceArtifactPreview({
  entryPath,
  workspaceFiles,
  liveDataConnections = [],
  previewInstanceKey,
  workspaceId,
  shareToken,
  ownerUsername,
  shareSlug,
  sharePassword,
  onExecutionStateChange,
}: UserSpaceArtifactPreviewProps) {
  const themeTokens = readThemeTokens();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const lastExecutionErrorLogRef = useRef<Record<string, number>>({});
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

      if (!event.data || event.data.bridge !== USERSPACE_EXEC_BRIDGE) return;

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.ERROR) {
        const componentId = typeof event.data.component_id === 'string' ? event.data.component_id : 'unknown';
        const error = typeof event.data.error === 'string' ? event.data.error : 'Unknown execution error';
        console.error('[UserSpacePreview] iframe execute error:', {
          component_id: componentId,
          error,
        });
        return;
      }

      if (event.data.type !== USERSPACE_EXEC_MESSAGE_TYPES.EXECUTE) return;

      const { callId, component_id, request } = event.data;
      if (typeof callId !== 'string' || typeof component_id !== 'string') return;

      setPendingExecutions((current) => current + 1);
      setExecutionError(null);

      if (!workspaceId && !shareToken && !(ownerUsername && shareSlug)) {
        const errorMessage = 'No workspace context available';
        setExecutionError(errorMessage);
        console.error('[UserSpacePreview] execute-component failed:', errorMessage);
        frameWindow.postMessage(
          {
            bridge: USERSPACE_EXEC_BRIDGE,
            type: USERSPACE_EXEC_MESSAGE_TYPES.RESULT,
            callId,
            result: { rows: [], columns: [], row_count: 0, error: errorMessage },
          },
          '*'
        );
        setPendingExecutions((current) => Math.max(0, current - 1));
        return;
      }

      try {
        const result = shareToken
          ? await api.executeUserSpaceSharedComponent(shareToken, {
            component_id,
            request,
          }, sharePassword)
          : ownerUsername && shareSlug
            ? await api.executeUserSpaceSharedComponentBySlug(ownerUsername, shareSlug, {
              component_id,
              request,
            }, sharePassword)
            : await api.executeWorkspaceComponent(workspaceId as string, {
              component_id,
              request,
            });
        const normalizedResult = normalizeExecuteResult(result);
        const normalizedError = typeof normalizedResult.error === 'string' ? normalizedResult.error.trim() : '';
        if (normalizedError) {
          const dedupeKey = `${component_id}:${normalizedError}`;
          const now = Date.now();
          const lastLoggedAt = lastExecutionErrorLogRef.current[dedupeKey] ?? 0;
          if (now - lastLoggedAt > 10_000) {
            lastExecutionErrorLogRef.current[dedupeKey] = now;
            console.warn('[UserSpacePreview] execute-component returned error:', {
              component_id,
              error: normalizedError,
              request,
            });
          }
        }
        frameWindow.postMessage(
          {
            bridge: USERSPACE_EXEC_BRIDGE,
            type: USERSPACE_EXEC_MESSAGE_TYPES.RESULT,
            callId,
            result: normalizedResult,
          },
          '*'
        );
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        console.error('[UserSpacePreview] execute-component request failed:', {
          component_id,
          error: errorMessage,
          request,
        });
        frameWindow.postMessage(
          {
            bridge: USERSPACE_EXEC_BRIDGE,
            type: USERSPACE_EXEC_MESSAGE_TYPES.RESULT,
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
    [workspaceId, shareToken, ownerUsername, shareSlug, sharePassword, normalizeExecuteResult]
  );

  useEffect(() => {
    window.addEventListener('message', handleIframeMessage);
    return () => window.removeEventListener('message', handleIframeMessage);
  }, [handleIframeMessage]);

  useEffect(() => {
    setExecutionError(null);
  }, [workspaceId, shareToken, ownerUsername, shareSlug, sharePassword, entryPath, previewInstanceKey]);

  useEffect(() => {
    onExecutionStateChange?.(pendingExecutions > 0);
  }, [pendingExecutions, onExecutionStateChange]);

  useEffect(() => {
    setPendingExecutions(0);
  }, [workspaceId, shareToken, ownerUsername, shareSlug, sharePassword, entryPath, previewInstanceKey]);

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
        if (!isLocalSpecifier(specifier)) {
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
