import { history, historyKeymap, redo, redoDepth, undo, undoDepth } from '@codemirror/commands';
import { EditorView, keymap } from '@codemirror/view';
import CodeMirror from '@uiw/react-codemirror';
import { Code2, Eye, Maximize2, Minimize2, Redo2, RefreshCw, Save, Undo2, X } from 'lucide-react';
import type { ReactNode } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  createCodeMirrorThemeCompartment,
  createCodeMirrorThemeExtension,
  reconfigureCodeMirrorTheme,
} from '@/theme/codemirrorTheme';
import { getThemeSnapshot, subscribeToThemeChanges } from '@/theme/themeSnapshot';
import { useCodeMirrorLanguageExtension } from '@/utils/codemirrorLanguage';
import {
  buildHtmlComponentSrcdoc,
  hashString,
  sampleHtmlComponentTheme,
} from '@/utils/htmlComponent/buildSrcdoc';
import {
  CHAT_HTML_COMPONENT_BRIDGE,
  CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS,
  CHAT_HTML_COMPONENT_MESSAGE_TYPES,
  HTML_COMPONENT_DEFAULT_HEIGHT,
  HTML_COMPONENT_EXPANDED_MAX_HEIGHT_RATIO,
  HTML_COMPONENT_MAX_HEIGHT_RATIO,
  HTML_COMPONENT_MIN_HEIGHT,
  HTML_COMPONENT_READY_TIMEOUT_MS,
} from '@/utils/htmlComponent/constants';
import { useChatComponentSandboxPolicy } from '@/utils/htmlComponent/sandboxPolicy';

import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ThemeChromeIcon } from './shared/ThemeChromeIcon';

export interface HtmlComponentData {
  __html_component__: true;
  title: string;
  html: string;
  data?: unknown;
  description?: string;
  height?: number | null;
  data_connection?: {
    component_id?: string;
    request?: unknown;
    source_tool_config_id?: string;
    source_input?: unknown;
    [k: string]: unknown;
  };
}

export interface HtmlComponentDisplayProps {
  component: HtmlComponentData;
  onDisplayError?: (message: string) => void;
  /** ChatPanel passes `<VisualizationVersionAnchor …/>` so this module never imports ChatPanel. */
  anchor?: ReactNode;
  /** Rendered beneath the title in the header; ChatPanel passes `<LinkifiedText …/>`. */
  descriptionNode?: ReactNode;
  /**
   * When provided, the source view becomes editable and Save persists the markup
   * (ChatPanel stores it as a new visualization version). Absent in read-only contexts.
   */
  onSaveHtml?: (html: string) => Promise<void>;
}

type FrameStatus = 'loading' | 'ready' | 'error';
type ViewMode = 'canvas' | 'code';

/**
 * Source view setup: mirrors the diff viewer but keeps line numbers for navigation.
 * History is provided explicitly below so its depth can be capped.
 */
const SOURCE_CODEMIRROR_SETUP = {
  lineNumbers: true,
  bracketMatching: true,
  indentOnInput: false,
  tabSize: 2,
  autocompletion: false,
  closeBrackets: false,
  foldGutter: false,
  highlightActiveLine: false,
  history: false,
  historyKeymap: false,
};

/** Virtual path so the shared language loader picks the html grammar. */
const SOURCE_LANGUAGE_PATH = 'component.html';

/** Undo/redo keeps the last 30 edit events (CodeMirror's `minDepth` is also its cap). */
const SOURCE_HISTORY_DEPTH = 30;

interface HistoryDepth {
  undo: number;
  redo: number;
}

const EMPTY_HISTORY_DEPTH: HistoryDepth = { undo: 0, redo: 0 };

interface BridgeMessage {
  bridge?: unknown;
  type?: unknown;
  height?: unknown;
  message?: unknown;
}

const POLICY_UNAVAILABLE_MESSAGE =
  'Interactive components are unavailable because the preview sandbox policy could not be loaded.';
const READY_TIMEOUT_MESSAGE =
  'The component has not signaled that it is ready yet. It may still be loading external libraries.';
const GENERIC_RUNTIME_ERROR = 'The component reported an error.';

/** Sentinel so the first DATA post is never skipped by identity comparison. */
const NO_DATA_POSTED = Symbol('no-data-posted');

const DENIED_SANDBOX_FLAGS = new Set<string>(CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS);

/**
 * Defensive last line of §6.17: even if the policy hook ever surfaced a denied flag,
 * the rendered `sandbox` attribute never carries same-origin or top-navigation grants.
 */
function stripDeniedSandboxFlags(sandboxAttribute: string): string {
  return sandboxAttribute
    .split(/\s+/)
    .filter((flag) => flag && !DENIED_SANDBOX_FLAGS.has(flag))
    .join(' ');
}

function clampFrameHeight(height: number, isExpanded: boolean): number {
  if (!Number.isFinite(height)) return HTML_COMPONENT_DEFAULT_HEIGHT;
  const ratio = isExpanded
    ? HTML_COMPONENT_EXPANDED_MAX_HEIGHT_RATIO
    : HTML_COMPONENT_MAX_HEIGHT_RATIO;
  const viewportHeight = typeof window === 'undefined' ? 0 : window.innerHeight;
  const maxHeight =
    viewportHeight > 0
      ? Math.max(HTML_COMPONENT_MIN_HEIGHT, Math.floor(viewportHeight * ratio))
      : Number.POSITIVE_INFINITY;
  return Math.min(Math.max(Math.round(height), HTML_COMPONENT_MIN_HEIGHT), maxHeight);
}

function isBridgeMessage(value: unknown): value is BridgeMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as BridgeMessage).bridge === CHAT_HTML_COMPONENT_BRIDGE
  );
}

export const HtmlComponentDisplay = memo(function HtmlComponentDisplay({
  component,
  onDisplayError,
  anchor,
  descriptionNode,
  onSaveHtml,
}: HtmlComponentDisplayProps): JSX.Element {
  const policy = useChatComponentSandboxPolicy();

  const [isExpanded, setIsExpanded] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('canvas');
  // Unsaved editor text; null means the editor mirrors the persisted markup.
  const [draftHtml, setDraftHtml] = useState<string | null>(null);
  const [isSavingHtml, setIsSavingHtml] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [historyDepth, setHistoryDepth] = useState<HistoryDepth>(EMPTY_HISTORY_DEPTH);
  const [instanceKey, setInstanceKey] = useState(0);
  const [status, setStatus] = useState<FrameStatus>('loading');
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [readyWarning, setReadyWarning] = useState<string | null>(null);
  const [frameHeight, setFrameHeight] = useState(() =>
    clampFrameHeight(component.height ?? HTML_COMPONENT_DEFAULT_HEIGHT, false),
  );

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const statusRef = useRef<FrameStatus>('loading');
  const latestDataRef = useRef<unknown>(component.data);
  const postedDataRef = useRef<unknown>(NO_DATA_POSTED);
  const dataVersionRef = useRef(0);
  const reportedErrorRef = useRef<string | null>(null);
  const contentHeightRef = useRef<number | null>(null);
  const readyTimeoutRef = useRef<number | null>(null);

  statusRef.current = status;
  latestDataRef.current = component.data;

  const clearReadyTimeout = useCallback(() => {
    if (readyTimeoutRef.current !== null) {
      window.clearTimeout(readyTimeoutRef.current);
      readyTimeoutRef.current = null;
    }
  }, []);

  // Keyed on html + instanceKey only: Refresh (data) and theme toggles must never reload
  // the frame. On READY the parent re-posts the latest DATA and THEME instead.
  const srcdoc = useMemo(
    () =>
      buildHtmlComponentSrcdoc({
        html: component.html,
        data: latestDataRef.current ?? null,
        theme: sampleHtmlComponentTheme(),
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- data/theme intentionally excluded (§6.14)
    [component.html, instanceKey],
  );

  const frameKey = `${hashString(component.html)}:${instanceKey}`;

  const postToFrame = useCallback((type: string, payload: Record<string, unknown>) => {
    const frameWindow = iframeRef.current?.contentWindow;
    if (!frameWindow) return false;
    frameWindow.postMessage({ bridge: CHAT_HTML_COMPONENT_BRIDGE, type, ...payload }, '*');
    return true;
  }, []);

  const postTheme = useCallback(() => {
    postToFrame(CHAT_HTML_COMPONENT_MESSAGE_TYPES.THEME, { theme: sampleHtmlComponentTheme() });
  }, [postToFrame]);

  const postData = useCallback(() => {
    const data = latestDataRef.current;
    dataVersionRef.current += 1;
    if (
      postToFrame(CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA, {
        data: data ?? null,
        version: dataVersionRef.current,
      })
    ) {
      postedDataRef.current = data;
    }
  }, [postToFrame]);

  // Reset per-frame state whenever a new frame instance is mounted.
  useEffect(() => {
    clearReadyTimeout();
    statusRef.current = 'loading';
    postedDataRef.current = NO_DATA_POSTED;
    reportedErrorRef.current = null;
    setStatus('loading');
    setRuntimeError(null);
    setReadyWarning(null);
  }, [component.html, instanceKey, clearReadyTimeout]);

  useEffect(() => clearReadyTimeout, [clearReadyTimeout]);

  // Parent-side bridge listener (§6.10): source, opaque origin, and bridge id are all required.
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const frameWindow = iframeRef.current?.contentWindow;
      if (!frameWindow || event.source !== frameWindow) return;
      if (event.origin !== 'null') return;
      if (!isBridgeMessage(event.data)) return;

      const message = event.data;
      switch (message.type) {
        case CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY: {
          clearReadyTimeout();
          statusRef.current = 'ready';
          setStatus('ready');
          setReadyWarning(null);
          postTheme();
          postData();
          return;
        }
        case CHAT_HTML_COMPONENT_MESSAGE_TYPES.RESIZE: {
          if (typeof message.height !== 'number' || !Number.isFinite(message.height)) return;
          contentHeightRef.current = message.height;
          setFrameHeight(clampFrameHeight(message.height, isExpanded));
          return;
        }
        case CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR: {
          const text =
            typeof message.message === 'string' && message.message.trim()
              ? message.message.trim()
              : GENERIC_RUNTIME_ERROR;
          clearReadyTimeout();
          statusRef.current = 'error';
          setStatus('error');
          setRuntimeError(text);
          if (reportedErrorRef.current !== text) {
            reportedErrorRef.current = text;
            onDisplayError?.(text);
          }
          return;
        }
        default:
          return;
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [clearReadyTimeout, isExpanded, onDisplayError, postData, postTheme]);

  // Push new data (Refresh / branch switch) into the running frame without reloading it.
  useEffect(() => {
    if (status !== 'ready') return;
    if (postedDataRef.current === component.data) return;
    postData();
  }, [component.data, postData, status]);

  // Re-theme in place when the color mode or theme pack changes.
  useEffect(() => subscribeToThemeChanges(postTheme), [postTheme]);

  // Re-clamp the last reported content height when the max ratio changes.
  useEffect(() => {
    if (contentHeightRef.current === null) return;
    setFrameHeight(clampFrameHeight(contentHeightRef.current, isExpanded));
  }, [isExpanded]);

  const handleFrameLoad = useCallback(() => {
    clearReadyTimeout();
    if (statusRef.current !== 'loading') return;
    readyTimeoutRef.current = window.setTimeout(() => {
      readyTimeoutRef.current = null;
      if (statusRef.current === 'loading') {
        setReadyWarning(READY_TIMEOUT_MESSAGE);
      }
    }, HTML_COMPONENT_READY_TIMEOUT_MS);
  }, [clearReadyTimeout]);

  const handleReload = useCallback(() => {
    setInstanceKey((key) => key + 1);
  }, []);

  const handleToggleViewMode = useCallback(() => {
    setViewMode((mode) => (mode === 'canvas' ? 'code' : 'canvas'));
  }, []);

  // Source view: the shared CodeMirror theme compartment + html grammar, kept in sync
  // with the app theme the same way UserSpaceFileDiffView does.
  const sourceViewRef = useRef<EditorView | null>(null);
  const sourceThemeCompartment = useMemo(() => createCodeMirrorThemeCompartment(), []);
  const sourceLanguageExtension = useCodeMirrorLanguageExtension(SOURCE_LANGUAGE_PATH);
  const historyDepthRef = useRef<HistoryDepth>(EMPTY_HISTORY_DEPTH);
  const sourceExtensions = useMemo(
    () => [
      sourceThemeCompartment.of(createCodeMirrorThemeExtension(getThemeSnapshot())),
      // Agent markup is often a single long line; wrap so it stays readable in the canvas width.
      EditorView.lineWrapping,
      history({ minDepth: SOURCE_HISTORY_DEPTH }),
      keymap.of(historyKeymap),
      // Mirror the history depth into React state so the Undo/Redo buttons track it.
      EditorView.updateListener.of((update) => {
        const next = { undo: undoDepth(update.state), redo: redoDepth(update.state) };
        const prev = historyDepthRef.current;
        if (next.undo !== prev.undo || next.redo !== prev.redo) {
          historyDepthRef.current = next;
          setHistoryDepth(next);
        }
      }),
      ...(sourceLanguageExtension ? [sourceLanguageExtension] : []),
    ],
    [sourceLanguageExtension, sourceThemeCompartment],
  );
  useEffect(
    () =>
      subscribeToThemeChanges(() => {
        reconfigureCodeMirrorTheme(
          sourceViewRef.current,
          sourceThemeCompartment,
          getThemeSnapshot(),
        );
      }),
    [sourceThemeCompartment],
  );
  const handleSourceEditorCreated = useCallback(
    (view: EditorView) => {
      sourceViewRef.current = view;
      reconfigureCodeMirrorTheme(view, sourceThemeCompartment, getThemeSnapshot());
    },
    [sourceThemeCompartment],
  );

  // A new persisted version (save, refresh, or branch switch) supersedes any draft; the
  // editor is remounted (keyed on the markup) so its undo history starts fresh too.
  useEffect(() => {
    setDraftHtml(null);
    setSaveError(null);
    historyDepthRef.current = EMPTY_HISTORY_DEPTH;
    setHistoryDepth(EMPTY_HISTORY_DEPTH);
  }, [component.html]);

  const handleUndo = useCallback(() => {
    const view = sourceViewRef.current;
    if (!view) return;
    undo(view);
    view.focus();
  }, []);

  const handleRedo = useCallback(() => {
    const view = sourceViewRef.current;
    if (!view) return;
    redo(view);
    view.focus();
  }, []);

  const handleSourceChange = useCallback(
    (value: string) => {
      setDraftHtml(value === component.html ? null : value);
      setSaveError(null);
    },
    [component.html],
  );

  const handleDiscardHtml = useCallback(() => {
    setDraftHtml(null);
    setSaveError(null);
  }, []);

  const handleSaveHtml = useCallback(async () => {
    if (!onSaveHtml || draftHtml === null || isSavingHtml) return;
    setIsSavingHtml(true);
    setSaveError(null);
    try {
      await onSaveHtml(draftHtml);
      setDraftHtml(null);
      setViewMode('canvas');
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save component');
    } finally {
      setIsSavingHtml(false);
    }
  }, [draftHtml, isSavingHtml, onSaveHtml]);

  const isSourceEditable = Boolean(onSaveHtml);
  const hasUnsavedChanges = draftHtml !== null;

  const handleToggleExpanded = useCallback(() => {
    setIsExpanded((value) => !value);
  }, []);

  const sandboxAttribute = stripDeniedSandboxFlags(policy.sandboxAttribute);
  const blockedReason =
    policy.status === 'error' ? POLICY_UNAVAILABLE_MESSAGE : policy.blockedReason;
  const canRenderFrame = policy.status === 'ready' && !blockedReason;

  return (
    <div
      className={`html-component-with-anchor${
        isExpanded ? ' html-component-with-anchor-expanded' : ''
      }`}
    >
      <div
        className={`html-component-container${
          isExpanded ? ' html-component-container-expanded' : ''
        }`}
      >
        <div className="html-component-header">
          <div className="html-component-heading">
            <span className="html-component-title">{component.title}</span>
            {descriptionNode ? (
              <div className="html-component-description">{descriptionNode}</div>
            ) : null}
          </div>
          <div className="html-component-actions">
            {viewMode === 'code' && isSourceEditable && (
              <>
                <button
                  type="button"
                  className="html-component-action-btn"
                  onClick={handleUndo}
                  disabled={historyDepth.undo === 0 || isSavingHtml}
                  aria-label="Undo"
                  title="Undo"
                >
                  <ThemeChromeIcon fallback={<Undo2 size={14} />} codicon="discard" size={14} />
                </button>
                <button
                  type="button"
                  className="html-component-action-btn"
                  onClick={handleRedo}
                  disabled={historyDepth.redo === 0 || isSavingHtml}
                  aria-label="Redo"
                  title="Redo"
                >
                  <ThemeChromeIcon fallback={<Redo2 size={14} />} codicon="redo" size={14} />
                </button>
                <button
                  type="button"
                  className="html-component-action-btn"
                  onClick={handleDiscardHtml}
                  disabled={!hasUnsavedChanges || isSavingHtml}
                  aria-label="Discard changes"
                  title="Discard changes"
                >
                  <ThemeChromeIcon fallback={<X size={14} />} codicon="close" size={14} />
                </button>
                <button
                  type="button"
                  className="html-component-action-btn html-component-action-primary"
                  onClick={handleSaveHtml}
                  disabled={!hasUnsavedChanges || isSavingHtml}
                  aria-label={isSavingHtml ? 'Saving changes' : 'Save changes'}
                  title={isSavingHtml ? 'Saving…' : 'Save changes'}
                >
                  {isSavingHtml ? (
                    <MiniLoadingSpinner variant="icon" size={14} />
                  ) : (
                    <ThemeChromeIcon fallback={<Save size={14} />} codicon="save" size={14} />
                  )}
                </button>
                <span className="html-component-actions-divider" aria-hidden="true" />
              </>
            )}
            <button
              type="button"
              className={`html-component-action-btn${
                viewMode === 'code' ? ' html-component-action-btn-active' : ''
              }`}
              onClick={handleToggleViewMode}
              aria-pressed={viewMode === 'code'}
              aria-label={viewMode === 'code' ? 'Show component' : 'View source'}
              title={viewMode === 'code' ? 'Show component' : 'View source'}
            >
              {viewMode === 'code' ? (
                <ThemeChromeIcon fallback={<Eye size={14} />} codicon="preview" size={14} />
              ) : (
                <ThemeChromeIcon fallback={<Code2 size={14} />} codicon="code" size={14} />
              )}
            </button>
            <button
              type="button"
              className="html-component-action-btn"
              onClick={handleReload}
              aria-label="Reload"
              title="Reload"
            >
              <ThemeChromeIcon fallback={<RefreshCw size={14} />} codicon="refresh" size={14} />
            </button>
            <button
              type="button"
              className="html-component-action-btn"
              onClick={handleToggleExpanded}
              aria-label={isExpanded ? 'Collapse component' : 'Expand component'}
              title={isExpanded ? 'Collapse component' : 'Expand component'}
            >
              <ThemeChromeIcon
                fallback={isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                codicon={isExpanded ? 'screen-normal' : 'screen-full'}
                size={14}
              />
            </button>
          </div>
        </div>
        {policy.status === 'loading' && (
          <div className="html-component-status" role="status">
            <MiniLoadingSpinner variant="icon" size={14} />
            <span>Loading sandbox policy…</span>
          </div>
        )}
        {policy.status !== 'loading' && blockedReason && (
          <div className="html-component-blocked" role="status">
            {blockedReason}
          </div>
        )}
        {canRenderFrame && (
          // Hidden rather than unmounted in code mode so toggling back never reloads the frame.
          <div className="html-component-canvas" hidden={viewMode === 'code'}>
            <iframe
              key={frameKey}
              ref={iframeRef}
              className="html-component-frame"
              title={component.title}
              sandbox={sandboxAttribute}
              srcDoc={srcdoc}
              style={{ height: `${frameHeight}px` }}
              referrerPolicy="no-referrer"
              onLoad={handleFrameLoad}
            />
            {status === 'loading' && (
              <div className="html-component-status html-component-status-overlay" role="status">
                <MiniLoadingSpinner variant="icon" size={16} />
                {readyWarning && (
                  <span className="html-component-ready-warning">{readyWarning}</span>
                )}
              </div>
            )}
          </div>
        )}
        {viewMode === 'canvas' && runtimeError && (
          <div className="html-component-error" role="alert">
            <span className="html-component-error-message">{runtimeError}</span>
            <button type="button" className="html-component-error-reload" onClick={handleReload}>
              Reload component
            </button>
          </div>
        )}
        {viewMode === 'code' && (
          <div className="html-component-source" style={{ height: `${frameHeight}px` }}>
            <CodeMirror
              key={hashString(component.html)}
              value={draftHtml ?? component.html}
              // "none" skips the wrapper's built-in light theme, whose white editor
              // background would otherwise override the app palette in dark mode.
              theme="none"
              basicSetup={SOURCE_CODEMIRROR_SETUP}
              editable={isSourceEditable}
              readOnly={!isSourceEditable}
              extensions={sourceExtensions}
              height="100%"
              onChange={handleSourceChange}
              onCreateEditor={handleSourceEditorCreated}
            />
          </div>
        )}
        {viewMode === 'code' && saveError && (
          <div className="html-component-save-error" role="alert">
            {saveError}
          </div>
        )}
      </div>
      {anchor}
    </div>
  );
});
