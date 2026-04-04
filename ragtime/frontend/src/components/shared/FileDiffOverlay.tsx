import { useEffect, useMemo, useRef } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { json } from '@codemirror/lang-json';
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { markdown } from '@codemirror/lang-markdown';
import { xml } from '@codemirror/lang-xml';
import { yaml } from '@codemirror/lang-yaml';
import { sql } from '@codemirror/lang-sql';
import { Decoration, type DecorationSet, EditorView } from '@codemirror/view';
import { StateField, type Extension } from '@codemirror/state';
import { diffLines } from 'diff';
import type { UserSpaceSnapshotFileDiff } from '@/types';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';

interface AlignedDiffResult {
  beforeText: string;
  afterText: string;
  beforeDeletedLines: Set<number>;
  afterAddedLines: Set<number>;
  beforePaddingLines: Set<number>;
  afterPaddingLines: Set<number>;
}

function getLanguageExtensionForPath(filePath: string) {
  const lower = filePath.toLowerCase();
  if (/\.[cm]?tsx?$/i.test(lower)) return javascript({ typescript: true, jsx: /x$/i.test(lower) });
  if (/\.[cm]?jsx?$/i.test(lower)) return javascript({ typescript: false, jsx: true });
  if (/\.py$/i.test(lower)) return python();
  if (/\.json[c5]?$/i.test(lower) || lower.endsWith('.jsonl')) return json();
  if (/\.css$/i.test(lower) || lower.endsWith('.scss') || lower.endsWith('.less')) return css();
  if (/\.html?$/i.test(lower) || lower.endsWith('.svelte') || lower.endsWith('.vue')) return html();
  if (/\.ya?ml$/i.test(lower)) return yaml();
  if (/\.xml$/i.test(lower) || lower.endsWith('.svg')) return xml();
  if (/\.sql$/i.test(lower)) return sql();
  if (/\.mdx?$/i.test(lower) || lower.endsWith('.markdown')) return markdown();
  return null;
}

function computeAlignedDiff(before: string, after: string): AlignedDiffResult {
  const changes = diffLines(before, after);

  const beforeArr: string[] = [];
  const afterArr: string[] = [];
  const beforeDeletedLines = new Set<number>();
  const afterAddedLines = new Set<number>();
  const beforePaddingLines = new Set<number>();
  const afterPaddingLines = new Set<number>();

  for (const change of changes) {
    const raw = change.value;
    const lines = raw.endsWith('\n') ? raw.slice(0, -1).split('\n') : raw.split('\n');

    if (change.removed) {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push('');
        beforeDeletedLines.add(beforeArr.length);
        afterPaddingLines.add(afterArr.length);
      }
    } else if (change.added) {
      for (const line of lines) {
        beforeArr.push('');
        afterArr.push(line);
        afterAddedLines.add(afterArr.length);
        beforePaddingLines.add(beforeArr.length);
      }
    } else {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push(line);
      }
    }
  }

  return {
    beforeText: beforeArr.join('\n'),
    afterText: afterArr.join('\n'),
    beforeDeletedLines,
    afterAddedLines,
    beforePaddingLines,
    afterPaddingLines,
  };
}

const diffLineDeletedMark = Decoration.line({ class: 'cm-diff-line-deleted' });
const diffLineAddedMark = Decoration.line({ class: 'cm-diff-line-added' });
const diffLinePaddingMark = Decoration.line({ class: 'cm-diff-line-padding' });

function buildDiffHighlightExtension(lineNumbers: Set<number>, decoration: Decoration) {
  return StateField.define<DecorationSet>({
    create(state) {
      const builder: import('@codemirror/state').Range<Decoration>[] = [];
      for (let i = 1; i <= state.doc.lines; i++) {
        if (lineNumbers.has(i)) {
          builder.push(decoration.range(state.doc.line(i).from));
        }
      }
      return Decoration.set(builder);
    },
    update(value) {
      return value;
    },
    provide: (field) => EditorView.decorations.from(field),
  });
}

function formatDiffStatus(status: 'A' | 'D' | 'M' | 'R'): string {
  switch (status) {
    case 'A':
      return 'Added';
    case 'D':
      return 'Deleted';
    case 'R':
      return 'Renamed';
    default:
      return 'Modified';
  }
}

const DIFF_CODEMIRROR_SETUP = {
  lineNumbers: true,
  bracketMatching: true,
  indentOnInput: true,
  tabSize: 2,
  autocompletion: false,
  closeBrackets: false,
  foldGutter: false,
  highlightActiveLine: false,
};

export interface FileDiffOverlayProps {
  diff: UserSpaceSnapshotFileDiff | null;
  diffKey: string | null;
  loading: boolean;
  error: string | null;
  title: string;
  beforeLabel: string;
  afterLabel: string;
  formatError?: (error: string | null) => string | null;
  onDismiss: () => void;
  onOverlayClick: () => void;
  onOverlayMouseEnter: () => void;
  onOverlayMouseLeave: () => void;
}

export function FileDiffOverlay({
  diff,
  diffKey,
  loading,
  error,
  title,
  beforeLabel,
  afterLabel,
  formatError,
  onDismiss,
  onOverlayClick,
  onOverlayMouseEnter,
  onOverlayMouseLeave,
}: FileDiffOverlayProps) {
  const beforeWrapRef = useRef<HTMLDivElement | null>(null);
  const afterWrapRef = useRef<HTMLDivElement | null>(null);
  const scrollSyncingRef = useRef(false);

  const languageExtensions = useMemo(() => {
    const path = diff?.path ?? '';
    const ext = getLanguageExtensionForPath(path);
    return ext ? [ext] : [];
  }, [diff?.path]);

  const alignedDiff = useMemo(() => {
    if (!diff || diff.is_binary) return null;
    return computeAlignedDiff(diff.before_content, diff.after_content);
  }, [diff]);

  const beforeExtensions = useMemo(() => {
    const exts: Extension[] = [...languageExtensions];
    if (alignedDiff) {
      if (alignedDiff.beforeDeletedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.beforeDeletedLines, diffLineDeletedMark));
      }
      if (alignedDiff.beforePaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.beforePaddingLines, diffLinePaddingMark));
      }
    }
    return exts;
  }, [languageExtensions, alignedDiff]);

  const afterExtensions = useMemo(() => {
    const exts: Extension[] = [...languageExtensions];
    if (alignedDiff) {
      if (alignedDiff.afterAddedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.afterAddedLines, diffLineAddedMark));
      }
      if (alignedDiff.afterPaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.afterPaddingLines, diffLinePaddingMark));
      }
    }
    return exts;
  }, [languageExtensions, alignedDiff]);

  /* Scroll sync between the two diff editors. */
  useEffect(() => {
    if (!diff) return;

    let beforeScroller: HTMLElement | null = null;
    let afterScroller: HTMLElement | null = null;
    let rafId: number | null = null;
    let attempts = 0;
    let detachScrollListeners: (() => void) | null = null;

    const syncScroll = (src: HTMLElement, dst: HTMLElement) => {
      if (scrollSyncingRef.current) return;
      scrollSyncingRef.current = true;
      dst.scrollTop = src.scrollTop;
      dst.scrollLeft = src.scrollLeft;
      requestAnimationFrame(() => {
        scrollSyncingRef.current = false;
      });
    };

    const attachScrollListeners = () => {
      const beforeWrap = beforeWrapRef.current;
      const afterWrap = afterWrapRef.current;
      if (!beforeWrap || !afterWrap) return false;

      beforeScroller = beforeWrap.querySelector<HTMLElement>('.cm-scroller');
      afterScroller = afterWrap.querySelector<HTMLElement>('.cm-scroller');
      if (!beforeScroller || !afterScroller) return false;

      const onBeforeScroll = () => syncScroll(beforeScroller!, afterScroller!);
      const onAfterScroll = () => syncScroll(afterScroller!, beforeScroller!);

      beforeScroller.addEventListener('scroll', onBeforeScroll, { passive: true });
      afterScroller.addEventListener('scroll', onAfterScroll, { passive: true });
      detachScrollListeners = () => {
        beforeScroller?.removeEventListener('scroll', onBeforeScroll);
        afterScroller?.removeEventListener('scroll', onAfterScroll);
      };
      return true;
    };

    const wireWhenReady = () => {
      if (attachScrollListeners()) return;
      attempts += 1;
      if (attempts >= 12) return;
      rafId = window.requestAnimationFrame(wireWhenReady);
    };

    wireWhenReady();

    return () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
      detachScrollListeners?.();
      scrollSyncingRef.current = false;
    };
  }, [diff, diffKey, alignedDiff]);

  if (!loading && !error && !diff) return null;

  const displayError = formatError ? formatError(error) : error;
  const resolvedBeforeLabel = diff?.is_snapshot_own_diff ? 'Previous' : beforeLabel;
  const resolvedAfterLabel = diff?.is_snapshot_own_diff ? 'Snapshot' : afterLabel;

  return (
    <div
      className="userspace-snapshot-diff-backdrop"
      onClick={onDismiss}
    >
      <div
        className="userspace-snapshot-diff-overlay"
        onClick={(event) => {
          event.stopPropagation();
          onOverlayClick();
        }}
        onMouseEnter={onOverlayMouseEnter}
        onMouseLeave={onOverlayMouseLeave}
      >
        <div className="userspace-snapshot-diff-overlay-header">
          <div>
            <div className="userspace-snapshot-diff-overlay-title">{title}</div>
            {diff && (
              <div className="userspace-snapshot-diff-overlay-subtitle">
                <span>{diff.path}</span>
                <span>{formatDiffStatus(diff.status)}</span>
                <span>+{diff.additions} -{diff.deletions}</span>
              </div>
            )}
          </div>
          <button type="button" className="modal-close" onClick={onDismiss}>&times;</button>
        </div>

        {loading ? (
          <div className="userspace-snapshot-diff-overlay-body">
            <div className="userspace-snapshot-expanded-status userspace-snapshot-diff-overlay-loading">
              <MiniLoadingSpinner variant="icon" size={14} />
              <span>Loading file diff...</span>
            </div>
          </div>
        ) : displayError ? (
          <div className="userspace-snapshot-diff-overlay-body">
            <p className="userspace-muted userspace-error">{displayError}</p>
          </div>
        ) : diff ? (
          diff.is_binary || diff.is_truncated ? (
            <div className="userspace-snapshot-diff-overlay-body">
              <p className="userspace-muted">{diff.message ?? 'Content cannot be rendered.'}</p>
            </div>
          ) : (
            <div className="userspace-snapshot-diff-columns">
              <div className="userspace-snapshot-diff-column">
                <div className="userspace-snapshot-diff-column-header">
                  <span>{resolvedBeforeLabel}</span>
                  <code>{diff.before_path ?? diff.path}</code>
                </div>
                <div className="userspace-snapshot-diff-editor-wrap" ref={beforeWrapRef}>
                  <CodeMirror
                    value={alignedDiff?.beforeText ?? diff.before_content}
                    basicSetup={DIFF_CODEMIRROR_SETUP}
                    editable={false}
                    extensions={beforeExtensions}
                    height="100%"
                  />
                </div>
              </div>
              <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
                <div className="userspace-snapshot-diff-column-header">
                  <span>{resolvedAfterLabel}</span>
                  <code>{diff.after_path ?? diff.path}</code>
                </div>
                <div className="userspace-snapshot-diff-editor-wrap" ref={afterWrapRef}>
                  <CodeMirror
                    value={alignedDiff?.afterText ?? diff.after_content}
                    basicSetup={DIFF_CODEMIRROR_SETUP}
                    editable={false}
                    extensions={afterExtensions}
                    height="100%"
                  />
                </div>
              </div>
            </div>
          )
        ) : null}
      </div>
    </div>
  );
}
