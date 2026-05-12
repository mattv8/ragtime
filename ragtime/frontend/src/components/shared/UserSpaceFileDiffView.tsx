import { useEffect, useMemo, useRef, memo } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { Decoration, type DecorationSet, EditorView, lineNumbers as codeMirrorLineNumbers } from '@codemirror/view';
import { StateField, type Extension } from '@codemirror/state';
import { diffLines } from 'diff';
import type { UserSpaceSnapshotFileDiff } from '@/types';
import { useCodeMirrorLanguageExtension } from '@/utils/codemirrorLanguage';

type DiffSourceLineNumber = number | null;

interface AlignedDiffResult {
  beforeText: string;
  afterText: string;
  beforeLineNumbers: DiffSourceLineNumber[];
  afterLineNumbers: DiffSourceLineNumber[];
  beforeDeletedLines: Set<number>;
  afterAddedLines: Set<number>;
  beforePaddingLines: Set<number>;
  afterPaddingLines: Set<number>;
  beforeGapLines?: Set<number>;
  afterGapLines?: Set<number>;
}

interface DiffLineWindow {
  start: number;
  end: number;
}

function computeAlignedDiff(before: string, after: string, startingBeforeLine: number = 1, startingAfterLine: number = 1): AlignedDiffResult {
  const changes = diffLines(before, after);

  const beforeArr: string[] = [];
  const afterArr: string[] = [];
  const beforeLineNumbers: DiffSourceLineNumber[] = [];
  const afterLineNumbers: DiffSourceLineNumber[] = [];
  const beforeDeletedLines = new Set<number>();
  const afterAddedLines = new Set<number>();
  const beforePaddingLines = new Set<number>();
  const afterPaddingLines = new Set<number>();
  let beforeSourceLineNumber = startingBeforeLine;
  let afterSourceLineNumber = startingAfterLine;

  for (const change of changes) {
    const raw = change.value;
    const lines = raw.endsWith('\n') ? raw.slice(0, -1).split('\n') : raw.split('\n');

    if (change.removed) {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push('');
        beforeLineNumbers.push(beforeSourceLineNumber);
        afterLineNumbers.push(null);
        beforeSourceLineNumber += 1;
        beforeDeletedLines.add(beforeArr.length);
        afterPaddingLines.add(afterArr.length);
      }
    } else if (change.added) {
      for (const line of lines) {
        beforeArr.push('');
        afterArr.push(line);
        beforeLineNumbers.push(null);
        afterLineNumbers.push(afterSourceLineNumber);
        afterSourceLineNumber += 1;
        afterAddedLines.add(afterArr.length);
        beforePaddingLines.add(beforeArr.length);
      }
    } else {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push(line);
        beforeLineNumbers.push(beforeSourceLineNumber);
        afterLineNumbers.push(afterSourceLineNumber);
        beforeSourceLineNumber += 1;
        afterSourceLineNumber += 1;
      }
    }
  }

  return {
    beforeText: beforeArr.join('\n'),
    afterText: afterArr.join('\n'),
    beforeLineNumbers,
    afterLineNumbers,
    beforeDeletedLines,
    afterAddedLines,
    beforePaddingLines,
    afterPaddingLines,
  };
}

const diffLineDeletedMark = Decoration.line({ class: 'cm-diff-line-deleted' });
const diffLineAddedMark = Decoration.line({ class: 'cm-diff-line-added' });
const diffLinePaddingMark = Decoration.line({ class: 'cm-diff-line-padding' });
const diffLineGapMark = Decoration.line({ class: 'cm-diff-line-gap' });

const DIFF_CONTEXT_LINE_COUNT = 10;

function splitDiffText(text: string): string[] {
  return text.length === 0 ? [''] : text.split('\n');
}

function remapLineSet(lineNumbers: Set<number>, lineMap: Map<number, number>): Set<number> {
  const remapped = new Set<number>();
  for (const lineNumber of lineNumbers) {
    const mapped = lineMap.get(lineNumber);
    if (mapped != null) {
      remapped.add(mapped);
    }
  }
  return remapped;
}

function buildDiffLineWindows(changedLines: Set<number>, lineCount: number): DiffLineWindow[] {
  if (changedLines.size === 0 || lineCount <= 0) return [];

  const windows: DiffLineWindow[] = Array.from(changedLines)
    .sort((a, b) => a - b)
    .map((lineNumber) => ({
      start: Math.max(1, lineNumber - DIFF_CONTEXT_LINE_COUNT),
      end: Math.min(lineCount, lineNumber + DIFF_CONTEXT_LINE_COUNT),
    }));

  const merged: DiffLineWindow[] = [];
  for (const window of windows) {
    const previous = merged[merged.length - 1];
    if (previous && window.start <= previous.end + 1) {
      previous.end = Math.max(previous.end, window.end);
    } else {
      merged.push({ ...window });
    }
  }

  return merged;
}

function windowAlignedDiff(alignedDiff: AlignedDiffResult): AlignedDiffResult {
  const beforeLines = splitDiffText(alignedDiff.beforeText);
  const afterLines = splitDiffText(alignedDiff.afterText);
  const lineCount = Math.max(beforeLines.length, afterLines.length);
  const changedLines = new Set<number>([
    ...alignedDiff.beforeDeletedLines,
    ...alignedDiff.afterAddedLines,
  ]);
  const windows = buildDiffLineWindows(changedLines, lineCount);

  if (windows.length === 0 || (windows.length === 1 && windows[0].start === 1 && windows[0].end === lineCount)) {
    return alignedDiff;
  }

  const lineMap = new Map<number, number>();
  const beforeWindowedLines: string[] = [];
  const afterWindowedLines: string[] = [];
  const beforeWindowedLineNumbers: DiffSourceLineNumber[] = [];
  const afterWindowedLineNumbers: DiffSourceLineNumber[] = [];
  const beforeGapLines = new Set<number>();
  const afterGapLines = new Set<number>();

  if (windows[0].start > 1) {
    beforeWindowedLines.push('');
    afterWindowedLines.push('');
    beforeWindowedLineNumbers.push(null);
    afterWindowedLineNumbers.push(null);
    beforeGapLines.add(beforeWindowedLines.length);
    afterGapLines.add(afterWindowedLines.length);
  }

  windows.forEach((window, index) => {
    if (index > 0) {
      beforeWindowedLines.push('');
      afterWindowedLines.push('');
      beforeWindowedLineNumbers.push(null);
      afterWindowedLineNumbers.push(null);
      beforeGapLines.add(beforeWindowedLines.length);
      afterGapLines.add(afterWindowedLines.length);
    }

    for (let lineNumber = window.start; lineNumber <= window.end; lineNumber += 1) {
      beforeWindowedLines.push(beforeLines[lineNumber - 1] ?? '');
      afterWindowedLines.push(afterLines[lineNumber - 1] ?? '');
      beforeWindowedLineNumbers.push(alignedDiff.beforeLineNumbers[lineNumber - 1] ?? null);
      afterWindowedLineNumbers.push(alignedDiff.afterLineNumbers[lineNumber - 1] ?? null);
      lineMap.set(lineNumber, beforeWindowedLines.length);
    }
  });

  const lastWindow = windows[windows.length - 1];
  if (lastWindow.end < lineCount) {
    beforeWindowedLines.push('');
    afterWindowedLines.push('');
    beforeWindowedLineNumbers.push(null);
    afterWindowedLineNumbers.push(null);
    beforeGapLines.add(beforeWindowedLines.length);
    afterGapLines.add(afterWindowedLines.length);
  }

  return {
    beforeText: beforeWindowedLines.join('\n'),
    afterText: afterWindowedLines.join('\n'),
    beforeLineNumbers: beforeWindowedLineNumbers,
    afterLineNumbers: afterWindowedLineNumbers,
    beforeDeletedLines: remapLineSet(alignedDiff.beforeDeletedLines, lineMap),
    afterAddedLines: remapLineSet(alignedDiff.afterAddedLines, lineMap),
    beforePaddingLines: remapLineSet(alignedDiff.beforePaddingLines, lineMap),
    afterPaddingLines: remapLineSet(alignedDiff.afterPaddingLines, lineMap),
    beforeGapLines,
    afterGapLines,
  };
}

function buildDiffLineNumberExtension(lineNumberMap: DiffSourceLineNumber[]): Extension {
  return codeMirrorLineNumbers({
    formatNumber(lineNumber) {
      const sourceLineNumber = lineNumberMap[lineNumber - 1];
      return sourceLineNumber == null ? '' : String(sourceLineNumber);
    },
  });
}

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

export function formatDiffStatus(status: 'A' | 'D' | 'M' | 'R'): string {
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

export function calculateDiffLineCounts(before: string, after: string): { additions: number; deletions: number } {
  const changes = diffLines(before, after);
  let additions = 0;
  let deletions = 0;

  for (const change of changes) {
    const raw = change.value;
    const lines = raw.endsWith('\n') ? raw.slice(0, -1).split('\n') : raw.split('\n');
    const lineCount = lines.length;
    if (change.added) {
      additions += lineCount;
    } else if (change.removed) {
      deletions += lineCount;
    }
  }

  return { additions, deletions };
}

export const DIFF_CODEMIRROR_SETUP = {
  lineNumbers: false,
  bracketMatching: true,
  indentOnInput: true,
  tabSize: 2,
  autocompletion: false,
  closeBrackets: false,
  foldGutter: false,
  highlightActiveLine: false,
};

function getDiffFallbackMessage(diff: UserSpaceSnapshotFileDiff): string {
  if (diff.message?.trim()) {
    return diff.message;
  }
  if (diff.is_binary) {
    return 'Binary content cannot be rendered.';
  }
  if (diff.is_truncated) {
    return 'Diff content was truncated and cannot be fully rendered inline.';
  }
  return 'Content cannot be rendered.';
}

export interface UserSpaceFileDiffViewProps {
  diff: UserSpaceSnapshotFileDiff;
  beforeLabel: string;
  afterLabel: string;
  compact?: boolean;
  syncScroll?: boolean;
  highlightSingleColumnChanges?: boolean;
  /**
   * When set, the editor wraps auto-fit to the actual line count (capped at
   * this value) instead of using the default fixed compact min-height. Used
   * by batched tool-call rendering so a 3-line file does not occupy 180px.
   */
  maxLines?: number;
}

// Approximate CodeMirror line-height at our 12px font size, plus a small
// allowance for the editor's internal vertical padding so the last line is
// not clipped when auto-fitting.
const DIFF_LINE_HEIGHT_PX = 18;
const DIFF_AUTO_FIT_PADDING_PX = 8;

function computeAutoFitHeight(text: string, maxLines: number): string {
  const lines = text.length === 0 ? 1 : text.split('\n').length;
  const clamped = Math.max(2, Math.min(lines, Math.max(2, maxLines)));
  return `${clamped * DIFF_LINE_HEIGHT_PX + DIFF_AUTO_FIT_PADDING_PX}px`;
}

export const UserSpaceFileDiffView = memo(function UserSpaceFileDiffView({
  diff,
  beforeLabel,
  afterLabel,
  compact = false,
  syncScroll = true,
  highlightSingleColumnChanges = true,
  maxLines,
}: UserSpaceFileDiffViewProps) {
  const beforeWrapRef = useRef<HTMLDivElement | null>(null);
  const afterWrapRef = useRef<HTMLDivElement | null>(null);
  const scrollSyncingRef = useRef(false);
  const languageExtension = useCodeMirrorLanguageExtension(diff.path ?? '');

  const languageExtensions = useMemo(() => {
    return languageExtension ? [languageExtension] : [];
  }, [languageExtension]);

  const alignedDiff = useMemo(() => {
    if (diff.is_binary || diff.is_truncated) return null;
    const startingBeforeLine = diff.starting_before_line ?? 1;
    const startingAfterLine = diff.starting_after_line ?? 1;
    return windowAlignedDiff(computeAlignedDiff(diff.before_content, diff.after_content, startingBeforeLine, startingAfterLine));
  }, [diff]);

  const beforeExtensions = useMemo(() => {
    const exts: Extension[] = alignedDiff
      ? [buildDiffLineNumberExtension(alignedDiff.beforeLineNumbers), ...languageExtensions]
      : [...languageExtensions];
    if (alignedDiff) {
      if (alignedDiff.beforeDeletedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.beforeDeletedLines, diffLineDeletedMark));
      }
      if (alignedDiff.beforePaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.beforePaddingLines, diffLinePaddingMark));
      }
      if (alignedDiff.beforeGapLines && alignedDiff.beforeGapLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.beforeGapLines, diffLineGapMark));
      }
    }
    return exts;
  }, [languageExtensions, alignedDiff]);

  const afterExtensions = useMemo(() => {
    const exts: Extension[] = alignedDiff
      ? [buildDiffLineNumberExtension(alignedDiff.afterLineNumbers), ...languageExtensions]
      : [...languageExtensions];
    if (alignedDiff) {
      if (alignedDiff.afterAddedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.afterAddedLines, diffLineAddedMark));
      }
      if (alignedDiff.afterPaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.afterPaddingLines, diffLinePaddingMark));
      }
      if (alignedDiff.afterGapLines && alignedDiff.afterGapLines.size > 0) {
        exts.push(buildDiffHighlightExtension(alignedDiff.afterGapLines, diffLineGapMark));
      }
    }
    return exts;
  }, [languageExtensions, alignedDiff]);

  useEffect(() => {
    if (!syncScroll || !alignedDiff) return;

    let beforeScroller: HTMLElement | null = null;
    let afterScroller: HTMLElement | null = null;
    let rafId: number | null = null;
    let attempts = 0;
    let detachScrollListeners: (() => void) | null = null;

    const sync = (src: HTMLElement, dst: HTMLElement) => {
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

      const onBeforeScroll = () => sync(beforeScroller!, afterScroller!);
      const onAfterScroll = () => sync(afterScroller!, beforeScroller!);

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
  }, [alignedDiff, syncScroll]);

  if (!alignedDiff) {
    return (
      <div className={`userspace-snapshot-diff-empty-state${compact ? ' userspace-snapshot-diff-empty-state-compact' : ''}`}>
        <p className="userspace-muted">{getDiffFallbackMessage(diff)}</p>
      </div>
    );
  }

  // Pure addition or deletion: show a single-column view
  const isPureAdd = !diff.before_content && diff.after_content;
  const isPureDelete = diff.before_content && !diff.after_content;

  if (isPureAdd || isPureDelete) {
    const singleContent = isPureAdd ? alignedDiff.afterText : alignedDiff.beforeText;
    const singleLabel = isPureAdd ? afterLabel : beforeLabel;
    const singlePath = isPureAdd ? (diff.after_path ?? diff.path) : (diff.before_path ?? diff.path);
    const singleLineNumbers = isPureAdd ? alignedDiff.afterLineNumbers : alignedDiff.beforeLineNumbers;
    const singleExtensions = highlightSingleColumnChanges
      ? (isPureAdd ? afterExtensions : beforeExtensions)
      : [buildDiffLineNumberExtension(singleLineNumbers), ...languageExtensions];

    const autoFitHeight = maxLines ? computeAutoFitHeight(singleContent, maxLines) : '100%';
    const wrapStyle = maxLines ? { minHeight: 0, height: autoFitHeight } : undefined;
    return (
      <div className={`userspace-snapshot-diff-columns userspace-snapshot-diff-columns-single${compact ? ' userspace-snapshot-diff-columns-compact' : ''}${maxLines ? ' userspace-snapshot-diff-columns-autofit' : ''}`}>
        <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
          <div className="userspace-snapshot-diff-column-header">
            <span>{singleLabel}</span>
            <code>{singlePath}</code>
          </div>
          <div
            className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`}
            style={wrapStyle}
          >
            <CodeMirror
              value={singleContent}
              basicSetup={DIFF_CODEMIRROR_SETUP}
              editable={false}
              extensions={singleExtensions}
              height={autoFitHeight}
            />
          </div>
        </div>
      </div>
    );
  }

  // For two-column diffs we size both panes to the larger of the aligned
  // texts so changes line up visually when auto-fitting.
  const dualAutoFitHeight = maxLines
    ? (() => {
        const beforeLines = alignedDiff.beforeText.split('\n').length;
        const afterLines = alignedDiff.afterText.split('\n').length;
        const lines = Math.max(beforeLines, afterLines, 1);
        const clamped = Math.max(2, Math.min(lines, Math.max(2, maxLines)));
        return `${clamped * DIFF_LINE_HEIGHT_PX + DIFF_AUTO_FIT_PADDING_PX}px`;
      })()
    : '100%';
  const dualWrapStyle = maxLines ? { minHeight: 0, height: dualAutoFitHeight } : undefined;
  return (
    <div className={`userspace-snapshot-diff-columns${compact ? ' userspace-snapshot-diff-columns-compact' : ''}${maxLines ? ' userspace-snapshot-diff-columns-autofit' : ''}`}>
      <div className="userspace-snapshot-diff-column">
        <div className="userspace-snapshot-diff-column-header">
          <span>{beforeLabel}</span>
          <code>{diff.before_path ?? diff.path}</code>
        </div>
        <div
          className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`}
          ref={beforeWrapRef}
          style={dualWrapStyle}
        >
          <CodeMirror
            value={alignedDiff.beforeText}
            basicSetup={DIFF_CODEMIRROR_SETUP}
            editable={false}
            extensions={beforeExtensions}
            height={dualAutoFitHeight}
          />
        </div>
      </div>
      <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
        <div className="userspace-snapshot-diff-column-header">
          <span>{afterLabel}</span>
          <code>{diff.after_path ?? diff.path}</code>
        </div>
        <div
          className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`}
          ref={afterWrapRef}
          style={dualWrapStyle}
        >
          <CodeMirror
            value={alignedDiff.afterText}
            basicSetup={DIFF_CODEMIRROR_SETUP}
            editable={false}
            extensions={afterExtensions}
            height={dualAutoFitHeight}
          />
        </div>
      </div>
    </div>
  );
});