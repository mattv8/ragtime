import { useEffect, useMemo, useRef, memo } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { Decoration, type DecorationSet, EditorView } from '@codemirror/view';
import { StateField, type Extension } from '@codemirror/state';
import { diffLines } from 'diff';
import type { UserSpaceSnapshotFileDiff } from '@/types';
import { useCodeMirrorLanguageExtension } from '@/utils/codemirrorLanguage';

interface AlignedDiffResult {
  beforeText: string;
  afterText: string;
  beforeDeletedLines: Set<number>;
  afterAddedLines: Set<number>;
  beforePaddingLines: Set<number>;
  afterPaddingLines: Set<number>;
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
  lineNumbers: true,
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
}

export const UserSpaceFileDiffView = memo(function UserSpaceFileDiffView({
  diff,
  beforeLabel,
  afterLabel,
  compact = false,
  syncScroll = true,
  highlightSingleColumnChanges = true,
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
    const singleExtensions = highlightSingleColumnChanges
      ? (isPureAdd ? afterExtensions : beforeExtensions)
      : languageExtensions;

    return (
      <div className={`userspace-snapshot-diff-columns userspace-snapshot-diff-columns-single${compact ? ' userspace-snapshot-diff-columns-compact' : ''}`}>
        <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
          <div className="userspace-snapshot-diff-column-header">
            <span>{singleLabel}</span>
            <code>{singlePath}</code>
          </div>
          <div className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`}>
            <CodeMirror
              value={singleContent}
              basicSetup={DIFF_CODEMIRROR_SETUP}
              editable={false}
              extensions={singleExtensions}
              height="100%"
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`userspace-snapshot-diff-columns${compact ? ' userspace-snapshot-diff-columns-compact' : ''}`}>
      <div className="userspace-snapshot-diff-column">
        <div className="userspace-snapshot-diff-column-header">
          <span>{beforeLabel}</span>
          <code>{diff.before_path ?? diff.path}</code>
        </div>
        <div className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`} ref={beforeWrapRef}>
          <CodeMirror
            value={alignedDiff.beforeText}
            basicSetup={DIFF_CODEMIRROR_SETUP}
            editable={false}
            extensions={beforeExtensions}
            height="100%"
          />
        </div>
      </div>
      <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
        <div className="userspace-snapshot-diff-column-header">
          <span>{afterLabel}</span>
          <code>{diff.after_path ?? diff.path}</code>
        </div>
        <div className={`userspace-snapshot-diff-editor-wrap${compact ? ' userspace-snapshot-diff-editor-wrap-compact' : ''}`} ref={afterWrapRef}>
          <CodeMirror
            value={alignedDiff.afterText}
            basicSetup={DIFF_CODEMIRROR_SETUP}
            editable={false}
            extensions={afterExtensions}
            height="100%"
          />
        </div>
      </div>
    </div>
  );
});