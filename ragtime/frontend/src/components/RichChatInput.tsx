import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ClipboardEvent as ReactClipboardEvent,
  type MutableRefObject,
} from 'react';

import type { ChatContextReference } from '@/types';

// Inline SVG markup mirroring lucide-react's `Code` and `X` icons so the
// imperatively-built chip DOM is self-contained.
const X_ICON_SVG =
  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>';

// A rich message composer document is an ordered list of segments: plain text
// runs interleaved with atomic inline context-reference chips. This lets the
// user weave references between the words they type, e.g.
//   "see <fileA:1-3> then look at <fileB:10>".
export type RichChatSegment =
  | { type: 'text'; text: string }
  | { type: 'ref'; reference: ChatContextReference };

export interface RichChatInputHandle {
  focus: () => void;
  insertReferenceAtCaret: (reference: ChatContextReference) => void;
}

interface RichChatInputProps {
  segments: RichChatSegment[];
  onChange: (segments: RichChatSegment[]) => void;
  onSubmit: () => void;
  // When false (default), Enter inserts a newline and never submits. Submission
  // is driven by an explicit send/save button instead.
  submitOnEnter?: boolean;
  onCancel?: () => void;
  onFocus?: () => void;
  onRemoveReference?: (referenceId: string) => void;
  onOpenReference?: (reference: ChatContextReference) => void | Promise<void>;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
  // Optional handle to the underlying editable element for layout measurement.
  elementRef?: MutableRefObject<HTMLDivElement | null>;
}

const REF_DATA_ATTR = 'data-ref-id';
const REF_PAYLOAD_ATTR = 'data-ref-payload';

function formatReferenceLabel(reference: ChatContextReference): string {
  const parts = reference.path.split('/').filter(Boolean);
  const filename = parts[parts.length - 1] || reference.path;
  if (!reference.startLine) {
    return filename;
  }
  const endLine = reference.endLine ?? reference.startLine;
  return endLine === reference.startLine
    ? `${filename}:${reference.startLine}`
    : `${filename}:${reference.startLine}-${endLine}`;
}

function formatReferenceLocation(reference: ChatContextReference): string {
  if (!reference.startLine) return reference.path;
  const endLine = reference.endLine ?? reference.startLine;
  return endLine === reference.startLine
    ? `${reference.path}:${reference.startLine}`
    : `${reference.path}:${reference.startLine}-${endLine}`;
}

export function segmentsToPlainText(segments: RichChatSegment[]): string {
  return segments.map((segment) => (segment.type === 'text' ? segment.text : '')).join('');
}

export function segmentsReferences(segments: RichChatSegment[]): ChatContextReference[] {
  const references: ChatContextReference[] = [];
  for (const segment of segments) {
    if (segment.type === 'ref') references.push(segment.reference);
  }
  return references;
}

export function segmentsAreEmpty(segments: RichChatSegment[]): boolean {
  return !segments.some(
    (segment) =>
      segment.type === 'ref' || (segment.type === 'text' && segment.text.trim().length > 0),
  );
}

export const EMPTY_RICH_SEGMENTS: RichChatSegment[] = [];

export function plainTextToSegments(text: string): RichChatSegment[] {
  return text ? [{ type: 'text', text }] : [];
}

// Serialize the composer document into the message string sent to the model.
// References appear inline (as `[file:lines]` tokens woven into the prose) and
// their code is appended once each in a trailing context block. Overlapping or
// duplicate references are de-duplicated by identity so context is only sent
// once even if the same chip is referenced multiple times in the text.
export function serializeRichChatSegments(segments: RichChatSegment[]): string {
  let inlineMessage = '';
  const seen = new Set<string>();
  const uniqueReferences: ChatContextReference[] = [];

  for (const segment of segments) {
    if (segment.type === 'text') {
      inlineMessage += segment.text;
      continue;
    }
    const reference = segment.reference;
    inlineMessage += `[${formatReferenceLocation(reference)}]`;
    if (!seen.has(reference.id)) {
      seen.add(reference.id);
      uniqueReferences.push(reference);
    }
  }

  const trimmedMessage = inlineMessage.trim();

  if (uniqueReferences.length === 0) {
    return trimmedMessage;
  }

  const blocks = uniqueReferences.map((reference) => {
    const location = formatReferenceLocation(reference);
    const content = reference.content ?? '';
    const truncatedNote = reference.contentTruncated
      ? '\n[Content truncated to context size cap.]'
      : '';
    if (!content.trim()) {
      return `- ${location}`;
    }
    return `- ${location}\n\`\`\`\n${content}${truncatedNote}\n\`\`\``;
  });

  const messageBody = trimmedMessage || 'Please use the referenced context.';
  return `${messageBody}\n\nContext references:\n${blocks.join('\n\n')}`;
}

// Normalize a segment list so adjacent text runs merge and empty text runs are
// dropped. References keep their order and identity.
function normalizeSegments(segments: RichChatSegment[]): RichChatSegment[] {
  const result: RichChatSegment[] = [];
  for (const segment of segments) {
    if (segment.type === 'text') {
      if (!segment.text) continue;
      const last = result[result.length - 1];
      if (last && last.type === 'text') {
        last.text += segment.text;
      } else {
        result.push({ type: 'text', text: segment.text });
      }
    } else {
      result.push(segment);
    }
  }
  return result;
}

export const RichChatInput = forwardRef<RichChatInputHandle, RichChatInputProps>(
  function RichChatInput(
    {
      segments,
      onChange,
      onSubmit,
      submitOnEnter = false,
      onCancel,
      onFocus,
      onRemoveReference,
      onOpenReference,
      placeholder,
      disabled = false,
      className,
      ariaLabel,
      elementRef,
    },
    ref,
  ) {
    const editorRef = useRef<HTMLDivElement | null>(null);
    // The most recent caret position inside the composer, stored as a logical
    // child-index of the editor (chips are atomic, so an index between top-level
    // nodes is a stable insertion point that survives re-renders). Lets reference
    // chips insert where the user last had their cursor even after focus moves to
    // the code editor (selecting code) or a file-tree action button.
    const savedCaretIndexRef = useRef<number | null>(null);
    const setEditorEl = useCallback(
      (node: HTMLDivElement | null) => {
        editorRef.current = node;
        if (elementRef) {
          elementRef.current = node;
        }
      },
      [elementRef],
    );

    // Compute the index among the editor's top-level child nodes at which a new
    // node should be inserted to land at the given range start.
    const caretIndexFromRange = useCallback((editor: HTMLDivElement, range: Range): number => {
      const { startContainer, startOffset } = range;
      if (startContainer === editor) {
        return startOffset;
      }
      // Find the top-level child that contains the start container.
      let topLevel: Node | null = startContainer;
      while (topLevel && topLevel.parentNode !== editor) {
        topLevel = topLevel.parentNode;
      }
      if (!topLevel) {
        return editor.childNodes.length;
      }
      const baseIndex = Array.prototype.indexOf.call(editor.childNodes, topLevel);
      // If the caret sits at the very start of a text node, insert before it;
      // otherwise (mid/after text, or after a chip) insert after it.
      if (startContainer.nodeType === Node.TEXT_NODE && startOffset === 0) {
        return baseIndex;
      }
      return baseIndex + 1;
    }, []);

    const rememberCaret = useCallback(() => {
      const editor = editorRef.current;
      if (!editor) return;
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0) return;
      const range = selection.getRangeAt(0);
      if (!editor.contains(range.startContainer)) return;
      savedCaretIndexRef.current = caretIndexFromRange(editor, range);
    }, [caretIndexFromRange]);
    // Tracks the segment list we last rendered into the DOM so we can avoid
    // clobbering the live editor (and the caret) on every keystroke.
    const renderedSegmentsRef = useRef<RichChatSegment[] | null>(null);
    const onChangeRef = useRef(onChange);
    const onSubmitRef = useRef(onSubmit);
    const onRemoveReferenceRef = useRef(onRemoveReference);
    const onOpenReferenceRef = useRef(onOpenReference);

    useEffect(() => {
      onChangeRef.current = onChange;
    }, [onChange]);
    useEffect(() => {
      onSubmitRef.current = onSubmit;
    }, [onSubmit]);
    useEffect(() => {
      onRemoveReferenceRef.current = onRemoveReference;
    }, [onRemoveReference]);
    useEffect(() => {
      onOpenReferenceRef.current = onOpenReference;
    }, [onOpenReference]);

    // Build the DOM for a chip token. The chip is non-editable so it behaves as
    // a single atomic unit the caret moves around.
    const buildChipElement = useCallback((reference: ChatContextReference): HTMLElement => {
      const chip = document.createElement('span');
      chip.className = 'chat-context-chip chat-context-chip-inline';
      chip.setAttribute('contenteditable', 'false');
      chip.setAttribute(REF_DATA_ATTR, reference.id);
      // Carry the full reference payload so DOM read-back can reconstruct chips
      // that were inserted imperatively (and aren't yet in the segment model).
      chip.setAttribute(REF_PAYLOAD_ATTR, JSON.stringify(reference));
      chip.setAttribute('title', formatReferenceLocation(reference));

      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'chat-context-chip-open';
      open.tabIndex = -1;
      open.addEventListener('mousedown', (event) => {
        event.preventDefault();
        event.stopPropagation();
        void onOpenReferenceRef.current?.(reference);
      });

      const label = document.createElement('span');
      label.className = 'chat-context-chip-label';
      label.textContent = formatReferenceLabel(reference);
      open.appendChild(label);

      if (reference.contentTruncated) {
        const meta = document.createElement('span');
        meta.className = 'chat-context-chip-meta';
        meta.textContent = 'truncated';
        open.appendChild(meta);
      }

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'chat-context-chip-remove';
      remove.tabIndex = -1;
      remove.setAttribute('aria-label', `Remove ${formatReferenceLabel(reference)}`);
      remove.addEventListener('mousedown', (event) => {
        event.preventDefault();
        event.stopPropagation();
        onRemoveReferenceRef.current?.(reference.id);
      });
      const removeIcon = document.createElement('span');
      removeIcon.className = 'chat-context-chip-remove-icon';
      removeIcon.innerHTML = X_ICON_SVG;
      remove.appendChild(removeIcon);

      chip.appendChild(open);
      chip.appendChild(remove);
      return chip;
    }, []);

    // Render the current segment model into the editor DOM. Used for external
    // value changes (programmatic clears, chip add/remove, conversation switch).
    const renderSegments = useCallback(
      (next: RichChatSegment[]) => {
        const editor = editorRef.current;
        if (!editor) return;

        editor.replaceChildren();
        for (const segment of next) {
          if (segment.type === 'text') {
            if (segment.text) {
              editor.appendChild(document.createTextNode(segment.text));
            }
          } else {
            editor.appendChild(buildChipElement(segment.reference));
          }
        }
        renderedSegmentsRef.current = next;
        // The DOM was rebuilt; keep the saved caret index within bounds so the next
        // out-of-focus insert lands at a valid top-level position.
        if (savedCaretIndexRef.current !== null) {
          savedCaretIndexRef.current = Math.min(
            savedCaretIndexRef.current,
            editor.childNodes.length,
          );
        }
      },
      [buildChipElement],
    );

    // Read the editor DOM back into a segment model.
    const readSegmentsFromDom = useCallback((): RichChatSegment[] => {
      const editor = editorRef.current;
      if (!editor) return [];

      const result: RichChatSegment[] = [];
      editor.childNodes.forEach((node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          result.push({ type: 'text', text: node.textContent ?? '' });
          return;
        }
        if (node.nodeType === Node.ELEMENT_NODE) {
          const element = node as HTMLElement;
          const refId = element.getAttribute(REF_DATA_ATTR);
          if (refId) {
            const payload = element.getAttribute(REF_PAYLOAD_ATTR);
            if (payload) {
              try {
                result.push({
                  type: 'ref',
                  reference: JSON.parse(payload) as ChatContextReference,
                });
                return;
              } catch {
                // Fall through to the previous-segment lookup below.
              }
            }
            const previous = (renderedSegmentsRef.current ?? []).find(
              (segment): segment is { type: 'ref'; reference: ChatContextReference } =>
                segment.type === 'ref' && segment.reference.id === refId,
            );
            if (previous) {
              result.push(previous);
            }
            return;
          }
          // <br>/<div> from contentEditable line breaks become newlines.
          if (element.tagName === 'BR') {
            result.push({ type: 'text', text: '\n' });
            return;
          }
          const text = element.textContent ?? '';
          if (text) {
            result.push({ type: 'text', text });
          }
        }
      });
      return normalizeSegments(result);
    }, []);

    const emitChange = useCallback(() => {
      const next = readSegmentsFromDom();
      renderedSegmentsRef.current = next;
      onChangeRef.current(next);
    }, [readSegmentsFromDom]);

    // Sync DOM when the segment model changes from outside (not from our own
    // input handler, which already mutated the DOM).
    useLayoutEffect(() => {
      if (renderedSegmentsRef.current === segments) {
        return;
      }
      renderSegments(segments);
    }, [renderSegments, segments]);

    const handleInput = useCallback(() => {
      const editor = editorRef.current;
      // When the user clears the composer (e.g. select-all + backspace), browsers
      // leave a stray <br> placeholder node behind. That <br> establishes a line
      // box on which the CSS ::before placeholder renders inline, painting the
      // caret after the placeholder text instead of at the far left. Strip it so
      // the empty editor matches its pristine (never-typed) state.
      if (editor && editor.childNodes.length === 1) {
        const only = editor.firstChild;
        if (only && only.nodeName === 'BR') {
          editor.removeChild(only);
          const selection = window.getSelection();
          if (selection) {
            const range = document.createRange();
            range.setStart(editor, 0);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
          }
        }
      }
      emitChange();
      rememberCaret();
    }, [emitChange, rememberCaret]);

    const handleKeyDown = useCallback(
      (event: ReactKeyboardEvent<HTMLDivElement>) => {
        if (disabled) return;
        if (submitOnEnter && event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          onSubmitRef.current();
          return;
        }
        if (onCancel && event.key === 'Escape') {
          event.preventDefault();
          onCancel();
        }
      },
      [disabled, onCancel, submitOnEnter],
    );

    // Track the caret so reference chips can target the user's last cursor
    // position even after focus shifts to the code editor or a file action.
    const handleSelectionTracked = useCallback(() => {
      rememberCaret();
    }, [rememberCaret]);

    const handleFocus = useCallback(() => {
      onFocus?.();
      rememberCaret();
    }, [onFocus, rememberCaret]);

    // Paste as plain text so we never inject foreign markup into the editor.
    const handlePaste = useCallback(
      (event: ReactClipboardEvent<HTMLDivElement>) => {
        event.preventDefault();
        const text = event.clipboardData.getData('text/plain');
        if (!text) return;
        const selection = window.getSelection();
        if (!selection || selection.rangeCount === 0) {
          document.execCommand('insertText', false, text);
        } else {
          const range = selection.getRangeAt(0);
          range.deleteContents();
          range.insertNode(document.createTextNode(text));
          range.collapse(false);
          selection.removeAllRanges();
          selection.addRange(range);
        }
        emitChange();
        rememberCaret();
      },
      [emitChange, rememberCaret],
    );

    const insertReferenceAtCaret = useCallback(
      (reference: ChatContextReference) => {
        const editor = editorRef.current;
        if (!editor) return;

        const selection = window.getSelection();
        const liveInEditor =
          !!selection && selection.rangeCount > 0 && editor.contains(selection.anchorNode);

        const chip = buildChipElement(reference);
        // A trailing space keeps the caret editable right after the chip and gives
        // visual separation from following text.
        const spacer = document.createTextNode(' ');

        if (liveInEditor) {
          // Focused composer: insert at the exact caret, splitting text if needed.
          const range = selection!.getRangeAt(0);
          range.deleteContents();
          range.insertNode(spacer);
          range.insertNode(chip);

          const after = document.createRange();
          after.setStartAfter(spacer);
          after.collapse(true);
          selection!.removeAllRanges();
          selection!.addRange(after);
        } else if (savedCaretIndexRef.current !== null) {
          // Focus moved away (e.g. selecting code or a file action), but the user
          // had placed a caret earlier — insert at that top-level node boundary.
          const index = Math.min(savedCaretIndexRef.current, editor.childNodes.length);
          const anchorNode = editor.childNodes[index] ?? null;
          editor.insertBefore(chip, anchorNode);
          editor.insertBefore(spacer, chip.nextSibling);
        } else {
          // The composer was never focused; require focusing first (no blind append).
          return;
        }

        // Remember the position right after the inserted chip so subsequent inserts
        // stack in order and later typing continues from there.
        savedCaretIndexRef.current = Array.prototype.indexOf.call(editor.childNodes, spacer) + 1;

        emitChange();
      },
      [buildChipElement, emitChange],
    );

    useImperativeHandle(
      ref,
      () => ({
        focus: () => editorRef.current?.focus(),
        insertReferenceAtCaret,
      }),
      [insertReferenceAtCaret],
    );

    const isEmpty = segmentsAreEmpty(segments);

    return (
      <div
        ref={setEditorEl}
        className={`chat-rich-input${isEmpty ? ' chat-rich-input-empty' : ''}${className ? ` ${className}` : ''}`}
        contentEditable={!disabled}
        role="textbox"
        aria-multiline="true"
        aria-label={ariaLabel}
        data-placeholder={placeholder}
        spellCheck
        suppressContentEditableWarning
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onKeyUp={handleSelectionTracked}
        onMouseUp={handleSelectionTracked}
        onFocus={handleFocus}
        onBlur={handleSelectionTracked}
        onPaste={handlePaste}
      >
        {/* Content is managed imperatively to support atomic inline chips. */}
      </div>
    );
  },
);

// Re-export chip formatting so callers (serialization) share the same shorthand.
export {
  formatReferenceLabel as formatContextReferenceLabel,
  formatReferenceLocation as formatContextReferenceLocation,
};
