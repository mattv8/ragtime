import { render } from '@testing-library/react';
import CodeMirror, { EditorView } from '@uiw/react-codemirror';
import { undo, redo } from '@codemirror/commands';
import { afterEach, describe, expect, it, vi } from 'vitest';

(Range.prototype as unknown as { getClientRects: () => unknown[] }).getClientRects = () => [];
(Range.prototype as unknown as { getBoundingClientRect: () => unknown }).getBoundingClientRect =
  () => ({
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });

describe('UserSpace editor history', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('undo after a programmatic value swap emits the previous file content via onChange', () => {
    let view: EditorView | null = null;
    const onChange = vi.fn();
    const props = { onChange, onCreateEditor: (v: EditorView) => (view = v) };
    const t0 = Date.now();
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(t0);
    const { rerender } = render(<CodeMirror value={'PACKAGE_LOCK'} {...props} />);
    now.mockReturnValue(t0 + 5000);
    rerender(<CodeMirror value={'INDEX_HTML'} {...props} />);
    now.mockReturnValue(t0 + 10000);
    rerender(<CodeMirror value={'GITIGNORE'} {...props} />);
    expect(onChange).not.toHaveBeenCalled();
    undo(view!);
    undo(view!);
    redo(view!);
    redo(view!);
    expect(onChange.mock.calls.map((c) => c[0])).toEqual([
      'INDEX_HTML',
      'PACKAGE_LOCK',
      'INDEX_HTML',
      'GITIGNORE',
    ]);
  });

  it('does not restore a previous file after switching keyed editors', () => {
    let view: EditorView | null = null;
    const onChange = vi.fn();
    const props = { onChange, onCreateEditor: (v: EditorView) => (view = v) };
    const { rerender } = render(
      <CodeMirror key="workspace-a:package-lock" value="PACKAGE_LOCK" {...props} />,
    );

    rerender(<CodeMirror key="workspace-a:index.html" value="INDEX_HTML" {...props} />);

    undo(view!);

    expect(onChange).not.toHaveBeenCalled();
  });
});
