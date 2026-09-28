import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { RichChatInput } from './RichChatInput';
const renderInput = (props: Partial<React.ComponentProps<typeof RichChatInput>> = {}) => {
  const onSubmit = vi.fn();
  render(
    <RichChatInput
      segments={[]}
      onChange={vi.fn()}
      onSubmit={onSubmit}
      ariaLabel="Message"
      {...props}
    />,
  );
  return { onSubmit, input: screen.getByRole('textbox', { name: 'Message' }) };
};
describe('RichChatInput keyboard submission', () => {
  afterEach(cleanup);
  it('keeps its legacy button-only default when sendMode is absent', () => {
    const { input, onSubmit } = renderInput();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).not.toHaveBeenCalled();
  });
  it('uses sendMode in preference to the legacy submitOnEnter prop', () => {
    const { input, onSubmit } = renderInput({ submitOnEnter: true, sendMode: 'ctrl-enter' });
    fireEvent.keyDown(input, { key: 'Enter' });
    fireEvent.keyDown(input, { key: 'Enter', ctrlKey: true });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
  it('does not submit composition, repeated, Shift, or Alt Enter events', () => {
    const { input, onSubmit } = renderInput({ sendMode: 'enter' });
    fireEvent.keyDown(input, { key: 'Enter', isComposing: true });
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 });
    fireEvent.keyDown(input, { key: 'Enter', repeat: true });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    fireEvent.keyDown(input, { key: 'Enter', altKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe('RichChatInput paste handling', () => {
  afterEach(cleanup);

  it('reports plain pasted text after inserting it', () => {
    const onChange = vi.fn();
    const onPasteText = vi.fn();
    const { input } = renderInput({ onChange, onPasteText });
    const range = document.createRange();
    range.selectNodeContents(input);
    range.collapse(false);
    window.getSelection()?.removeAllRanges();
    window.getSelection()?.addRange(range);

    fireEvent.paste(input, {
      clipboardData: { getData: () => 'Pasted plain text' },
    });

    expect(onChange).toHaveBeenLastCalledWith([{ type: 'text', text: 'Pasted plain text' }]);
    expect(onPasteText).toHaveBeenCalledWith('Pasted plain text');
    expect(onChange.mock.invocationCallOrder[0]).toBeLessThan(
      onPasteText.mock.invocationCallOrder[0],
    );
  });
});
