import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatSendModeControl } from './ChatSendModeControl';
describe('ChatSendModeControl', () => {
  afterEach(cleanup);
  it('uses a labelled native select with the three send modes', () => {
    const onChange = vi.fn();
    render(<ChatSendModeControl mode="ctrl-enter" onChange={onChange} />);
    const select = screen.getByLabelText('Send behavior');
    expect((select as HTMLSelectElement).value).toBe('ctrl-enter');
    expect(screen.getAllByRole('option')).toHaveLength(3);
    fireEvent.change(select, { target: { value: 'enter' } });
    expect(onChange).toHaveBeenCalledWith('enter');
  });
  it('honors disabled state', () => {
    render(<ChatSendModeControl mode="button" onChange={vi.fn()} disabled />);
    expect((screen.getByLabelText('Send behavior') as HTMLSelectElement).disabled).toBe(true);
  });
});
