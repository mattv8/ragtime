import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FetchableModelField } from './FetchableModelField';

describe('FetchableModelField', () => {
  const props = {
    idPrefix: 'test-model',
    label: 'Model',
    value: 'pinned-model',
    onChange: vi.fn(),
    models: [{ id: 'listed-model', label: 'Listed model' }],
    loaded: false,
    fetching: false,
    error: null,
    onFetch: vi.fn(),
    fetchDisabled: false,
    placeholder: 'model-id',
    help: 'Choose a model.',
  };
  it('uses an input before loading and a select after loading while preserving an unlisted value', () => {
    const { rerender } = render(<FetchableModelField {...props} />);
    expect((screen.getByRole('textbox', { name: 'Model' }) as HTMLInputElement).value).toBe(
      'pinned-model',
    );
    rerender(<FetchableModelField {...props} loaded />);
    expect((screen.getByRole('combobox', { name: 'Model' }) as HTMLSelectElement).value).toBe(
      'pinned-model',
    );
    expect(screen.getByRole('option', { name: 'pinned-model' })).toBeTruthy();
  });
  it('reports fetch state, errors, and disabled state through the field controls', async () => {
    const user = userEvent.setup();
    render(
      <FetchableModelField {...props} fetching error="Could not load models." fetchDisabled />,
    );
    const button = screen.getByRole('button', { name: 'Fetching...' });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    expect(button.className).toContain('btn-test');
    const error = screen.getByRole('alert');
    expect(error.textContent).toContain('Could not load models.');
    expect(error.id).toBe('test-model-error');
    expect(screen.getByRole('textbox', { name: 'Model' }).getAttribute('aria-describedby')).toBe(
      'test-model-help test-model-error',
    );
    expect(button.getAttribute('aria-describedby')).toBe('test-model-help test-model-error');
    await user.click(button);
    expect(props.onFetch).not.toHaveBeenCalled();
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
