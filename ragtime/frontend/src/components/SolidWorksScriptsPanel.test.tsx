import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { contentProtectionApi } from '../api/contentProtection';
import { SolidWorksScriptsPanel, SolidWorksScriptsReview } from './SolidWorksScriptsPanel';

vi.mock('../api/contentProtection', () => ({ contentProtectionApi: { getConfig: vi.fn() } }));
const getConfig = vi.mocked(contentProtectionApi.getConfig);
const onChange = vi.fn();
const config = (enabled: boolean) => ({ enabled }) as Awaited<ReturnType<typeof getConfig>>;
const panel = (
  value = false,
  instanceMode: 'background' | 'attach' = 'background',
  disabled = false,
) => (
  <SolidWorksScriptsPanel
    value={value}
    onChange={onChange}
    instanceMode={instanceMode}
    disabled={disabled}
  />
);

beforeEach(() => {
  vi.clearAllMocks();
  getConfig.mockResolvedValue(config(true));
});
afterEach(cleanup);

describe('SolidWorks scripts', () => {
  it('defaults off and explains the risks without a PDM link', async () => {
    render(panel());
    expect(
      (screen.getByRole('switch', { name: 'Allow scripts' }) as HTMLInputElement).checked,
    ).toBe(false);
    expect(screen.getByText('Scripts')).toBeTruthy();
    expect(screen.getByText('Off: the AI cannot run scripts')).toBeTruthy();
    await userEvent.click(screen.getByRole('button', { name: 'About scripts' }));
    const help = document.getElementById('solidworks-allow-scripts-help-content')!.textContent!;
    for (const text of [
      'not sandboxed',
      'Windows user',
      'network',
      'Content protection',
      'attach mode',
      'coverage rules include this tool',
    ])
      expect(help).toContain(text);
    expect(getConfig).not.toHaveBeenCalled();
  });

  it('toggles without moving focus', () => {
    const { rerender } = render(panel());
    const checkbox = screen.getByLabelText('Allow scripts');
    checkbox.focus();
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith(true);
    expect(document.activeElement).toBe(checkbox);
    rerender(panel(true));
    expect(document.getElementById('solidworks-allow-scripts-summary')?.textContent).toContain(
      'not sandboxed',
    );
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('toggles with Space without a confirmation and retains focus', async () => {
    const confirm = vi.spyOn(window, 'confirm');
    function Harness() {
      const [value, setValue] = useState(false);
      return <SolidWorksScriptsPanel value={value} onChange={setValue} instanceMode="background" />;
    }
    render(<Harness />);
    const checkbox = screen.getByLabelText('Allow scripts') as HTMLInputElement;
    checkbox.focus();
    await userEvent.keyboard(' ');
    expect(checkbox.checked).toBe(true);
    expect(document.activeElement).toBe(checkbox);
    expect(confirm).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).toBeNull();
    confirm.mockRestore();
  });

  it('shows the protection warning and only rendered descriptions', async () => {
    getConfig.mockResolvedValue(config(false));
    render(panel(true, 'attach'));
    const warning = await screen.findByText(
      'Content protection is off, so scripts run without review.',
    );
    expect(warning.closest('#solidworks-allow-scripts-warning')).toBeTruthy();
    const checkbox = screen.getByLabelText('Allow scripts');
    expect(checkbox.getAttribute('aria-describedby')).toBe(
      'solidworks-allow-scripts-summary solidworks-allow-scripts-attach solidworks-allow-scripts-warning',
    );
    expect(screen.getByRole('status').contains(warning)).toBe(true);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows no message when protection is on', async () => {
    render(panel(true));
    await waitFor(() => expect(getConfig).toHaveBeenCalledTimes(1));
    expect(document.getElementById('solidworks-allow-scripts-warning')).toBeNull();
    expect(document.getElementById('solidworks-allow-scripts-unknown')).toBeNull();
  });

  it.each(['403', 'network'])('shows a neutral note on %s failure', async (error) => {
    getConfig.mockRejectedValue(new Error(error));
    render(panel(true));
    expect(
      await screen.findByText(/Ragtime could not check whether content protection is on/),
    ).toBeTruthy();
    expect(document.getElementById('solidworks-allow-scripts-warning')).toBeNull();
  });

  it('shows the warning only after a pending lookup resolves', async () => {
    let resolve!: (value: ReturnType<typeof config>) => void;
    getConfig.mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    render(panel(true));
    expect(document.getElementById('solidworks-allow-scripts-warning')).toBeNull();
    expect(document.getElementById('solidworks-allow-scripts-unknown')).toBeNull();
    resolve(config(false));
    expect(
      await screen.findByText('Content protection is off, so scripts run without review.'),
    ).toBeTruthy();
  });

  it('requests once per mounted panel', async () => {
    const { rerender } = render(panel(true));
    rerender(panel(true, 'attach'));
    rerender(panel(false));
    rerender(panel(true));
    await waitFor(() => expect(getConfig).toHaveBeenCalledTimes(1));
  });

  it('shows attach note only when enabled in attach mode', () => {
    const { rerender } = render(panel(true, 'attach'));
    expect(document.getElementById('solidworks-allow-scripts-attach')).toBeTruthy();
    rerender(panel(true));
    expect(document.getElementById('solidworks-allow-scripts-attach')).toBeNull();
    rerender(panel(false, 'attach'));
    expect(document.getElementById('solidworks-allow-scripts-attach')).toBeNull();
  });

  it('keeps messages visible while disabled', async () => {
    getConfig.mockResolvedValue(config(false));
    render(panel(true, 'background', true));
    expect(screen.getByLabelText('Allow scripts').matches(':disabled')).toBe(true);
    expect(
      await screen.findByText('Content protection is off, so scripts run without review.'),
    ).toBeTruthy();
  });

  it('reviews both states and warns when protection is off', async () => {
    getConfig.mockResolvedValue(config(false));
    const { rerender } = render(<SolidWorksScriptsReview allowed={false} />);
    expect(screen.getByText('Scripts: Off')).toBeTruthy();
    rerender(<SolidWorksScriptsReview allowed />);
    expect(screen.getByText(/Scripts: Allowed/)).toBeTruthy();
    expect(
      await screen.findByText('Content protection is off, so scripts run without review.'),
    ).toBeTruthy();
  });

  it('shows the unknown note in the review row when the lookup fails', async () => {
    getConfig.mockRejectedValue(new Error('403'));
    render(<SolidWorksScriptsReview allowed />);
    const note = await screen.findByText(
      /Ragtime could not check whether content protection is on/,
    );
    expect(note.closest('#tool-review-solidworks-scripts')).toBeTruthy();
    expect(
      screen.queryByText('Content protection is off, so scripts run without review.'),
    ).toBeNull();
  });
});
