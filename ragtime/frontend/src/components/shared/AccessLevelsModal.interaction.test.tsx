import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { contentProtectionApi, type ContentProtectionConfig } from '@/api/contentProtection';
import { AccessLevelsModal } from './AccessLevelsModal';

const config = (): ContentProtectionConfig => ({
  schema_version: 2,
  revision: 1,
  enabled: true,
  share_with_assistant: true,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: '' }, llm_model: null },
  strictness: 'balanced',
  categories: [],
  access_levels: [
    { id: 'standard', name: 'Standard', granted_category_ids: [], guidance: '' },
    { id: 'finance', name: 'Finance', granted_category_ids: [], guidance: '' },
  ],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'all_supported_traffic',
  requirements: [],
  user_overrides: [],
});

const toast = { success: vi.fn(), error: vi.fn() };

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue(config());
  vi.spyOn(contentProtectionApi, 'preview').mockResolvedValue({
    prompt_fragment: 'preview',
  } as never);
});

afterEach(cleanup);

describe('AccessLevelsModal interactions', () => {
  it('closes the detail before the modal when Escape is pressed', async () => {
    const onClose = vi.fn();
    render(<AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />);
    await screen.findByLabelText('Name');

    document.getElementById('access-level-detail-title')?.focus();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByLabelText('Name')).toBeNull();
    expect(onClose).not.toHaveBeenCalled();

    screen.getByLabelText('Search access levels').focus();
    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('closes from a clean overlay click and guards a dirty overlay click', async () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />,
    );
    await screen.findByLabelText('Name');
    fireEvent.mouseDown(document.getElementById('access-levels-modal-overlay')!);
    expect(onClose).toHaveBeenCalledOnce();

    rerender(<AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />);
    await userEvent.type(await screen.findByLabelText('Name'), ' changed');
    fireEvent.mouseDown(document.getElementById('access-levels-modal-overlay')!);
    expect(screen.getByRole('alertdialog', { name: 'Unsaved changes' })).toBeTruthy();
  });

  it('restores focus to the launcher after close', async () => {
    const Launcher = () => {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Launcher
          </button>
          <AccessLevelsModal
            open={open}
            onClose={() => setOpen(false)}
            authGroups={[]}
            toast={toast}
          />
        </>
      );
    };
    render(<Launcher />);
    const launcher = screen.getByRole('button', { name: 'Launcher' });
    await userEvent.click(launcher);
    await screen.findByLabelText('Name');
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    await waitFor(() => expect(document.activeElement).toBe(launcher));
  });

  it('moves initial focus to the detail heading when the auto-selected rail search is hidden', async () => {
    render(<AccessLevelsModal open onClose={vi.fn()} authGroups={[]} toast={toast} />);
    await screen.findByLabelText('Search access levels');
    const detailHeading = screen.getByRole('heading', { name: 'Standard' });
    await waitFor(() => expect(document.activeElement).toBe(detailHeading));
    expect(
      document.getElementById('manage-access-levels-modal')?.contains(document.activeElement),
    ).toBe(true);
  });

  it('traps Tab from the preview summary back to the first modal control', async () => {
    render(<AccessLevelsModal open onClose={vi.fn()} authGroups={[]} toast={toast} />);
    const summary = await screen.findByText('Prompt preview');
    summary.focus();
    fireEvent.keyDown(document.getElementById('manage-access-levels-modal')!, { key: 'Tab' });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close' }));
  });

  it('handles Escape and traps Tab in loading and error shells', async () => {
    let resolveConfig!: (value: ContentProtectionConfig) => void;
    vi.spyOn(contentProtectionApi, 'getConfig').mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveConfig = resolve;
        }),
    );
    const onClose = vi.fn();
    const loadingRender = render(
      <AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />,
    );
    const loadingDialog = screen.getByRole('dialog');
    const close = screen.getByRole('button', { name: 'Close' });
    close.focus();
    fireEvent.keyDown(loadingDialog, { key: 'Tab' });
    expect(document.activeElement).toBe(close);
    fireEvent.keyDown(loadingDialog, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledOnce();
    resolveConfig(config());
    loadingRender.unmount();

    vi.spyOn(contentProtectionApi, 'getConfig').mockRejectedValueOnce(new Error('offline'));
    const errorClose = vi.fn();
    render(<AccessLevelsModal open onClose={errorClose} authGroups={[]} toast={toast} />);
    const errorDialog = await screen.findByRole('dialog');
    await screen.findByRole('alert');
    fireEvent.keyDown(errorDialog, { key: 'Escape' });
    expect(errorClose).toHaveBeenCalledOnce();
  });

  it('reopens with fresh query and the default selected after closing a new draft', async () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />,
    );
    await screen.findByDisplayValue('Standard');
    await userEvent.type(screen.getByLabelText('Search access levels'), 'finance');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    rerender(<AccessLevelsModal open={false} onClose={onClose} authGroups={[]} toast={toast} />);
    rerender(<AccessLevelsModal open onClose={onClose} authGroups={[]} toast={toast} />);
    expect(await screen.findByDisplayValue('Standard')).toBeTruthy();
    expect((screen.getByLabelText('Search access levels') as HTMLInputElement).value).toBe('');
  });

  it('retries a failed preview request', async () => {
    vi.spyOn(contentProtectionApi, 'preview')
      .mockRejectedValueOnce(new Error('preview failed'))
      .mockResolvedValueOnce({ prompt_fragment: 'retried preview' } as never);
    render(<AccessLevelsModal open onClose={vi.fn()} authGroups={[]} toast={toast} />);

    expect((await screen.findByRole('alert')).textContent).toContain('preview failed');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(await screen.findByText('retried preview')).toBeTruthy();
    expect(contentProtectionApi.preview).toHaveBeenCalledTimes(2);
  });
});
