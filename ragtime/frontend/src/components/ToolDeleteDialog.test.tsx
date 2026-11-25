import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useRef, useState } from 'react';

import { ToolDeleteDialog } from './ToolDeleteDialog';

describe('ToolDeleteDialog', () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('lists each deletion step with a labeled status', async () => {
    const user = userEvent.setup();
    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'pairing_only',
            steps: { pairing_cleanup: 'unconfirmed', ssh_service: 'not_requested' },
            warnings: ['Removal finalizer was launched.', 'Removal finalizer was launched.'],
            local_removal_command: 'powershell.exe -File remove.ps1',
          },
        })}
        onClose={vi.fn()}
      />,
    );
    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    const steps = within(await screen.findByRole('list', { name: 'Deletion steps' })).getAllByRole(
      'listitem',
    );
    expect(steps.map((step) => step.textContent)).toEqual([
      'Remove tool from RagtimeDone',
      'Pairing cleanupNot confirmed',
      'SSH serviceNot requested',
      'Finish cleanup on the hostRun the command below as an Administrator.Required',
    ]);
    expect(screen.getByText('Manual cleanup needed.')).toBeTruthy();
    expect(document.querySelector('#tool-delete-dialog-description')?.getAttribute('role')).toBe(
      'alert',
    );
    expect(screen.getAllByText('Removal finalizer was launched.')).toHaveLength(1);
  });

  it('confirms a copied command on the copy button', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'pairing_only',
            steps: { pairing_cleanup: 'unconfirmed', ssh_service: 'not_requested' },
            warnings: [],
            local_removal_command: 'powershell.exe -File remove.ps1',
          },
        })}
        onClose={vi.fn()}
      />,
    );
    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    const copyButton = await screen.findByRole('button', { name: 'Copy command' });
    expect((screen.getByLabelText('PowerShell command') as HTMLElement).tabIndex).toBe(0);
    expect(document.activeElement).toBe(copyButton);
    await user.click(copyButton);
    expect(screen.getByRole('button', { name: 'Copied' })).toBe(copyButton);
    const status = screen.getByText('Command copied.');
    expect(status.getAttribute('role')).toBe('status');
  });

  it('marks tool removal unconfirmed when the delete request fails', async () => {
    const user = userEvent.setup();
    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockRejectedValue(new Error('Network error'))}
        onClose={vi.fn()}
      />,
    );
    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    const steps = within(await screen.findByRole('list', { name: 'Deletion steps' })).getAllByRole(
      'listitem',
    );
    expect(steps.map((step) => step.textContent)).toEqual([
      'Remove tool from RagtimeNot confirmed',
    ]);
    expect(screen.getByText('Delete failed.')).toBeTruthy();
  });

  it('requires acknowledgement before disabling SSH and sends the confirmed cleanup mode', async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn().mockResolvedValue({
      cleanup: {
        mode: 'disable_ssh',
        steps: { pairing_cleanup: 'confirmed', ssh_service: 'unconfirmed' },
        warnings: [],
        local_removal_command: 'powershell.exe -File remove.ps1',
      },
    });

    render(<ToolDeleteDialog toolName="Windows host" onDelete={onDelete} onClose={vi.fn()} />);

    const deleteButton = screen.getByRole('button', { name: 'Delete tool' });
    expect((deleteButton as HTMLButtonElement).disabled).toBe(true);

    await user.click(
      screen.getByLabelText(
        /I understand this stops SSH for all users and other Ragtime connections/i,
      ),
    );
    await user.click(deleteButton);

    expect(onDelete).toHaveBeenCalledWith({
      sshCleanup: 'disable_ssh',
      confirmMachineWide: true,
    });
    expect(await screen.findByRole('heading', { name: 'Tool deletion result' })).toBeTruthy();
    expect(screen.getAllByText(/Not confirmed/i)).toHaveLength(2);
    expect(screen.getByText('powershell.exe -File remove.ps1')).toBeTruthy();
  });

  it('reports confirmed cleanup as a success without a manual cleanup step', async () => {
    const user = userEvent.setup();
    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'disable_ssh',
            steps: { pairing_cleanup: 'confirmed', ssh_service: 'confirmed' },
            warnings: [],
            local_removal_command: '',
          },
        })}
        onClose={vi.fn()}
      />,
    );
    await user.click(
      screen.getByLabelText(
        /I understand this stops SSH for all users and other Ragtime connections/i,
      ),
    );
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    expect(await screen.findByText('The tool was deleted.')).toBeTruthy();
    expect(document.querySelector('#tool-delete-dialog-description')?.getAttribute('role')).toBe(
      'status',
    );
    expect(screen.queryByText('Manual cleanup needed.')).toBeNull();
    expect(document.querySelector('[data-tool-delete-step="manual-cleanup"]')).toBeNull();
  });

  it('allows pairing-only removal without machine-wide acknowledgement', async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn().mockResolvedValue({});

    render(<ToolDeleteDialog toolName="Windows host" onDelete={onDelete} onClose={vi.fn()} />);

    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    expect(onDelete).toHaveBeenCalledWith({ sshCleanup: 'pairing_only' });
  });

  it('treats a successful response without requested cleanup details as unconfirmed', async () => {
    const user = userEvent.setup();

    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({})}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    expect(await screen.findByText(/Host cleanup was requested but not confirmed/i)).toBeTruthy();
    expect(screen.getByText(/Complete cleanup manually on the Windows host/i)).toBeTruthy();
    expect(
      document.querySelector('[data-tool-delete-step="manual-cleanup"]')?.textContent,
    ).toContain('did not return host cleanup details');
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });

  it('renders non-runnable recovery guidance as instructions rather than a command', async () => {
    const user = userEvent.setup();
    const localRemovalInstructions = 'Use the Windows Settings app to remove the paired client.';

    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'pairing_only',
            steps: { pairing_cleanup: 'unconfirmed', ssh_service: 'not_requested' },
            warnings: [],
            local_removal_command: '',
            local_removal_instructions: localRemovalInstructions,
          },
        })}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    expect(await screen.findByText('Manual cleanup instructions:')).toBeTruthy();
    expect(screen.getByText(localRemovalInstructions)).toBeTruthy();
    expect(
      document.querySelector('[data-tool-delete-step="manual-cleanup"]')?.textContent,
    ).toContain('Follow the instructions below.');
    expect(document.querySelector('[data-tool-delete-recovery-instructions]')).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });

  it('moves focus to working and result controls after the prior phase unmounts', async () => {
    const user = userEvent.setup();
    let resolveDeletion!: (result: object) => void;
    const onDelete = vi.fn(
      () =>
        new Promise<object>((resolve) => {
          resolveDeletion = resolve;
        }),
    );

    render(<ToolDeleteDialog toolName="Windows host" onDelete={onDelete} onClose={vi.fn()} />);

    await user.click(
      screen.getByLabelText(
        /I understand this stops SSH for all users and other Ragtime connections/i,
      ),
    );
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    const workingStatus = await screen.findByRole('status');
    expect(document.activeElement).toBe(workingStatus);
    await user.tab();
    expect(document.activeElement).toBe(workingStatus);
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(workingStatus);

    resolveDeletion({});
    const dismissButton = await screen.findByRole('button', { name: 'Dismiss' });
    await waitFor(() => expect(document.activeElement).toBe(dismissButton));
  });

  it('focuses the copy command button when a recovery command is shown', async () => {
    const user = userEvent.setup();
    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'pairing_only',
            steps: { pairing_cleanup: 'unconfirmed', ssh_service: 'not_requested' },
            warnings: [],
            local_removal_command: 'powershell.exe -File remove.ps1',
          },
        })}
        onClose={vi.fn()}
      />,
    );
    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    const copyButton = await screen.findByRole('button', { name: 'Copy command' });
    await waitFor(() => expect(document.activeElement).toBe(copyButton));
  });

  it('bounds stalled requests at 45 seconds without retrying or accepting late success', async () => {
    vi.useFakeTimers();
    let resolveDeletion!: (result: object) => void;
    const onDelete = vi.fn(
      () =>
        new Promise<object>((resolve) => {
          resolveDeletion = resolve;
        }),
    );
    render(<ToolDeleteDialog toolName="Windows host" onDelete={onDelete} onClose={vi.fn()} />);
    fireEvent.click(screen.getByLabelText(/Remove Ragtime only/i));
    fireEvent.click(screen.getByRole('button', { name: 'Delete tool' }));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(44_999);
    });
    expect(screen.getByText(/Deleting the tool and requesting host cleanup/i)).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(screen.getByText(/It may still be running on the server/i)).toBeTruthy();
    expect(screen.getByText(/could not confirm deletion or host cleanup status/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Dismiss' })).toBeTruthy();

    await act(async () => {
      resolveDeletion({
        cleanup: {
          mode: 'pairing_only',
          steps: { pairing_cleanup: 'confirmed', ssh_service: 'not_requested' },
          warnings: [],
        },
      });
    });
    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/could not confirm deletion or host cleanup status/i)).toBeTruthy();
    expect(screen.queryByText('The tool was deleted.')).toBeNull();
  });

  it('restores fallback focus when the deletion opener was removed', async () => {
    const user = userEvent.setup();

    function Harness() {
      const [open, setOpen] = useState(true);
      const fallbackRef = useRef<HTMLButtonElement>(null);
      return (
        <>
          <button ref={fallbackRef} type="button">
            Tool connections
          </button>
          {open && (
            <ToolDeleteDialog
              toolName="Windows host"
              onDelete={vi.fn()}
              onClose={() => setOpen(false)}
              fallbackFocusRef={fallbackRef}
            />
          )}
        </>
      );
    }

    render(<Harness />);
    await user.click(screen.getByRole('button', { name: 'Close' }));

    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Tool connections' }));
  });

  it('keeps recovery instructions after cleanup or deletion failures and reports copy outcomes', async () => {
    const user = userEvent.setup();
    const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) };
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: clipboard });
    const onDelete = vi.fn().mockRejectedValue({
      structuredDetail: {
        message: 'Database deletion failed.',
        cleanup: {
          mode: 'disable_ssh',
          steps: { pairing_cleanup: 'failed', ssh_service: 'unconfirmed' },
          warnings: ['The remote host did not confirm cleanup.'],
          local_removal_command: 'powershell.exe -File remove.ps1',
        },
      },
    });

    render(<ToolDeleteDialog toolName="Windows host" onDelete={onDelete} onClose={vi.fn()} />);
    await user.click(
      screen.getByLabelText(
        /I understand this stops SSH for all users and other Ragtime connections/i,
      ),
    );
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));

    expect(await screen.findByText(/Database deletion failed/i)).toBeTruthy();
    expect(screen.getByText(/Host changes may already have been attempted/i)).toBeTruthy();
    expect(screen.getByText('The remote host did not confirm cleanup.')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Copy command' }));
    expect(clipboard.writeText).toHaveBeenCalledWith('powershell.exe -File remove.ps1');
    expect((await screen.findByText('Command copied.')).getAttribute('role')).toBe('status');
  });

  it('keeps the recovery command visible when copying fails', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockRejectedValue(new Error('blocked')) },
    });

    render(
      <ToolDeleteDialog
        toolName="Windows host"
        onDelete={vi.fn().mockResolvedValue({
          cleanup: {
            mode: 'pairing_only',
            steps: { pairing_cleanup: 'unconfirmed', ssh_service: 'not_requested' },
            warnings: [],
            local_removal_command: 'powershell.exe -File remove.ps1',
          },
        })}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText(/Remove Ragtime only/i));
    await user.click(screen.getByRole('button', { name: 'Delete tool' }));
    await user.click(await screen.findByRole('button', { name: 'Copy command' }));

    expect(await screen.findByText(/Could not copy the command/i)).toBeTruthy();
    expect(screen.getByText('powershell.exe -File remove.ps1')).toBeTruthy();
  });
});
