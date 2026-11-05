import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useRef, useState } from 'react';

import { ToolDeleteDialog } from './ToolDeleteDialog';

describe('ToolDeleteDialog', () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
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
    await user.click(screen.getByRole('button', { name: 'Copy command' }));
    expect(clipboard.writeText).toHaveBeenCalledWith('powershell.exe -File remove.ps1');
    expect(await screen.findByRole('status', { name: /Command copied/i })).toBeTruthy();
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
