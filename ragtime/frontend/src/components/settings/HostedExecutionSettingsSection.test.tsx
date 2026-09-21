import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { UpdateSettingsRequest } from '@/types';

import { HostedExecutionSettingsSection } from './HostedExecutionSettingsSection';

afterEach(() => {
  cleanup();
});

function renderSection({
  formData = {},
  hostedExecutionSaving = false,
  handleSaveHostedExecution = vi.fn(),
}: {
  formData?: UpdateSettingsRequest;
  hostedExecutionSaving?: boolean;
  handleSaveHostedExecution?: () => void | Promise<void>;
} = {}) {
  function Wrapper(): JSX.Element {
    const [currentFormData, setCurrentFormData] = useState<UpdateSettingsRequest>(formData);

    return (
      <HostedExecutionSettingsSection
        open
        onToggle={() => {}}
        formData={currentFormData}
        setFormData={setCurrentFormData}
        handleSaveHostedExecution={handleSaveHostedExecution}
        hostedExecutionSaving={hostedExecutionSaving}
      />
    );
  }

  return render(<Wrapper />);
}

describe('HostedExecutionSettingsSection', () => {
  it('defaults to enabled and explains the inherited effective policy', () => {
    renderSection();

    const checkbox = screen.getByLabelText('Enable hosted chat') as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    expect(screen.getByText(/enabled for users without an individual override/i)).toBeTruthy();
  });

  it('renders the controlled disabled policy and updates it from the checkbox', async () => {
    const user = userEvent.setup();
    renderSection({ formData: { hosted_chat_enabled: false } });

    const checkbox = screen.getByLabelText('Enable hosted chat') as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    expect(
      screen.getByText(/unavailable globally.*individual user overrides cannot re-enable it/i),
    ).toBeTruthy();

    await user.click(checkbox);
    expect(checkbox.checked).toBe(true);
    expect(screen.getByText(/enabled for users without an individual override/i)).toBeTruthy();
  });

  it('invokes the local save callback and disables it while saving', async () => {
    const user = userEvent.setup();
    const handleSaveHostedExecution = vi.fn();
    const { rerender } = renderSection({ handleSaveHostedExecution });

    await user.click(screen.getByRole('button', { name: 'Save Hosted Execution' }));
    expect(handleSaveHostedExecution).toHaveBeenCalledTimes(1);

    rerender(
      <HostedExecutionSettingsSection
        open
        onToggle={() => {}}
        formData={{}}
        setFormData={vi.fn()}
        handleSaveHostedExecution={handleSaveHostedExecution}
        hostedExecutionSaving
      />,
    );

    expect((screen.getByRole('button', { name: 'Saving...' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it('associates the visible label with the checkbox', () => {
    renderSection();

    expect(screen.getByLabelText('Enable hosted chat').id).toBe('hosted-execution-enabled');
  });
});
