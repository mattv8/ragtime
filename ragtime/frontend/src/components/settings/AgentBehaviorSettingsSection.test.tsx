import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AgentBehaviorSettingsSection } from './AgentBehaviorSettingsSection';
import type { UpdateSettingsRequest } from '@/types';

afterEach(() => {
  cleanup();
});

function renderSection({
  formData = {},
  agentBehaviorSaving = false,
  handleSaveAgentBehavior = vi.fn(),
  isAdmin = false,
  userspaceExecTimeoutDefaultDraft = '120',
  userspaceExecTimeoutMaxDraft = '600',
  userspaceExecTimeoutError = null,
}: {
  formData?: UpdateSettingsRequest;
  agentBehaviorSaving?: boolean;
  handleSaveAgentBehavior?: () => void | Promise<void>;
  isAdmin?: boolean;
  userspaceExecTimeoutDefaultDraft?: string;
  userspaceExecTimeoutMaxDraft?: string;
  userspaceExecTimeoutError?: string | null;
} = {}) {
  function Wrapper(): JSX.Element {
    const [currentFormData, setCurrentFormData] = useState<UpdateSettingsRequest>(formData);

    return (
      <AgentBehaviorSettingsSection
        open
        onToggle={() => {}}
        formData={currentFormData}
        setFormData={setCurrentFormData}
        handleSaveAgentBehavior={handleSaveAgentBehavior}
        agentBehaviorSaving={agentBehaviorSaving}
        isAdmin={isAdmin}
        userspaceExecTimeoutDefaultDraft={userspaceExecTimeoutDefaultDraft}
        userspaceExecTimeoutMaxDraft={userspaceExecTimeoutMaxDraft}
        onUserspaceExecTimeoutDefaultDraftChange={() => {}}
        onUserspaceExecTimeoutMaxDraftChange={() => {}}
        userspaceExecTimeoutError={userspaceExecTimeoutError}
      />
    );
  }

  return render(<Wrapper />);
}

describe('AgentBehaviorSettingsSection', () => {
  it('renders the accordion, fieldset, switch card, and numeric controls with defaults', () => {
    const { container } = renderSection();

    expect(screen.getByRole('button', { name: 'Agent Behavior' })).toBeTruthy();
    expect(document.getElementById('setting-agent_behavior')).toBeTruthy();
    expect(document.getElementById('setting-tool_skills_enabled')).toBeTruthy();
    expect(screen.getByLabelText('Load tools on demand')).toBeTruthy();
    expect((screen.getByLabelText('Load tools on demand') as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText('Max Tool Iterations') as HTMLInputElement).value).toBe('30');
    expect((screen.getByLabelText('Max Tool Output (chars)') as HTMLInputElement).value).toBe(
      '5000',
    );
    expect((screen.getByLabelText('Context Window (steps)') as HTMLInputElement).value).toBe('6');
    expect(container.querySelector('.agent-behavior-settings-switch-card')).toBeTruthy();
    expect(container.querySelector('.agent-behavior-settings-grid')).toBeTruthy();
    expect(screen.getByText('30')).toBeTruthy();
    expect(screen.getByText('5K')).toBeTruthy();
    expect(screen.getByText('6')).toBeTruthy();
  });

  it('updates the toggle and numeric controls', async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByLabelText('Load tools on demand'));
    expect((screen.getByLabelText('Load tools on demand') as HTMLInputElement).checked).toBe(false);

    fireEvent.change(screen.getByLabelText('Max Tool Iterations'), { target: { value: '42' } });
    expect(screen.getByText('42')).toBeTruthy();

    fireEvent.change(screen.getByLabelText('Max Tool Output (chars)'), {
      target: { value: '0' },
    });
    expect(screen.getByText('Off')).toBeTruthy();

    fireEvent.change(screen.getByLabelText('Context Window (steps)'), {
      target: { value: '0' },
    });
    expect(screen.getByText('All')).toBeTruthy();
  });

  it('calls save once and shows the saving state', async () => {
    const user = userEvent.setup();
    const handleSaveAgentBehavior = vi.fn();
    const { rerender } = renderSection({ handleSaveAgentBehavior });

    await user.click(screen.getByRole('button', { name: 'Save Agent Behavior' }));
    expect(handleSaveAgentBehavior).toHaveBeenCalledTimes(1);

    rerender(
      <AgentBehaviorSettingsSection
        open
        onToggle={() => {}}
        formData={{}}
        setFormData={vi.fn()}
        handleSaveAgentBehavior={handleSaveAgentBehavior}
        agentBehaviorSaving
        isAdmin={false}
        userspaceExecTimeoutDefaultDraft="120"
        userspaceExecTimeoutMaxDraft="600"
        onUserspaceExecTimeoutDefaultDraftChange={() => {}}
        onUserspaceExecTimeoutMaxDraftChange={() => {}}
        userspaceExecTimeoutError={null}
      />,
    );

    expect((screen.getByRole('button', { name: 'Saving...' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it('shows editable workspace command timeout controls only to admins', () => {
    const { rerender } = renderSection({ isAdmin: true });

    expect(screen.getByLabelText('Default command timeout (seconds)')).toBeTruthy();
    expect(screen.getByLabelText('Maximum command timeout (seconds)')).toBeTruthy();
    expect(screen.getByText(/workspace terminal commands/i)).toBeTruthy();

    rerender(
      <AgentBehaviorSettingsSection
        open
        onToggle={() => {}}
        formData={{}}
        setFormData={vi.fn()}
        handleSaveAgentBehavior={vi.fn()}
        agentBehaviorSaving={false}
        isAdmin={false}
        userspaceExecTimeoutDefaultDraft="120"
        userspaceExecTimeoutMaxDraft="600"
        onUserspaceExecTimeoutDefaultDraftChange={() => {}}
        onUserspaceExecTimeoutMaxDraftChange={() => {}}
        userspaceExecTimeoutError={null}
      />,
    );

    expect(screen.queryByLabelText('Default command timeout (seconds)')).toBeNull();
  });

  it('blocks saving an invalid timeout pair and exposes its constraint error', () => {
    const handleSaveAgentBehavior = vi.fn();
    renderSection({
      isAdmin: true,
      userspaceExecTimeoutDefaultDraft: '',
      userspaceExecTimeoutMaxDraft: '30',
      userspaceExecTimeoutError: 'Enter whole-number timeout values.',
      handleSaveAgentBehavior,
    });

    expect(screen.getByRole('alert').textContent).toContain('Enter whole-number timeout values.');
    expect(
      (screen.getByRole('button', { name: 'Save Agent Behavior' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
