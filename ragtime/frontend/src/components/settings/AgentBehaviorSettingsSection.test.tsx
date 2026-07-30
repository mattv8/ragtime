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
}: {
  formData?: UpdateSettingsRequest;
  agentBehaviorSaving?: boolean;
  handleSaveAgentBehavior?: () => void | Promise<void>;
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
    expect(screen.getByText(/global agent execution and tool behavior/i)).toBeTruthy();
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

  it('explains on-demand tool loading by default and legacy eager loading when disabled', async () => {
    const user = userEvent.setup();
    renderSection();

    expect(screen.getByText(/Only essential tools and the tool-skill controls are sent initially/i)).toBeTruthy();

    await user.click(screen.getByLabelText('Load tools on demand'));

    expect(screen.getByText(/All eligible tools and schemas are sent with every request/i)).toBeTruthy();
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
      />,
    );

    expect((screen.getByRole('button', { name: 'Saving...' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });
});
