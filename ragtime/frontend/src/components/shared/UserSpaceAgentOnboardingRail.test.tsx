import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UserSpaceAgentOnboardingRail } from './UserSpaceAgentOnboardingRail';

describe('UserSpaceAgentOnboardingRail', () => {
  it('routes every client action to its selected setup guide', async () => {
    const user = userEvent.setup();
    const onSelectClient = vi.fn();
    render(<UserSpaceAgentOnboardingRail onSelectClient={onSelectClient} onDismiss={vi.fn()} />);

    for (const [label, clientId] of [
      ['Claude Cowork / Desktop', 'claude-desktop'],
      ['Claude Code', 'claude-code'],
      ['Cursor', 'cursor'],
      ['ChatGPT', 'chatgpt'],
      ['Cline', 'cline'],
      ['Continue', 'continue'],
      ['OpenCode', 'opencode'],
      ['Hermes', 'hermes'],
      ['Codex', 'codex'],
    ] as const) {
      await user.click(screen.getByRole('button', { name: `Set up ${label}` }));
      expect(onSelectClient).toHaveBeenLastCalledWith(clientId, expect.any(HTMLButtonElement));
    }
  });

  it('dismisses without selecting a client', async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(<UserSpaceAgentOnboardingRail onSelectClient={vi.fn()} onDismiss={onDismiss} />);

    await user.click(screen.getByRole('button', { name: 'Dismiss agent setup' }));
    expect(onDismiss).toHaveBeenCalledOnce();
  });
});

afterEach(cleanup);
