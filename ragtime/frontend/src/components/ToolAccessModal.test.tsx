import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ToolAccessModal } from './ToolAccessModal';
import type { ToolAccessPolicy } from './ToolAccessEditor';

const POLICY: ToolAccessPolicy = {
  tool_id: 'tool-1',
  default_chat_access: 'read',
  default_workspace_access: 'deny',
  users: [],
  groups: [],
};

afterEach(() => {
  cleanup();
});

describe('ToolAccessModal', () => {
  it('renders a lucide close icon with no ascii text and still closes on click', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <ToolAccessModal
        open
        toolName="Example Tool"
        policy={POLICY}
        userOptions={[]}
        groupOptions={[]}
        onChange={() => undefined}
        onSave={() => undefined}
        onClose={onClose}
      />,
    );

    const closeButton = screen.getByRole('button', { name: 'Close' });

    expect(closeButton.querySelector('svg.lucide-x')).toBeTruthy();
    expect(closeButton.textContent?.trim()).toBe('');

    await user.click(closeButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
