import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('@xterm/xterm', () => ({ Terminal: class {} }));
vi.mock('@xterm/addon-fit', () => ({ FitAddon: class {} }));

import {
  getWorkspaceToolReadOnlyDescription,
  getWorkspaceToolStatusBadgeForState,
} from './UserSpacePanel';

afterEach(() => {
  cleanup();
});

describe('UserSpacePanel workspace tool descriptions', () => {
  it('gives admins an inline Settings > Tools link for read-only tools', async () => {
    const user = userEvent.setup();
    const onNavigateToTools = vi.fn();

    render(
      <div>{getWorkspaceToolReadOnlyDescription(true, 'tool-read-only', onNavigateToTools)}</div>,
    );

    await user.click(screen.getByRole('link', { name: 'Settings > Tools' }));

    expect(onNavigateToTools).toHaveBeenCalledWith('tool:tool-read-only');
  });

  it('does not expose a settings link to non-admins', () => {
    render(<div>{getWorkspaceToolReadOnlyDescription(false, 'tool-read-only')}</div>);

    expect(screen.queryByRole('link', { name: 'Settings > Tools' })).toBeNull();
    expect(screen.getByText(/Ask an admin to enable/)).toBeTruthy();
  });

  it('does not render a badge for globally read-only tools', () => {
    expect(getWorkspaceToolStatusBadgeForState('ineligible')).toBeNull();
  });

  it('labels workspace write states explicitly', () => {
    expect(getWorkspaceToolStatusBadgeForState('enabled')).toMatchObject({
      label: 'Workspace write',
      title: 'Workspace write enabled for this workspace',
    });
    expect(getWorkspaceToolStatusBadgeForState('eligible')).toMatchObject({
      label: 'Workspace write',
      title: 'Workspace write can be enabled for this workspace. Right-click to enable.',
    });
  });
});
