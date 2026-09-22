import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionProfileCard } from './ContentProtectionProfileCard';

const profile = { id: 'standard', name: 'Standard', level: 0, scope: 'Operational content' };

afterEach(cleanup);

describe('ContentProtectionProfileCard', () => {
  it('commits a trimmed name on Enter', async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentProtectionProfileCard
        profile={profile}
        affectedGroups={0}
        onUpdate={onUpdate}
        onDelete={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Edit Standard name' }));
    const input = screen.getByLabelText('Standard name');
    await user.clear(input);
    await user.type(input, '  Restricted  {Enter}');

    expect(onUpdate).toHaveBeenCalledWith('standard', { name: 'Restricted' });
  });

  it('cancels an active edit on Escape without updating the draft', async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentProtectionProfileCard
        profile={profile}
        affectedGroups={0}
        onUpdate={onUpdate}
        onDelete={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Edit Standard permitted information' }));
    const scope = screen.getByLabelText('Standard scope');
    await user.clear(scope);
    await user.type(scope, 'Cancelled{Escape}');

    expect(onUpdate).not.toHaveBeenCalled();
    expect(screen.getByText('Operational content')).toBeTruthy();
  });

  it('keeps ordinary scope Enter as a newline and commits it with Ctrl+Enter', async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentProtectionProfileCard
        profile={profile}
        affectedGroups={0}
        onUpdate={onUpdate}
        onDelete={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Edit Standard permitted information' }));
    const scope = screen.getByLabelText('Standard scope');
    await user.clear(scope);
    await user.type(scope, 'First{Enter}Second');
    await user.keyboard('{Control>}{Enter}{/Control}');

    expect(onUpdate).toHaveBeenCalledWith('standard', { scope: 'First\nSecond' });
  });

  it('reverts an empty name and bounds a committed level', async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentProtectionProfileCard
        profile={profile}
        affectedGroups={0}
        onUpdate={onUpdate}
        onDelete={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Edit Standard name' }));
    await user.clear(screen.getByLabelText('Standard name'));
    await user.tab();
    expect(onUpdate).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Edit Standard level' }));
    const level = screen.getByLabelText('Standard level');
    await user.clear(level);
    await user.type(level, '9{Enter}');
    expect(onUpdate).toHaveBeenCalledWith('standard', { level: 2 });
  });
});
