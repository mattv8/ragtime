import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ToolAccessModal } from './ToolAccessModal';
import type { ToolAccessPolicy } from './ToolAccessEditor';
import type { ContentProtectionConfig } from '@/api/contentProtection';

const contentProtectionMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
}));

vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig: contentProtectionMock.getConfig },
  updateContentProtectionConfigSlice: contentProtectionMock.updateContentProtectionConfigSlice,
}));

const POLICY: ToolAccessPolicy = {
  tool_id: 'tool-1',
  default_chat_access: 'read',
  default_workspace_access: 'deny',
  users: [],
  groups: [],
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ToolAccessModal', () => {
  it('renders a lucide close icon with no ascii text and still closes on click', async () => {
    contentProtectionMock.getConfig.mockResolvedValue(null);
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

  it('saves the tool content protection requirement immediately', async () => {
    const user = userEvent.setup();
    const config: ContentProtectionConfig = {
      revision: 1,
      enabled: true,
      classifier_model: null,
      coverage_mode: 'selected_scopes' as const,
      profiles: [],
      group_profiles: [],
      requirements: [],
      user_overrides: [],
    };
    contentProtectionMock.getConfig.mockResolvedValue(config);
    contentProtectionMock.updateContentProtectionConfigSlice.mockImplementation(
      async (mutate: (value: ContentProtectionConfig) => ContentProtectionConfig) => mutate(config),
    );
    const onContentProtectionModeChange = vi.fn();

    render(
      <ToolAccessModal
        open
        toolName="Example Tool"
        policy={POLICY}
        userOptions={[]}
        groupOptions={[]}
        onChange={() => undefined}
        onSave={() => undefined}
        onContentProtectionModeChange={onContentProtectionModeChange}
        onClose={() => undefined}
      />,
    );

    await user.selectOptions(await screen.findByLabelText('Content protection'), 'require');

    expect(contentProtectionMock.updateContentProtectionConfigSlice).toHaveBeenCalledOnce();
    expect(
      contentProtectionMock.updateContentProtectionConfigSlice.mock.calls[0][0](config)
        .requirements,
    ).toEqual([{ scope_kind: 'tool', scope_key: 'tool-1', mode: 'require' }]);
    expect(onContentProtectionModeChange).toHaveBeenCalledWith('tool-1', 'require');
  });

  it('shows the frozen disabled and all-traffic hints', async () => {
    contentProtectionMock.getConfig.mockResolvedValue({
      revision: 1,
      enabled: false,
      classifier_model: null,
      coverage_mode: 'all_supported_traffic',
      profiles: [],
      group_profiles: [],
      requirements: [],
      user_overrides: [],
    });

    render(
      <ToolAccessModal
        open
        toolName="Example Tool"
        policy={POLICY}
        userOptions={[]}
        groupOptions={[]}
        onChange={() => undefined}
        onSave={() => undefined}
        onClose={() => undefined}
      />,
    );

    expect((await screen.findByLabelText('Content protection')).hasAttribute('disabled')).toBe(
      true,
    );
    expect(screen.getByText('Content protection is disabled in Settings.')).toBeTruthy();
    expect(
      screen.getByText(
        'Coverage is currently All supported traffic; scope requirements apply when coverage is Selected scopes.',
      ),
    ).toBeTruthy();
  });
});
