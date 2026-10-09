import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionApiError, type ContentProtectionConfig } from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { GroupProtectionPanel } from './GroupProtectionPanel';

const api = vi.hoisted(() => ({ getConfig: vi.fn(), saveConfig: vi.fn(), preview: vi.fn() }));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: api,
}));

const group: AuthGroup = {
  id: 'finance',
  key: 'finance',
  display_name: 'Finance',
  description: '',
  provider: 'local_managed',
  role: null,
  member_count: 1,
  manual_member_count: 1,
  ldap_member_count: 0,
  member_previews: [],
  is_logon_group: false,
};
const config: ContentProtectionConfig = {
  schema_version: 2,
  revision: 1,
  enabled: true,
  share_with_assistant: false,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev' }, llm_model: null },
  strictness: 'strict',
  categories: [
    {
      id: 'rule_override',
      name: 'Override',
      description: '',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: true,
    },
  ],
  access_levels: [{ id: 'default', name: 'Default', granted_category_ids: [], guidance: '' }],
  group_access_levels: [],
  default_access_level_id: 'default',
  coverage_mode: 'selected_scopes',
  requirements: [],
  user_overrides: [],
};
function panel(overrides: Partial<ComponentProps<typeof GroupProtectionPanel>> = {}) {
  return render(
    <GroupProtectionPanel
      group={group}
      authGroups={[group]}
      config={config}
      onConfigSaved={vi.fn()}
      onUpdateMapping={vi.fn()}
      onUpdateRequirement={vi.fn()}
      onBack={vi.fn()}
      toast={{ success: vi.fn(), error: vi.fn() }}
      {...overrides}
    />,
  );
}
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('GroupProtectionPanel', () => {
  it('requests an initial saved-policy preview and keeps the default editor available', async () => {
    api.preview.mockResolvedValue({ prompt_fragment: 'server fragment' });
    panel();
    await waitFor(() => expect(api.preview).toHaveBeenCalledWith({ access_level_ids: [] }));
    expect(screen.getByLabelText('Name')).toHaveProperty('disabled', false);
    expect(screen.queryByLabelText('Override')).toBeNull();
    expect(screen.getByText(/prompt not sent/i)).toBeTruthy();
  });

  it('rebases a 409 only with the freshly fetched complete config', async () => {
    const user = userEvent.setup();
    const unrelated = { ...config, revision: 2, strictness: 'permissive' as const };
    api.preview.mockResolvedValue({ prompt_fragment: '' });
    api.saveConfig.mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409));
    api.getConfig.mockResolvedValue(unrelated);
    api.saveConfig.mockResolvedValueOnce({
      ...unrelated,
      revision: 3,
      access_levels: [{ ...config.access_levels[0], name: 'Changed' }],
    });
    panel();
    await user.clear(screen.getByLabelText('Name'));
    await user.type(screen.getByLabelText('Name'), 'Changed');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    expect(api.saveConfig.mock.calls[1][0]).toBe(2);
    expect(api.saveConfig.mock.calls[1][1].strictness).toBe('permissive');
  });
});
