import { useState } from 'react';

import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import type { UserDirectoryEntry, UserSpaceWorkspaceShareLinkStatus } from '@/types';

import { ShareLinkModal } from './ShareLinkModal';

const SHARE_STATUS: UserSpaceWorkspaceShareLinkStatus = {
  id: 'share-1',
  workspace_id: 'ws-1',
  has_share_link: true,
  owner_username: 'ada',
  label: 'Share one',
  share_slug: 'share-one',
  share_token: 'token-1',
  share_url: 'https://example.test/share-one',
  anonymous_share_url: 'https://example.test/a/share-one',
  subdomain_share_url: null,
  subdomain_share_enabled: false,
  subdomain_share_disabled_reason: null,
  created_at: '2026-07-14T00:00:00Z',
  public_hit_count: 0,
  last_public_hit_at: null,
  share_access_mode: 'token',
  selected_user_ids: [],
  selected_ldap_groups: [],
  has_password: false,
  active_share_style: 'anonymous',
};

const SHARE_USER: UserDirectoryEntry = {
  id: 'user-1',
  username: 'ada',
  display_name: 'Ada',
};

function buildProps(overrides: Partial<React.ComponentProps<typeof ShareLinkModal>> = {}) {
  return {
    isOpen: true,
    loadingShareStatus: false,
    shareLinkType: 'anonymous' as const,
    shareStatus: SHARE_STATUS,
    shareLinks: [SHARE_STATUS],
    selectedShareId: SHARE_STATUS.id,
    shareSlugDraft: 'share-one',
    shareSlugAvailable: true,
    shareAccessMode: 'token' as const,
    sharePasswordDraft: '',
    shareSelectableUsers: [SHARE_USER],
    shareSelectedUserIdsDraft: [],
    shareSelectedLdapGroupsDraft: [],
    shareLdapGroupDraft: '',
    ldapDiscoveredGroups: [],
    loadingLdapGroups: false,
    shareSubdomainEnabled: false,
    shareSubdomainDisabledReason: null,
    showProtectedSubdomainNotice: false,
    effectiveShareUrl: 'https://example.test/a/share-one',
    activeShareCreatedLabel: 'Created today',
    savingShareAccess: false,
    sharingWorkspace: false,
    checkingShareSlug: false,
    shareHasUnsavedChanges: false,
    onClose: () => undefined,
    onSelectShare: () => undefined,
    onCreateShareLink: () => undefined,
    onSaveShareLabel: () => undefined,
    onDeleteSelectedShareLink: () => undefined,
    onShareSlugChange: () => undefined,
    onShareAccessModeChange: () => undefined,
    onSharePasswordDraftChange: () => undefined,
    onToggleShareSelectedUser: () => undefined,
    onShareLdapGroupDraftChange: () => undefined,
    onAddShareLdapGroup: () => undefined,
    onRemoveShareLdapGroup: () => undefined,
    onSaveShareAccess: () => undefined,
    onOpenFullPreview: () => undefined,
    formatUserLabel: (user: UserDirectoryEntry, fallback: string) =>
      user.display_name || user.username || fallback,
    ...overrides,
  };
}

function renderModal(overrides: Partial<React.ComponentProps<typeof ShareLinkModal>> = {}) {
  return render(<ShareLinkModal {...buildProps(overrides)} />);
}

function getTab(name: 'Share Links' | 'API Access'): HTMLElement {
  return screen.getByRole('tab', { name });
}

afterEach(() => {
  cleanup();
});

describe('ShareLinkModal', () => {
  it('shows tabs only when api access content exists and selects Share Links by default', () => {
    const { rerender } = renderModal();

    expect(screen.queryByRole('tablist')).toBeNull();

    rerender(
      <ShareLinkModal
        {...buildProps({
          agentAccessSection: <div>Agent access</div>,
          apiAccessSection: <div>API access</div>,
        })}
      />,
    );

    expect(screen.getByRole('tablist').getAttribute('id')).toBe('share-workspace-tabs');
    expect(screen.getByRole('tab', { name: 'Share Links' }).getAttribute('aria-selected')).toBe(
      'true',
    );
    expect(screen.getByRole('tab', { name: 'API Access' }).getAttribute('aria-selected')).toBe(
      'false',
    );
  });

  it('lazy mounts api content on selection and hides the share edit footer on API Access', async () => {
    const user = userEvent.setup();
    renderModal({
      apiAccessSection: <div>API access content</div>,
    });

    await user.click(screen.getAllByRole('button', { name: 'Edit link' })[0]);

    expect(screen.getByRole('button', { name: 'Save Access' })).toBeDefined();
    expect(screen.getByRole('button', { name: 'Back to all links' })).toBeDefined();
    expect(screen.queryByText('API access content')).toBeNull();

    await user.click(getTab('API Access'));

    expect(screen.getByText('API access content')).toBeDefined();
    expect(screen.queryByRole('button', { name: 'Save Access' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Back to all links' })).toBeNull();
  });

  it('preserves api child state when switching between tabs', async () => {
    const user = userEvent.setup();

    function StatefulApiSection() {
      const [value, setValue] = useState('');
      return (
        <input
          aria-label="API token"
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
      );
    }

    renderModal({ apiAccessSection: <StatefulApiSection /> });

    await user.click(getTab('API Access'));
    await user.type(screen.getByRole('textbox', { name: 'API token' }), 'persist me');
    await user.click(getTab('Share Links'));
    await user.click(getTab('API Access'));

    expect((screen.getByRole('textbox', { name: 'API token' }) as HTMLInputElement).value).toBe(
      'persist me',
    );
  });

  it('keeps the modal untabbed when api access content is omitted', () => {
    renderModal({
      agentAccessSection: <div>Agent access</div>,
    });

    expect(screen.queryByRole('tablist')).toBeNull();
    expect(screen.getByText('Agent access')).toBeDefined();
  });

  it('uses ArrowRight and Home to move focus and selection between tabs', async () => {
    const user = userEvent.setup();
    renderModal({
      apiAccessSection: <div>API access</div>,
    });

    const linksTab = getTab('Share Links');
    linksTab.focus();

    await user.keyboard('{ArrowRight}');

    const apiTab = getTab('API Access');
    expect(document.activeElement).toBe(apiTab);
    expect(apiTab.getAttribute('aria-selected')).toBe('true');

    await user.keyboard('{Home}');

    expect(document.activeElement).toBe(linksTab);
    expect(linksTab.getAttribute('aria-selected')).toBe('true');
  });

  it('uses ArrowLeft and End to move focus and selection between tabs', async () => {
    const user = userEvent.setup();
    renderModal({
      apiAccessSection: <div>API access</div>,
    });

    const linksTab = getTab('Share Links');
    const apiTab = getTab('API Access');

    await user.click(apiTab);
    expect(apiTab.getAttribute('aria-selected')).toBe('true');

    apiTab.focus();
    await user.keyboard('{ArrowLeft}');

    expect(document.activeElement).toBe(linksTab);
    expect(linksTab.getAttribute('aria-selected')).toBe('true');

    linksTab.focus();
    await user.keyboard('{End}');

    expect(document.activeElement).toBe(apiTab);
    expect(apiTab.getAttribute('aria-selected')).toBe('true');
  });

  it('selects Share Links when api access content disappears while open', async () => {
    const user = userEvent.setup();
    const { rerender } = renderModal({
      apiAccessSection: <div>API access content</div>,
    });

    await user.click(getTab('API Access'));
    expect(getTab('API Access').getAttribute('aria-selected')).toBe('true');

    rerender(
      <ShareLinkModal
        {...buildProps({
          agentAccessSection: <div>Agent access</div>,
          apiAccessSection: undefined,
        })}
      />,
    );

    expect(screen.queryByRole('tablist')).toBeNull();
    expect(screen.queryByRole('tab', { name: 'API Access' })).toBeNull();
    expect(screen.getByText('Agent access')).toBeDefined();
  });
});
