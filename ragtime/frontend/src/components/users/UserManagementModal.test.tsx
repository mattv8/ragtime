import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ContentProtectionApiError,
  type ContentProtectionConfig,
  type PolicyConflictResult,
} from '@/api/contentProtection';
import type { AuthGroup, User } from '@/types';
import { UserManagementModal } from './UserManagementModal';

const apiMock = vi.hoisted(() => ({
  updateLocalUser: vi.fn(),
  updateUserRole: vi.fn(),
  setUserGroups: vi.fn(),
  updateUserGenerationPolicy: vi.fn(),
  listUsers: vi.fn(),
  getUser: vi.fn(),
}));
vi.mock('@/api', () => ({ api: apiMock }));
const contentProtectionMock = vi.hoisted(() => ({
  getPolicyConflicts: vi.fn().mockResolvedValue({ conflicts: [], total: 0, truncated: false }),
  updateContentProtectionConfigSlice: vi.fn(),
  withUserOverride: vi.fn(),
}));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getPolicyConflicts: contentProtectionMock.getPolicyConflicts },
  updateContentProtectionConfigSlice: contentProtectionMock.updateContentProtectionConfigSlice,
  withUserOverride: contentProtectionMock.withUserOverride,
}));
vi.mock('./UserSecurityTab', () => ({
  UserSecurityTab: ({ onBusyChange }: { onBusyChange?: (busy: boolean) => void }) => (
    <button type="button" onClick={() => onBusyChange?.(true)}>
      Start deferred security mutation
    </button>
  ),
}));
const user = {
  id: 'u1',
  username: 'alex',
  display_name: 'Alex',
  email: 'a@example.test',
  role: 'admin',
  auth_provider: 'local_managed',
  mfa_enabled: false,
  mfa_required: false,
} as User;
const CONTENT_PROTECTION_CONFIG: ContentProtectionConfig = {
  revision: 1,
  enabled: true,
  schema_version: 2,
  share_with_assistant: false,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev-latest' }, llm_model: null },
  strictness: 'strict',
  categories: [],
  access_levels: [{ id: 'standard', name: 'Standard', granted_category_ids: [], guidance: '' }],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'selected_scopes',
  requirements: [],
  user_overrides: [],
};
const CONFLICT_DETAIL = {
  code: 'conflicting_group_policies' as const,
  message: 'Conflicting group policies',
  conflicts: [
    {
      user_id: user.id,
      username: user.username,
      display_name: 'Alex Candidate',
      skip_group_ids: ['skip'],
      require_group_ids: ['require'],
    },
  ],
  total: 1,
  truncated: false,
};
function authGroup(id: string, display_name: string): AuthGroup {
  return {
    id,
    key: id,
    display_name,
    description: '',
    provider: 'local_managed',
    role: null,
    member_count: 0,
    manual_member_count: 0,
    ldap_member_count: 0,
    member_previews: [],
    is_logon_group: false,
  };
}
const props = () => ({
  user,
  currentUser: null,
  authGroups: [],
  users: [user],
  workspaces: [],
  chats: [],
  chatsLoaded: true,
  chatsLoading: false,
  chatsError: null,
  workspaceStateById: {},
  workspaceLastMessageAtById: {},
  workspaceLastConversationById: {},
  workspaceMetaLoading: false,
  storageByWorkspaceId: {},
  storageFailuresByWorkspaceId: {},
  storageLoading: false,
  deletingWorkspaceIds: new Set<string>(),
  contentProtectionConfig: null,
  contentProtectionLoadFailed: false,
  onRetryContentProtectionConfig: vi.fn(),
  onClose: vi.fn(),
  onUserUpdated: vi.fn(),
  onDelete: vi.fn(),
  onWorkspaceDelete: vi.fn(),
  onWorkspaceTransfer: vi.fn(),
  onOpenWorkspace: vi.fn(),
  onOpenChat: vi.fn(),
  onDeleteChat: vi.fn(),
  onCancelChat: vi.fn(),
  onResourcesOpen: vi.fn(),
  onComputeStorage: vi.fn(),
  formatBytes: (bytes: number) => `${bytes} B`,
  formatDateTime: () => 'n/a',
  getConversationContextMeta: () => 'n/a',
  onContentProtectionConfigChange: vi.fn(),
});
describe('UserManagementModal', () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });
  it('preserves a profile draft through tab changes and asks before discarding it', () => {
    const p = props();
    render(<UserManagementModal {...p} />);
    fireEvent.change(screen.getByLabelText('Display name'), { target: { value: 'Changed' } });
    fireEvent.click(screen.getByRole('tab', { name: 'Policies' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Account & access' }));
    expect((screen.getByLabelText('Display name') as HTMLInputElement).value).toBe('Changed');
    fireEvent.click(screen.getByLabelText('Close manage user'));
    expect(screen.getByRole('alertdialog').textContent).toContain('Discard unsaved changes?');
    fireEvent.click(screen.getByRole('button', { name: 'Keep editing' }));
    expect(p.onClose).not.toHaveBeenCalled();
  });
  it('keeps a failed profile draft and reports a local error', async () => {
    apiMock.updateLocalUser.mockRejectedValueOnce(new Error('offline'));
    render(<UserManagementModal {...props()} />);
    fireEvent.change(screen.getByLabelText('Display name'), { target: { value: 'Changed' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }));
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('offline'));
    expect((screen.getByLabelText('Display name') as HTMLInputElement).value).toBe('Changed');
  });
  it('keeps independent drafts dirty after another section saves', async () => {
    apiMock.updateLocalUser.mockResolvedValueOnce({ ...user, display_name: 'Changed' });
    const p = props();
    render(<UserManagementModal {...p} />);
    fireEvent.change(screen.getByLabelText('Display name'), { target: { value: 'Changed' } });
    fireEvent.change(screen.getByLabelText('Role'), { target: { value: 'user' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }));
    await waitFor(() => expect(p.onUserUpdated).toHaveBeenCalled());
    fireEvent.click(screen.getByLabelText('Close manage user'));
    expect(screen.getByRole('alertdialog').textContent).toContain('Discard unsaved changes?');
  });
  it('reconciles profile inputs from the canonical saved user, including cleared email', async () => {
    apiMock.updateLocalUser.mockResolvedValueOnce({
      ...user,
      display_name: 'alex',
      email: null,
    });
    const p = props();
    render(<UserManagementModal {...p} />);
    fireEvent.change(screen.getByLabelText('Display name'), { target: { value: '' } });
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }));
    await waitFor(() =>
      expect(p.onUserUpdated).toHaveBeenCalledWith(expect.objectContaining({ email: null })),
    );
    expect(apiMock.updateLocalUser).toHaveBeenCalledWith('u1', { display_name: null, email: null });
    expect(screen.getByLabelText('Display name')).toHaveProperty('value', 'alex');
    expect(screen.getByLabelText('Email')).toHaveProperty('value', '');
  });
  it('keeps the delete confirmation open with an inline retryable error', async () => {
    const p = props();
    p.onDelete.mockRejectedValueOnce(new Error('delete failed'));
    render(<UserManagementModal {...p} />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete user' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete user' }));
    await screen.findByText('delete failed');
    expect(p.onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Delete user' })).toBeTruthy();
  });
  it('blocks close routes while a security mutation is pending', () => {
    const p = props();
    render(<UserManagementModal {...p} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Security' }));
    fireEvent.click(screen.getByRole('button', { name: 'Start deferred security mutation' }));
    fireEvent.click(screen.getByLabelText('Close manage user'));
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    fireEvent.mouseDown(document.getElementById('user-management-modal-overlay')!);
    expect(p.onClose).not.toHaveBeenCalled();
    expect(screen.queryByRole('alertdialog', { name: 'Discard unsaved changes' })).toBeNull();
  });
  it('focuses non-destructive confirmation actions and Escape restores their trigger', async () => {
    render(<UserManagementModal {...props()} />);
    const close = screen.getByLabelText('Close manage user');
    fireEvent.change(screen.getByLabelText('Display name'), { target: { value: 'Changed' } });
    fireEvent.click(close);
    const keepEditing = screen.getByRole('button', { name: 'Keep editing' });
    await waitFor(() => expect(document.activeElement).toBe(keepEditing));
    fireEvent.keyDown(screen.getByRole('alertdialog', { name: 'Discard unsaved changes' }), {
      key: 'Escape',
    });
    await waitFor(() => expect(document.activeElement).toBe(close));
    expect(screen.queryByRole('alertdialog', { name: 'Discard unsaved changes' })).toBeNull();

    const deleteTrigger = screen.getByRole('button', { name: 'Delete user' });
    fireEvent.click(deleteTrigger);
    const cancel = screen.getByRole('button', { name: 'Cancel' });
    await waitFor(() => expect(document.activeElement).toBe(cancel));
    fireEvent.keyDown(screen.getByRole('alertdialog', { name: 'Confirm delete user' }), {
      key: 'Escape',
    });
    await waitFor(() => expect(document.activeElement?.textContent).toBe('Delete user'));
  });
  it('uses roving tabs and traps focus without intercepting field arrows', async () => {
    render(<UserManagementModal {...props()} />);
    const account = screen.getByRole('tab', { name: 'Account & access' });
    expect(account.getAttribute('aria-controls')).toBe('user-management-u1-panel-account');
    fireEvent.keyDown(account, { key: 'ArrowRight' });
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'Policies' })),
    );
    const name = screen.getByLabelText('Display name');
    name.focus();
    fireEvent.keyDown(name, { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Policies' }).getAttribute('aria-selected')).toBe(
      'true',
    );
    expect(document.activeElement).toBe(name);
  });
  it('traps focus through the selected tab when a self-managed account has no controls', () => {
    const originalOffsetParent = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      'offsetParent',
    );
    Object.defineProperty(HTMLElement.prototype, 'offsetParent', {
      configurable: true,
      get: function (this: HTMLElement) {
        return this.closest('[hidden]') ? null : document.body;
      },
    });

    try {
      const self = { ...user, auth_provider: 'local' } as User;
      render(
        <UserManagementModal
          {...{
            ...props(),
            user: self,
            currentUser: self,
            users: [self],
          }}
        />,
      );

      const dialog = screen.getByRole('dialog');
      const close = screen.getByLabelText('Close manage user');
      const account = screen.getByRole('tab', { name: 'Account & access' });
      const policies = screen.getByRole('tab', { name: 'Policies' });
      const resources = screen.getByRole('tab', { name: 'Resources' });

      expect(account.getAttribute('tabindex')).toBe('0');
      expect(policies.getAttribute('tabindex')).toBe('-1');
      expect(resources.getAttribute('tabindex')).toBe('-1');

      close.focus();
      const forwardTab = new KeyboardEvent('keydown', {
        key: 'Tab',
        bubbles: true,
        cancelable: true,
      });
      expect(close.dispatchEvent(forwardTab)).toBe(true);
      account.focus(); // Native forward Tab moves from Close to the selected Account tab.
      fireEvent.keyDown(dialog, { key: 'Tab' });
      expect(document.activeElement).toBe(close);

      fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
      expect(document.activeElement).toBe(account);
      expect(document.activeElement).not.toBe(policies);
      expect(document.activeElement).not.toBe(resources);
    } finally {
      if (originalOffsetParent) {
        Object.defineProperty(HTMLElement.prototype, 'offsetParent', originalOffsetParent);
      } else {
        delete (HTMLElement.prototype as { offsetParent?: HTMLElement | null }).offsetParent;
      }
    }
  });
  it.each([
    { auth_provider: 'local_managed', label: 'Internal' },
    { auth_provider: 'local', label: 'Local admin' },
    { auth_provider: 'ldap', label: 'LDAP' },
    { auth_provider: 'custom_sso', label: 'custom_sso' },
  ])(
    'labels the $auth_provider auth provider as $label in the identity header',
    ({ auth_provider, label }) => {
      render(<UserManagementModal {...{ ...props(), user: { ...user, auth_provider } as User }} />);

      const provider = document.querySelector<HTMLElement>(
        '[data-user-management-identity="provider"]',
      );
      expect(provider?.textContent).toBe(label);
      expect(provider?.getAttribute('title')).toBe(auth_provider);
    },
  );
  it('loads resource details once per Resources tab activation', () => {
    const p = { ...props(), chatsLoaded: false };
    render(<UserManagementModal {...p} />);
    expect(p.onResourcesOpen).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('tab', { name: 'Resources' }));
    expect(p.onResourcesOpen).toHaveBeenCalledOnce();
    expect(screen.getByText('Loading chats...')).toBeTruthy();
    fireEvent.click(screen.getByRole('tab', { name: 'Resources' }));
    expect(p.onResourcesOpen).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('tab', { name: 'Account & access' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Resources' }));
    expect(p.onResourcesOpen).toHaveBeenCalledTimes(2);
  });

  it('renders a nullable conflict display name with its username fallback', async () => {
    contentProtectionMock.getPolicyConflicts.mockResolvedValueOnce({
      conflicts: [
        {
          user_id: user.id,
          username: user.username,
          display_name: null,
          skip_group_ids: ['skip'],
          require_group_ids: ['require'],
        },
      ],
      total: 1,
      truncated: false,
    });
    render(
      <UserManagementModal
        {...{
          ...props(),
          contentProtectionConfig: { ...CONTENT_PROTECTION_CONFIG, enabled: false },
        }}
      />,
    );

    expect(await screen.findByText('alex')).toBeTruthy();
  });

  it('retains a membership draft and candidate names after a structured 409, then clears it after retry', async () => {
    const actor = userEvent.setup();
    const authGroups = [authGroup('skip', 'Skip group'), authGroup('require', 'Require group')];
    apiMock.setUserGroups
      .mockReset()
      .mockRejectedValueOnce(
        new ContentProtectionApiError('Conflicting group policies', 409, CONFLICT_DETAIL),
      )
      .mockResolvedValueOnce({ ...user, manual_group_ids: ['skip'] });
    apiMock.getUser.mockResolvedValue({ ...user, manual_group_ids: ['skip'] });
    contentProtectionMock.getPolicyConflicts
      .mockResolvedValueOnce({
        conflicts: CONFLICT_DETAIL.conflicts,
        total: CONFLICT_DETAIL.total,
        truncated: CONFLICT_DETAIL.truncated,
      })
      .mockResolvedValueOnce({ conflicts: [], total: 0, truncated: false });
    const p = { ...props(), authGroups, contentProtectionConfig: CONTENT_PROTECTION_CONFIG };
    render(<UserManagementModal {...p} />);

    await actor.click(screen.getByRole('button', { name: 'No manual groups' }));
    await actor.click(screen.getByRole('checkbox', { name: 'Skip group' }));
    expect(await screen.findByText('Current group policy conflict')).toBeTruthy();
    await actor.click(screen.getByRole('button', { name: 'Save groups' }));

    const candidate = await waitFor(() => {
      const element = document.querySelector('[data-user-group-policy-conflict]');
      expect(element).not.toBeNull();
      return element as HTMLElement;
    });
    expect(candidate.textContent).toContain('Alex Candidate');
    expect(candidate.textContent).toContain('Skip group');
    expect(candidate.textContent).toContain('Require group');
    expect(screen.getByRole('button', { name: 'Skip group' }).textContent).toContain('Skip group');

    await actor.click(screen.getByRole('button', { name: 'Save groups' }));
    await waitFor(() =>
      expect(p.onUserUpdated).toHaveBeenCalledWith(expect.objectContaining({ id: 'u1' })),
    );
    expect(document.querySelector('[data-user-group-policy-conflict]')).toBeNull();
    await waitFor(() => expect(contentProtectionMock.getPolicyConflicts).toHaveBeenCalledTimes(2));
    expect(screen.queryByText('Current group policy conflict')).toBeNull();
  });

  it('clears a membership candidate conflict when its draft is discarded', async () => {
    const actor = userEvent.setup();
    const authGroups = [authGroup('skip', 'Skip group'), authGroup('require', 'Require group')];
    apiMock.setUserGroups
      .mockReset()
      .mockRejectedValue(
        new ContentProtectionApiError('Conflicting group policies', 409, CONFLICT_DETAIL),
      );
    render(
      <UserManagementModal
        {...{ ...props(), authGroups, contentProtectionConfig: CONTENT_PROTECTION_CONFIG }}
      />,
    );

    await actor.click(screen.getByRole('button', { name: 'No manual groups' }));
    await actor.click(screen.getByRole('checkbox', { name: 'Skip group' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Save groups' })).toHaveProperty('disabled', false),
    );
    await actor.click(screen.getByRole('button', { name: 'Save groups' }));
    await waitFor(() =>
      expect(document.querySelector('[data-user-group-policy-conflict]')).not.toBeNull(),
    );
    const groupActions = screen.getByRole('button', { name: 'Save groups' }).parentElement;
    await actor.click(groupActions!.querySelector<HTMLButtonElement>('button.btn-secondary')!);

    expect(document.querySelector('[data-user-group-policy-conflict]')).toBeNull();
    expect(screen.getByRole('button', { name: 'No manual groups' })).toBeTruthy();
  });

  it('keeps a protection conflict beside its policy Save when another section saves', async () => {
    const actor = userEvent.setup();
    contentProtectionMock.updateContentProtectionConfigSlice.mockRejectedValueOnce(
      new ContentProtectionApiError('Conflicting group policies', 422, CONFLICT_DETAIL),
    );
    apiMock.updateUserGenerationPolicy.mockResolvedValueOnce(user);
    render(
      <UserManagementModal
        {...{ ...props(), contentProtectionConfig: CONTENT_PROTECTION_CONFIG }}
      />,
    );

    await actor.click(screen.getByRole('tab', { name: 'Policies' }));
    await actor.selectOptions(
      screen.getByLabelText('Content protection override'),
      'never_classify',
    );
    await actor.click(screen.getByRole('button', { name: 'Save content protection' }));
    const conflict = await waitFor(() => {
      const element = document.querySelector('[data-user-protection-policy-conflict]');
      expect(element).not.toBeNull();
      return element as HTMLElement;
    });
    expect(conflict.textContent).toContain('Alex Candidate');
    expect(document.querySelector('[data-user-group-policy-conflict]')).toBeNull();

    await actor.selectOptions(screen.getByLabelText('Chat generation'), 'enabled');
    await actor.click(screen.getByRole('button', { name: 'Save generation settings' }));
    await waitFor(() => expect(apiMock.updateUserGenerationPolicy).toHaveBeenCalled());
    expect(document.querySelector('[data-user-protection-policy-conflict]')).not.toBeNull();
  });

  it('does not let a stale conflict lookup for an earlier user overwrite the current user', async () => {
    let resolveFirst!: (value: PolicyConflictResult) => void;
    const first = new Promise<PolicyConflictResult>((resolve) => {
      resolveFirst = resolve;
    });
    contentProtectionMock.getPolicyConflicts
      .mockReturnValueOnce(first)
      .mockResolvedValueOnce({ conflicts: [], total: 0, truncated: false });
    const { rerender } = render(
      <UserManagementModal
        {...{ ...props(), contentProtectionConfig: CONTENT_PROTECTION_CONFIG }}
      />,
    );
    await waitFor(() =>
      expect(contentProtectionMock.getPolicyConflicts).toHaveBeenCalledWith('u1'),
    );
    const laterUser = { ...user, id: 'u2', username: 'casey', display_name: 'Casey' } as User;
    rerender(
      <UserManagementModal
        {...{
          ...props(),
          user: laterUser,
          users: [laterUser],
          contentProtectionConfig: CONTENT_PROTECTION_CONFIG,
        }}
      />,
    );

    await waitFor(() =>
      expect(contentProtectionMock.getPolicyConflicts).toHaveBeenCalledWith('u2'),
    );
    await act(async () =>
      resolveFirst({
        conflicts: CONFLICT_DETAIL.conflicts,
        total: CONFLICT_DETAIL.total,
        truncated: CONFLICT_DETAIL.truncated,
      }),
    );
    expect(document.querySelector('[data-user-policy-conflict-unavailable]')).toBeNull();
    expect(screen.queryByText('Current group policy conflict')).toBeNull();
    expect(screen.queryByText('Alex Candidate')).toBeNull();
    expect(contentProtectionMock.getPolicyConflicts).toHaveBeenNthCalledWith(1, 'u1');
    expect(contentProtectionMock.getPolicyConflicts).toHaveBeenNthCalledWith(2, 'u2');
  });

  it('retries a failed current-conflict lookup and displays the successful response', async () => {
    const actor = userEvent.setup();
    contentProtectionMock.getPolicyConflicts
      .mockRejectedValueOnce(new Error('Conflict service offline'))
      .mockResolvedValueOnce({ conflicts: CONFLICT_DETAIL.conflicts, total: 1, truncated: false });
    render(
      <UserManagementModal
        {...{ ...props(), contentProtectionConfig: CONTENT_PROTECTION_CONFIG }}
      />,
    );

    expect(await screen.findByText(/Conflict service offline/)).toBeTruthy();
    await actor.click(screen.getByRole('button', { name: 'Retry' }));

    expect(await screen.findByText('Current group policy conflict')).toBeTruthy();
    expect(contentProtectionMock.getPolicyConflicts).toHaveBeenCalledTimes(2);
  });
});
