import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { User } from '@/types';
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
vi.mock('@/api/contentProtection', () => ({
  updateContentProtectionConfigSlice: vi.fn(),
  withUserOverride: vi.fn(),
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
  afterEach(cleanup);
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
  it('loads resource details only when the Resources tab is opened', () => {
    const p = { ...props(), chatsLoaded: false };
    render(<UserManagementModal {...p} />);
    expect(p.onResourcesOpen).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('tab', { name: 'Resources' }));
    expect(p.onResourcesOpen).toHaveBeenCalledOnce();
    expect(screen.getByText('Loading chats...')).toBeTruthy();
  });
});
