import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ToolAccessEditor,
  type ToolAccessGroupOption,
  type ToolAccessPolicy,
  type ToolAccessUserOption,
} from './ToolAccessEditor';

const USER_OPTIONS: ToolAccessUserOption[] = [
  {
    id: 'user-1',
    username: 'alice',
    display_name: 'Alice Admin',
    is_admin: false,
  },
  {
    id: 'user-2',
    username: 'bob',
    display_name: null,
    is_admin: false,
  },
  {
    id: 'user-3',
    username: 'admin',
    display_name: 'Admin User',
    is_admin: true,
  },
];
const STANDARD_USER_OPTIONS = USER_OPTIONS.filter((user) => !user.is_admin);

const GROUP_OPTIONS: ToolAccessGroupOption[] = [
  {
    id: 'group-1',
    key: 'engineering',
    display_name: 'Engineering',
    provider: 'ldap',
    member_count: 4,
  },
  {
    id: 'group-2',
    key: 'orphaned',
    display_name: 'Orphaned Group',
    provider: 'local_managed',
    member_count: 0,
  },
];

const BASE_POLICY: ToolAccessPolicy = {
  tool_id: 'tool-1',
  default_chat_access: 'read',
  default_workspace_access: 'deny',
  users: [],
  groups: [],
};

function Harness({
  initialPolicy = BASE_POLICY,
  userOptions = STANDARD_USER_OPTIONS,
  disabled = false,
  globalWriteEnabled = true,
}: {
  initialPolicy?: ToolAccessPolicy;
  userOptions?: ToolAccessUserOption[];
  disabled?: boolean;
  globalWriteEnabled?: boolean;
}) {
  const [policy, setPolicy] = useState(initialPolicy);

  return (
    <ToolAccessEditor
      policy={policy}
      userOptions={userOptions}
      groupOptions={GROUP_OPTIONS}
      disabled={disabled}
      globalWriteEnabled={globalWriteEnabled}
      onChange={setPolicy}
    />
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('ToolAccessEditor', () => {
  it('renders compact defaults without a warning callout by default', () => {
    render(<Harness />);

    expect(screen.getByText('Default')).toBeTruthy();
    expect(screen.getByText(/when no override matches/i)).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.queryByText(/admins always have full access/i)).toBeNull();
  });

  it('shows a warning callout only for explicit denies', () => {
    render(
      <ToolAccessEditor
        policy={{
          ...BASE_POLICY,
          default_workspace_access: 'read_write',
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'deny',
              workspace_access: null,
              display_name: 'Alice Admin',
              principal_detail: 'alice',
            },
          ],
        }}
        userOptions={USER_OPTIONS}
        groupOptions={GROUP_OPTIONS}
        globalWriteEnabled={false}
        onChange={() => undefined}
      />,
    );

    const alert = screen.getByRole('alert');
    expect(alert.textContent).toMatch(/explicit deny overrides/i);
    expect(alert.textContent).not.toContain(
      'Read+Write stays read-only here until the tool itself allows writes.',
    );
  });

  it('does not show a write warning when only Read+Write is unavailable', () => {
    render(
      <ToolAccessEditor
        policy={{ ...BASE_POLICY, default_workspace_access: 'read_write' }}
        userOptions={USER_OPTIONS}
        groupOptions={GROUP_OPTIONS}
        globalWriteEnabled={false}
        onChange={() => undefined}
      />,
    );

    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('displays Read as the effective default access when tool writes are disabled', async () => {
    const user = userEvent.setup();
    render(
      <Harness
        globalWriteEnabled={false}
        initialPolicy={{ ...BASE_POLICY, default_chat_access: 'read_write' }}
      />,
    );

    const defaultChatAccess = screen.getByRole('combobox', { name: 'Default chat access' });
    expect(defaultChatAccess.textContent).toBe('Read');
    expect((defaultChatAccess as HTMLButtonElement).value).toBe('read');

    await user.click(defaultChatAccess);
    const options = screen.getByRole('listbox', { name: 'Default chat access options' });
    expect(
      within(options).getByRole('option', { name: 'Read' }).getAttribute('aria-selected'),
    ).toBe('true');
    expect(
      within(options).getByRole('option', { name: 'Read+Write' }).getAttribute('aria-selected'),
    ).toBe('false');
  });

  it('disables Read+Write selections and explains when tool writes are disabled', async () => {
    const user = userEvent.setup();
    const originalGetComputedStyle = window.getComputedStyle.bind(window);
    vi.spyOn(window, 'getComputedStyle').mockImplementation((element) => {
      if (element.parentElement?.classList.contains('tool-access-select-dropdown')) {
        const dropdownStyle = Object.create(
          originalGetComputedStyle(element),
        ) as CSSStyleDeclaration;
        Object.defineProperty(dropdownStyle, 'zIndex', { value: '9100' });
        return dropdownStyle;
      }
      return originalGetComputedStyle(element);
    });

    render(
      <Harness
        globalWriteEnabled={false}
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    await user.click(screen.getByRole('combobox', { name: 'Alice Admin chat access' }));
    const chatOptions = screen.getByRole('listbox', { name: 'Alice Admin chat access options' });
    const explicitChatOption = within(chatOptions).getByRole('option', { name: 'Read+Write' });
    expect(explicitChatOption.getAttribute('aria-disabled')).toBeNull();

    await user.click(explicitChatOption);
    expect(screen.getByRole('combobox', { name: 'Alice Admin chat access' }).textContent).toBe(
      'Read+Write',
    );

    await user.click(screen.getByRole('combobox', { name: 'Default chat access' }));
    const defaultOptions = screen.getByRole('listbox', { name: 'Default chat access options' });
    const blockedDefaultOption = within(defaultOptions).getByRole('option', { name: 'Read+Write' });
    expect(blockedDefaultOption.getAttribute('aria-disabled')).toBe('true');

    await user.hover(blockedDefaultOption);
    const tooltip = await screen.findByRole('tooltip');
    expect(tooltip.textContent).toContain('would grant everyone write access');
    expect(tooltip.textContent).toContain('Enable Write Access on this tool first');
    expect(tooltip.textContent).toContain(
      'Specific users and groups can still be granted Read+Write below.',
    );
    expect(tooltip.style.zIndex).toBe('9101');
  });

  it('preserves explicit read+write overrides when tool writes are disabled globally', () => {
    render(
      <Harness
        globalWriteEnabled={false}
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read_write',
              workspace_access: 'read_write',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    expect(screen.getByRole('combobox', { name: 'Alice Admin chat access' }).textContent).toBe(
      'Read+Write',
    );
    expect(screen.getByRole('combobox', { name: 'Alice Admin workspace access' }).textContent).toBe(
      'Read+Write',
    );
  });

  it('opens permission options upward when there is not enough space below the trigger', async () => {
    const user = userEvent.setup();
    const originalInnerHeight = window.innerHeight;
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 430 });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 353,
      height: 34,
      left: 699,
      right: 816,
      top: 319,
      width: 117,
      x: 699,
      y: 319,
      toJSON: () => ({}),
    } as DOMRect);

    render(<Harness />);
    await user.click(screen.getByRole('combobox', { name: 'Default chat access' }));

    const options = screen.getByRole('listbox', { name: 'Default chat access options' });
    expect(options.style.bottom).toBe('115px');
    expect(options.style.top).toBe('');
    expect(options.style.maxHeight).toBe('311px');

    Object.defineProperty(window, 'innerHeight', {
      configurable: true,
      value: originalInnerHeight,
    });
  });

  it('shows persistent browse affordance and no open listbox by default', () => {
    render(<Harness />);

    const browseButton = screen.getByRole('button', { name: /browse all/i });

    expect(screen.getByText(/type to search, or/i)).toBeTruthy();
    expect(browseButton).toBeTruthy();
    expect((browseButton as HTMLButtonElement).disabled).toBe(false);
    expect(browseButton.getAttribute('aria-expanded')).toBe('false');
    expect(browseButton.getAttribute('aria-controls')).toBeTruthy();
    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
    expect(
      screen.getByText(/no user or group overrides\. search above to add one\./i),
    ).toBeTruthy();
  });

  it('disables browse when there are no principals left to add', () => {
    render(
      <ToolAccessEditor
        policy={BASE_POLICY}
        userOptions={[]}
        groupOptions={[]}
        onChange={() => undefined}
      />,
    );

    const browseButton = screen.getByRole('button', { name: /browse all/i });
    expect((browseButton as HTMLButtonElement).disabled).toBe(true);
    expect(browseButton.getAttribute('aria-expanded')).toBe('false');
    expect(browseButton.getAttribute('aria-controls')).toBeNull();
  });

  it('opens browse-all candidates for every unselected principal on empty query and closes after add', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    const input = screen.getByRole('textbox', { name: /search or add users and groups/i });
    const browseButton = screen.getByRole('button', { name: /browse all/i });

    expect((input as HTMLInputElement).value).toBe('');

    await user.click(browseButton);

    const dropdown = screen.getByRole('listbox', { name: /add users and groups/i });
    expect(browseButton.getAttribute('aria-expanded')).toBe('true');

    const options = within(dropdown).getAllByRole('option');
    expect(options).toHaveLength(3);
    expect(within(dropdown).queryByRole('option', { name: /alice admin/i })).toBeNull();
    expect(within(dropdown).getByRole('option', { name: /bob/i })).toBeTruthy();
    expect(within(dropdown).getByRole('option', { name: /engineering/i })).toBeTruthy();
    expect(within(dropdown).getByRole('option', { name: /orphaned group/i })).toBeTruthy();

    await user.click(within(dropdown).getByRole('option', { name: /bob/i }));

    expect(screen.getByText('bob')).toBeTruthy();
    expect((input as HTMLInputElement).value).toBe('');
    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
    expect(browseButton.getAttribute('aria-expanded')).toBe('false');
  });

  it('dismisses browse-all candidates when clicking outside the selector', async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: /browse all/i }));
    expect(screen.getByRole('listbox', { name: /add users and groups/i })).toBeTruthy();

    await user.click(screen.getByRole('columnheader', { name: 'Who' }));

    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
    expect(screen.getByRole('button', { name: /browse all/i }).getAttribute('aria-expanded')).toBe(
      'false',
    );
  });

  it('dismisses browse-all candidates when clicking the search input', async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: /browse all/i }));
    expect(screen.getByRole('listbox', { name: /add users and groups/i })).toBeTruthy();

    await user.click(screen.getByRole('textbox', { name: /search or add users and groups/i }));

    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
    expect(screen.getByRole('button', { name: /browse all/i }).getAttribute('aria-expanded')).toBe(
      'false',
    );
  });

  it('adds a principal from the search dropdown, initializes read/read, and clears search', async () => {
    const user = userEvent.setup();

    render(<Harness userOptions={USER_OPTIONS} />);

    const input = screen.getByRole('textbox', { name: /search or add users and groups/i });
    await user.type(input, 'ali');

    const dropdown = screen.getByRole('listbox', { name: /add users and groups/i });
    await user.click(within(dropdown).getByRole('option', { name: /alice admin/i }));

    expect(screen.getByText('Alice Admin')).toBeTruthy();
    expect((input as HTMLInputElement).value).toBe('');
    expect(
      (screen.getByRole('combobox', { name: 'Alice Admin chat access' }) as HTMLSelectElement)
        .value,
    ).toBe('read');
    expect(
      (screen.getByRole('combobox', { name: 'Alice Admin workspace access' }) as HTMLSelectElement)
        .value,
    ).toBe('read');
    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
  });

  it('shows a Group badge for group candidates but not user candidates', async () => {
    const user = userEvent.setup();

    render(<Harness />);

    await user.click(screen.getByRole('button', { name: /browse all/i }));

    const dropdown = screen.getByRole('listbox', { name: /add users and groups/i });
    const groupOption = within(dropdown).getByRole('option', { name: /engineering/i });
    const userOption = within(dropdown).getByRole('option', { name: /bob/i });

    expect(within(groupOption).getByText('Group')).toBeTruthy();
    expect(within(userOption).queryByText('Group')).toBeNull();
  });

  it('marks admin candidates as unavailable because they already have full access', async () => {
    const user = userEvent.setup();

    render(<Harness userOptions={USER_OPTIONS} />);

    await user.click(screen.getByRole('button', { name: /browse all/i }));

    const adminOption = screen.getByRole('option', { name: /admin user/i });
    expect(within(adminOption).getByText('Admin')).toBeTruthy();
    expect((adminOption as HTMLButtonElement).disabled).toBe(true);
    const accessTable = screen.getByRole('table', { name: 'Tool access policy editor' });
    expect(within(accessTable).getByRole('row', { name: /Admin User/ })).toBeTruthy();

    await user.click(adminOption);
    expect(within(accessTable).getAllByRole('row', { name: /Admin User/ })).toHaveLength(1);
  });

  it('uses structured provider and member details for group candidates', async () => {
    const user = userEvent.setup();

    render(<Harness />);

    await user.click(screen.getByRole('button', { name: /browse all/i }));

    const dropdown = screen.getByRole('listbox', { name: /add users and groups/i });
    const ldapGroup = within(dropdown).getByRole('option', { name: /engineering/i });
    const internalGroup = within(dropdown).getByRole('option', { name: /orphaned group/i });

    expect(within(ldapGroup).getByText('LDAP')).toBeTruthy();
    expect(within(ldapGroup).getByText('4 members')).toBeTruthy();
    expect(within(ldapGroup).queryByText(/·/)).toBeNull();
    expect(within(internalGroup).queryByText('LDAP')).toBeNull();
    expect(within(internalGroup).getByText('0 members')).toBeTruthy();
  });

  it('prevents duplicate add candidates once a principal is already present', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: 'alice',
            },
          ],
        }}
      />,
    );

    await user.type(
      screen.getByRole('textbox', { name: /search or add users and groups/i }),
      'ali',
    );

    expect(screen.queryByRole('option', { name: /alice admin/i })).toBeNull();
    expect(screen.queryByRole('listbox', { name: /add users and groups/i })).toBeNull();
  });

  it('updates chat and workspace selects independently', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: null,
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    const chatSelect = screen.getByRole('combobox', { name: 'Alice Admin chat access' });
    await user.click(chatSelect);
    await user.click(screen.getByRole('option', { name: 'Deny' }));

    const workspaceSelect = screen.getByRole('combobox', { name: 'Alice Admin workspace access' });
    await user.click(workspaceSelect);
    await user.click(screen.getByRole('option', { name: 'Read+Write' }));

    expect(chatSelect.textContent).toBe('Deny');
    expect(workspaceSelect.textContent).toBe('Read+Write');
  });

  it('removes a row when both surfaces are set back to inherit', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'deny',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    await user.click(screen.getByRole('combobox', { name: 'Alice Admin chat access' }));
    await user.click(screen.getByRole('option', { name: 'Inherit' }));
    expect(screen.getByText('Alice Admin')).toBeTruthy();

    await user.click(screen.getByRole('combobox', { name: 'Alice Admin workspace access' }));
    await user.click(screen.getByRole('option', { name: 'Inherit' }));

    expect(screen.queryByText('Alice Admin')).toBeNull();
    expect(
      screen.getByText(/no user or group overrides\. search above to add one\./i),
    ).toBeTruthy();
  });

  it('filters existing rows by the search query and shows a no matches row', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
          groups: [
            {
              principal_id: 'group-1',
              chat_access: null,
              workspace_access: 'read',
              display_name: 'Engineering',
              principal_detail: 'LDAP · 4 members',
            },
          ],
        }}
      />,
    );

    const input = screen.getByRole('textbox', { name: /search or add users and groups/i });
    await user.type(input, 'engine');

    expect(screen.queryByText('Alice Admin')).toBeNull();
    expect(screen.getByText('Engineering')).toBeTruthy();

    await user.clear(input);
    await user.type(input, 'missing');

    expect(screen.getByText(/no matching overrides/i)).toBeTruthy();
    expect(screen.getByText('Default')).toBeTruthy();
  });

  it('removes rows with the remove button', async () => {
    const user = userEvent.setup();

    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          groups: [
            {
              principal_id: 'group-2',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Orphaned Group',
              principal_detail: 'Internal · 0 members',
              orphaned: true,
            },
          ],
        }}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Remove Orphaned Group' }));

    expect(screen.queryByText('Orphaned Group')).toBeNull();
  });

  it('shows a Group badge only on saved group rows', () => {
    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
          groups: [
            {
              principal_id: 'group-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Engineering',
              principal_detail: 'LDAP · 4 members',
            },
          ],
        }}
      />,
    );

    const userRow = screen.getByText('Alice Admin').closest('[role="row"]');
    const groupRow = screen.getByText('Engineering').closest('[role="row"]');

    expect(userRow).toBeTruthy();
    expect(groupRow).toBeTruthy();
    expect(within(groupRow as HTMLElement).getByText('Group')).toBeTruthy();
    expect(within(userRow as HTMLElement).queryByText('Group')).toBeNull();
  });

  it('shows admins as automatic locked rows with the access explanation on their controls', async () => {
    const user = userEvent.setup();

    render(<Harness userOptions={USER_OPTIONS} globalWriteEnabled={false} />);

    const adminRow = screen.getByText('Admin User').closest('[role="row"]');
    expect(adminRow).toBeTruthy();
    expect(within(adminRow as HTMLElement).getByText('Admin')).toBeTruthy();
    expect(screen.queryByText(/^Admins always have full access\./)).toBeNull();

    const adminChatAccess = within(adminRow as HTMLElement).getByRole('combobox', {
      name: 'Admin User chat access',
    });
    expect(adminChatAccess.textContent).toBe('Read+Write');
    expect(adminChatAccess.getAttribute('aria-disabled')).toBe('true');
    expect(
      (
        within(adminRow as HTMLElement).getByRole('button', {
          name: 'Admin User access is automatic',
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);

    await user.hover(adminChatAccess);
    expect(await screen.findByText('Admins always have full access.')).toBeTruthy();
  });

  it('disables search, selects, and remove buttons when disabled', () => {
    render(
      <Harness
        disabled
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    expect(
      (screen.getByRole('textbox', { name: /search or add users and groups/i }) as HTMLInputElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('combobox', { name: 'Default chat access' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('combobox', { name: 'Alice Admin chat access' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Remove Alice Admin' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: /browse all/i }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('renders a disabled default remove button instead of pinned text', () => {
    render(<Harness />);

    expect(screen.queryByText('Pinned')).toBeNull();

    const button = screen.getByRole('button', { name: /default access cannot be removed/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    expect(button.getAttribute('title')).toMatch(/cannot be removed/i);
  });

  it('exposes accessible select labels for default and principal rows', () => {
    render(
      <Harness
        initialPolicy={{
          ...BASE_POLICY,
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: 'read',
              display_name: 'Alice Admin',
              principal_detail: '@alice',
            },
          ],
        }}
      />,
    );

    expect(screen.getByRole('combobox', { name: 'Default chat access' })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: 'Default workspace access' })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: 'Alice Admin chat access' })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: 'Alice Admin workspace access' })).toBeTruthy();
  });
});
