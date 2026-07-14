import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getLdapGroupDisplayName, LdapGroupChips, LdapGroupSelect } from './LdapGroupSelect';

afterEach(() => {
  cleanup();
});

describe('LDAP group display helpers', () => {
  it.each([
    {
      label: 'display_name',
      group: {
        dn: 'cn=engineering,ou=groups,dc=example,dc=com',
        name: 'engineering',
        display_name: 'Engineering Team',
      },
      fallback: 'fallback',
      expected: 'Engineering Team',
    },
    {
      label: 'displayName',
      group: {
        dn: 'cn=sales,ou=groups,dc=example,dc=com',
        name: 'sales',
        displayName: 'Sales Team',
      },
      fallback: 'fallback',
      expected: 'Sales Team',
    },
    {
      label: 'name',
      group: {
        dn: 'cn=ops,ou=groups,dc=example,dc=com',
        name: 'Operations',
      },
      fallback: 'fallback',
      expected: 'Operations',
    },
    {
      label: 'fallback distinguished name',
      group: undefined,
      fallback: 'cn=unknown,dc=example,dc=com',
      expected: 'cn=unknown,dc=example,dc=com',
    },
  ])('prefers $label when resolving LDAP group display names', ({ group, fallback, expected }) => {
    expect(getLdapGroupDisplayName(group, fallback)).toBe(expected);
  });
});

describe('LdapGroupChips', () => {
  it('renders display-name chips and removes the selected group', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();

    render(
      <LdapGroupChips
        selectedDns={['cn=engineering,ou=groups,dc=example,dc=com']}
        groups={[
          {
            dn: 'cn=engineering,ou=groups,dc=example,dc=com',
            name: 'engineering',
            display_name: 'Engineering Team',
          },
        ]}
        onRemove={onRemove}
      />,
    );

    expect(screen.getByText('Engineering Team')).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Remove Engineering Team' }));

    expect(onRemove).toHaveBeenCalledWith('cn=engineering,ou=groups,dc=example,dc=com');
  });
});

describe('LdapGroupSelect', () => {
  it('hides groups that are already selected elsewhere', () => {
    render(
      <LdapGroupSelect
        value=""
        onChange={() => undefined}
        groups={[
          {
            dn: 'cn=engineering,ou=groups,dc=example,dc=com',
            name: 'engineering',
            display_name: 'Engineering Team',
          },
          {
            dn: 'cn=sales,ou=groups,dc=example,dc=com',
            name: 'sales',
            display_name: 'Sales Team',
          },
        ]}
        excludedDns={['cn=engineering,ou=groups,dc=example,dc=com']}
      />,
    );

    expect(screen.queryByRole('option', { name: 'Engineering Team' })).toBeNull();
    expect(screen.getByRole('option', { name: 'Sales Team' })).toBeTruthy();
  });
});
