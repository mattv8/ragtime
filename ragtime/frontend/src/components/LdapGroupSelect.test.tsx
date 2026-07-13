import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getLdapGroupDisplayName, LdapGroupChips, LdapGroupSelect } from './LdapGroupSelect';

afterEach(() => {
  cleanup();
});

describe('LDAP group display helpers', () => {
  it('prefers discovered display names before names and distinguished names', () => {
    expect(
      getLdapGroupDisplayName(
        {
          dn: 'cn=engineering,ou=groups,dc=example,dc=com',
          name: 'engineering',
          display_name: 'Engineering Team',
        },
        'fallback',
      ),
    ).toBe('Engineering Team');

    expect(
      getLdapGroupDisplayName(
        {
          dn: 'cn=sales,ou=groups,dc=example,dc=com',
          name: 'sales',
          displayName: 'Sales Team',
        },
        'fallback',
      ),
    ).toBe('Sales Team');

    expect(
      getLdapGroupDisplayName(
        {
          dn: 'cn=ops,ou=groups,dc=example,dc=com',
          name: 'Operations',
        },
        'fallback',
      ),
    ).toBe('Operations');

    expect(getLdapGroupDisplayName(undefined, 'cn=unknown,dc=example,dc=com')).toBe(
      'cn=unknown,dc=example,dc=com',
    );
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
