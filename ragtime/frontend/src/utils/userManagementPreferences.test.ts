import { afterEach, describe, expect, it } from 'vitest';
import {
  getUserManagementRowsCookieName,
  getUserManagementRowsPreference,
  setUserManagementRowsPreference,
} from './userManagementPreferences';

afterEach(() => {
  document.cookie.split('; ').forEach((entry) => {
    document.cookie = `${entry.split('=')[0]}=; path=/; max-age=0`;
  });
});

describe('user management row preferences', () => {
  it('restores only accepted values and falls back for invalid cookies', () => {
    document.cookie = `${getUserManagementRowsCookieName('admin')}=20; path=/`;
    expect(getUserManagementRowsPreference('admin')).toBe(20);
    document.cookie = `${getUserManagementRowsCookieName('other')}=15; path=/`;
    expect(getUserManagementRowsPreference('other')).toBe(10);
    document.cookie = `${getUserManagementRowsCookieName('hex')}=0x14; path=/`;
    expect(getUserManagementRowsPreference('hex')).toBe(10);
  });

  it('isolates administrators and clears the default preference', () => {
    setUserManagementRowsPreference('admin/a', 50);
    expect(getUserManagementRowsPreference('admin/a')).toBe(50);
    expect(getUserManagementRowsPreference('admin-b')).toBe(10);
    setUserManagementRowsPreference('admin/a', 10);
    expect(getUserManagementRowsPreference('admin/a')).toBe(10);
    expect(document.cookie).not.toContain(encodeURIComponent('admin/a'));
  });
});
