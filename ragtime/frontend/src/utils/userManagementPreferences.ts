import { clearCookieValue, getCookieValue, setSessionCookieValue } from './cookies';

export const USER_MANAGEMENT_DEFAULT_ROWS = 10;
const ACCEPTED_ROW_COUNTS = new Set(['5', '10', '20', '50']);

export function getUserManagementRowsCookieName(adminId: string): string {
  return `users_management_rows_${encodeURIComponent(adminId)}`;
}

export function getUserManagementRowsPreference(adminId: string | null | undefined): number {
  if (!adminId) return USER_MANAGEMENT_DEFAULT_ROWS;
  const value = getCookieValue(getUserManagementRowsCookieName(adminId));
  return value && ACCEPTED_ROW_COUNTS.has(value) ? Number(value) : USER_MANAGEMENT_DEFAULT_ROWS;
}

export function setUserManagementRowsPreference(adminId: string, rows: number): void {
  const value = ACCEPTED_ROW_COUNTS.has(String(rows)) ? rows : USER_MANAGEMENT_DEFAULT_ROWS;
  const name = getUserManagementRowsCookieName(adminId);
  if (value === USER_MANAGEMENT_DEFAULT_ROWS) {
    clearCookieValue(name);
  } else {
    setSessionCookieValue(name, String(value));
  }
}
