import { describe, expect, it } from 'vitest';
import { shouldSendCollabUpdate } from './collabUpdate';

const eligibleUpdate = {
  canEditWorkspace: true,
  collabReadOnly: false,
  hasUnsentEdit: true,
  ownsCurrentDocument: true,
  collabVersion: 1,
};

describe('shouldSendCollabUpdate', () => {
  it('allows a user edit for the current document after a snapshot', () => {
    expect(shouldSendCollabUpdate(eligibleUpdate)).toBe(true);
  });

  it.each([
    { canEditWorkspace: false },
    { collabReadOnly: true },
    { hasUnsentEdit: false },
    { ownsCurrentDocument: false },
    { collabVersion: 0 },
  ])('rejects ineligible updates: %o', (overrides) => {
    expect(shouldSendCollabUpdate({ ...eligibleUpdate, ...overrides })).toBe(false);
  });
});
