export const USERSPACE_EXEC_BRIDGE = 'userspace-exec-v1' as const;

export const USERSPACE_EXEC_MESSAGE_TYPES = {
  EXECUTE: 'ragtime-execute',
  RESULT: 'ragtime-execute-result',
  ERROR: 'ragtime-execute-error',
} as const;

export const USERSPACE_EXECUTE_TIMEOUT_MS = 60_000;
