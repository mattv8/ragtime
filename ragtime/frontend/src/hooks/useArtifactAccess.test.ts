import { act, renderHook, waitFor, cleanup } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { api } from '@/api';
import { ArtifactRequestError } from '@/utils/modelArtifact';
import { sessionLifecycle } from '@/auth/sessionLifecycle';
import { clearArtifactAccessCache, useArtifactAccess } from './useArtifactAccess';

afterEach(() => {
  cleanup();
  clearArtifactAccessCache();
  vi.restoreAllMocks();
});

it('rechecks mounted consumers after sign-out', async () => {
  act(() => {
    sessionLifecycle.markAnonymous(sessionLifecycle.capture('session'));
    sessionLifecycle.adoptSession(sessionLifecycle.capture('session'), 'user-a');
  });
  const request = vi
    .spyOn(api, 'getModelArtifact')
    .mockRejectedValueOnce(new ArtifactRequestError(403))
    .mockRejectedValueOnce(new ArtifactRequestError(401));
  const hook = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(hook.result.current.state).toBe('forbidden'));
  act(() => {
    const context = sessionLifecycle.beginLogout();
    sessionLifecycle.finishLogout(context);
  });
  await waitFor(() => expect(hook.result.current.state).toBe('signin'));
  expect(request).toHaveBeenCalledTimes(2);
});

it('keeps the previous state until the post-clear recheck resolves', async () => {
  act(() => {
    sessionLifecycle.markAnonymous(sessionLifecycle.capture('session'));
    sessionLifecycle.adoptSession(sessionLifecycle.capture('session'), 'user-a');
  });
  type Metadata = Awaited<ReturnType<typeof api.getModelArtifact>>;
  let resolveRecheck: (metadata: Metadata) => void = () => {};
  const recheck = new Promise<Metadata>((resolve) => {
    resolveRecheck = resolve;
  });
  const request = vi
    .spyOn(api, 'getModelArtifact')
    .mockRejectedValueOnce(new ArtifactRequestError(403))
    .mockReturnValueOnce(recheck);
  const hook = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(hook.result.current.state).toBe('forbidden'));
  act(() => {
    clearArtifactAccessCache();
  });
  expect(hook.result.current.state).toBe('forbidden');
  await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
  expect(hook.result.current.state).toBe('forbidden');
  await act(async () => resolveRecheck({ id: 'id', expired: false } as Metadata));
  await waitFor(() => expect(hook.result.current.state).toBe('ready'));
  expect(request).toHaveBeenCalledTimes(2);
});

it('rechecks after switching users', async () => {
  act(() => {
    sessionLifecycle.markAnonymous(sessionLifecycle.capture('session'));
    sessionLifecycle.adoptSession(sessionLifecycle.capture('session'), 'user-a');
  });
  const request = vi
    .spyOn(api, 'getModelArtifact')
    .mockRejectedValueOnce(new ArtifactRequestError(403))
    .mockRejectedValueOnce(new ArtifactRequestError(404));
  const hook = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(hook.result.current.state).toBe('forbidden'));
  act(() => {
    sessionLifecycle.adoptSession(sessionLifecycle.capture('session'), 'user-b');
  });
  await waitFor(() => expect(hook.result.current.state).toBe('notfound'));
  expect(request).toHaveBeenCalledTimes(2);
});

it('retries errors on refresh and remount', async () => {
  const request = vi
    .spyOn(api, 'getModelArtifact')
    .mockRejectedValueOnce(new Error('network'))
    .mockRejectedValueOnce(new Error('network'))
    .mockRejectedValueOnce(new ArtifactRequestError(403));
  const hook = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(hook.result.current.state).toBe('error'));
  await act(async () => hook.result.current.refresh());
  expect(request).toHaveBeenCalledTimes(2);
  hook.unmount();
  const again = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(again.result.current.state).toBe('forbidden'));
  expect(request).toHaveBeenCalledTimes(3);
});

it.each([
  [401, 'signin'],
  [403, 'forbidden'],
  [404, 'notfound'],
  [410, 'expired'],
  [500, 'error'],
] as const)('shares status %i between subscribers', async (status, expected) => {
  const request = vi
    .spyOn(api, 'getModelArtifact')
    .mockRejectedValue(new ArtifactRequestError(status));
  const first = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  const second = renderHook(() => useArtifactAccess('id', '2099-01-01'));
  await waitFor(() => expect(first.result.current.state).toBe(expected));
  expect(second.result.current.state).toBe(expected);
  expect(request).toHaveBeenCalledTimes(1);
});
