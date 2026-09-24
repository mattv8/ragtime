import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import {
  sessionLifecycle,
  type RequestAuthContext,
  type SessionLifecycle,
  type SessionPhase,
} from './sessionLifecycle';
import type { AuthStatus, User } from '@/types';

export type AuthSnapshotPhase =
  | 'bootstrapping'
  | 'anonymous'
  | 'authenticated'
  | 'establishing'
  | 'logging-out'
  | 'unavailable';
export type RefreshReason = 'bootstrap' | 'return' | 'policy-save' | 'login' | 'expiry' | 'step-up';
export type RefreshResult = 'applied' | 'superseded';

type Snapshot = {
  phase: AuthSnapshotPhase;
  status: AuthStatus | null;
  user: User | null;
  error: string | null;
};
type Ticket = { generation: number; sequence: number; promise: Promise<RefreshResult> };

const initialSnapshot: Snapshot = { phase: 'bootstrapping', status: null, user: null, error: null };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unable to check the current session.';
}

function isUnauthorized(error: unknown): boolean {
  return Boolean(error && typeof error === 'object' && 'status' in error && error.status === 401);
}

function reconcileCapabilities(status: AuthStatus, user: User): AuthStatus {
  return {
    ...status,
    chat_enabled: status.chat_enabled === true && user.chat_enabled_effective === true,
    userspace_generation_enabled:
      status.userspace_generation_enabled === true &&
      user.userspace_generation_enabled_effective === true,
  };
}

function reconcileUserCapabilities(user: User, status: AuthStatus): User {
  return {
    ...user,
    chat_enabled_effective: user.chat_enabled_effective === true && status.chat_enabled === true,
    userspace_generation_enabled_effective:
      user.userspace_generation_enabled_effective === true &&
      status.userspace_generation_enabled === true,
  };
}

function failClosed(snapshot: Snapshot, error: string | null = snapshot.error): Snapshot {
  if (snapshot.phase !== 'authenticated' || !snapshot.status || !snapshot.user) return snapshot;
  return {
    ...snapshot,
    status: {
      ...snapshot.status,
      chat_enabled: false,
      userspace_generation_enabled: false,
    },
    user: {
      ...snapshot.user,
      chat_enabled_effective: false,
      userspace_generation_enabled_effective: false,
    },
    error,
  };
}

function anonymousStatus(status: AuthStatus): AuthStatus {
  return {
    ...status,
    authenticated: false,
    chat_enabled: false,
    userspace_generation_enabled: false,
  };
}

/** Owns the browser's single publishable status/user snapshot. */
export function useAuthSession(lifecycle: SessionLifecycle = sessionLifecycle) {
  const [snapshot, setSnapshot] = useState<Snapshot>(initialSnapshot);
  const [recoveryBusy, setRecoveryBusy] = useState(false);
  const snapshotRef = useRef(snapshot);
  const ticketRef = useRef<Ticket | null>(null);
  const logoutContextRef = useRef<RequestAuthContext | null>(null);
  const logoutPromiseRef = useRef<Promise<void> | null>(null);
  const verifyingLogoutRef = useRef(false);
  const terminalLoginErrorRef = useRef<string | null>(null);
  const sequenceRef = useRef(0);
  const mountedRef = useRef(true);
  snapshotRef.current = snapshot;

  const publish = useCallback((next: Snapshot) => {
    snapshotRef.current = next;
    if (mountedRef.current) setSnapshot(next);
  }, []);

  const refresh = useCallback(
    (reason: RefreshReason): Promise<RefreshResult> => {
      if (
        lifecycle.phase === 'logging-out' ||
        ((reason === 'return' || reason === 'step-up') && lifecycle.phase === 'establishing')
      ) {
        return Promise.resolve('superseded');
      }

      const existing = ticketRef.current;
      const supersedes = reason === 'policy-save' || reason === 'login';
      if (existing && !supersedes && existing.generation === lifecycle.generation) {
        return existing.promise;
      }

      let context = lifecycle.capture('public');
      const sequence = ++sequenceRef.current;
      const ticket: Ticket = {
        generation: context.generation,
        sequence,
        promise: Promise.resolve('superseded'),
      };
      const isTicketCurrent = () =>
        ticketRef.current === ticket &&
        sequenceRef.current === sequence &&
        lifecycle.isCurrent(context);

      const recoverFromCurrentUser401 = async (originalError: unknown): Promise<RefreshResult> => {
        if (!isTicketCurrent()) return 'superseded';
        const accepted = lifecycle.markAnonymous(context);
        if (!accepted) return 'superseded';
        context = accepted;
        ticket.generation = accepted.generation;

        try {
          const recoveredStatus = await api.getAuthStatus();
          if (!isTicketCurrent()) return 'superseded';
          if (recoveredStatus.authenticated) {
            publish({
              phase: 'unavailable',
              status: null,
              user: null,
              error: `The session could not be verified: ${errorMessage(originalError)}`,
            });
            return 'applied';
          }
          lifecycle.markAnonymous(context);
          publish({ phase: 'anonymous', status: recoveredStatus, user: null, error: null });
          return 'applied';
        } catch (recoveryError) {
          if (!isTicketCurrent()) return 'superseded';
          publish({
            phase: 'unavailable',
            status: null,
            user: null,
            error: errorMessage(recoveryError),
          });
          throw recoveryError;
        }
      };

      ticket.promise = (async () => {
        let readingCurrentUser = false;
        try {
          const status = await api.getAuthStatus();
          if (!isTicketCurrent()) return 'superseded';
          if (!status.authenticated) {
            const accepted = lifecycle.markAnonymous(context);
            if (!accepted) return 'superseded';
            const terminalLoginError = terminalLoginErrorRef.current;
            terminalLoginErrorRef.current = null;
            publish({ phase: 'anonymous', status, user: null, error: terminalLoginError });
            return 'applied';
          }

          if (reason === 'expiry') terminalLoginErrorRef.current = null;

          if (lifecycle.signedOutIntent) {
            if (verifyingLogoutRef.current) {
              publish({
                phase: 'unavailable',
                status: null,
                user: null,
                error: 'Server sign-out completed, but the session is still reported as active.',
              });
            } else {
              publish({
                phase: 'anonymous',
                status: anonymousStatus(status),
                user: null,
                error: null,
              });
            }
            return 'applied';
          }

          readingCurrentUser = true;
          const user = await api.getCurrentUser();
          readingCurrentUser = false;
          if (!isTicketCurrent()) return 'superseded';
          const accepted = lifecycle.adoptSession(context, user.id);
          if (!accepted) return 'superseded';
          terminalLoginErrorRef.current = null;
          const reconciledStatus = reconcileCapabilities(status, user);
          publish({
            phase: 'authenticated',
            status: reconciledStatus,
            user: reconcileUserCapabilities(user, status),
            error: null,
          });
          return 'applied';
        } catch (error) {
          if (!isTicketCurrent()) return 'superseded';
          if (readingCurrentUser && isUnauthorized(error)) {
            return await recoverFromCurrentUser401(error);
          }

          const previous = snapshotRef.current;
          if (previous.phase === 'authenticated') {
            publish(
              reason === 'policy-save'
                ? failClosed(previous, errorMessage(error))
                : { ...previous, error: errorMessage(error) },
            );
          } else {
            publish({ phase: 'unavailable', status: null, user: null, error: errorMessage(error) });
          }
          throw error;
        } finally {
          if (ticketRef.current === ticket) ticketRef.current = null;
        }
      })();
      ticketRef.current = ticket;
      return ticket.promise;
    },
    [lifecycle, publish],
  );

  const completeLogin = useCallback(
    async (_user: User): Promise<RefreshResult> => {
      if (lifecycle.phase !== 'establishing') return 'superseded';
      terminalLoginErrorRef.current = null;
      publish({ phase: 'establishing', status: null, user: null, error: null });
      try {
        return await refresh('login');
      } catch {
        // Refresh owns the recovery state; credential UI callers must not leak rejections.
        return 'superseded';
      }
    },
    [lifecycle, publish, refresh],
  );

  const logout = useCallback((): Promise<void> => {
    if (logoutPromiseRef.current) return logoutPromiseRef.current;

    const promise = (async () => {
      const context = lifecycle.beginLogout();
      logoutContextRef.current = context;
      sequenceRef.current += 1;
      ticketRef.current = null;
      publish({ phase: 'logging-out', status: null, user: null, error: null });

      try {
        await api.logout();
      } catch (error) {
        if (lifecycle.isCurrent(context)) {
          publish({
            phase: 'unavailable',
            status: null,
            user: null,
            error: `Server sign-out could not be confirmed: ${errorMessage(error)}`,
          });
        }
        return;
      }

      if (!lifecycle.finishLogout(context)) return;
      logoutContextRef.current = null;
      verifyingLogoutRef.current = true;
      try {
        await refresh('bootstrap');
      } catch {
        // Refresh publishes a check-session recovery state for metadata failures.
      } finally {
        verifyingLogoutRef.current = false;
      }
    })();
    logoutPromiseRef.current = promise;
    void promise.finally(() => {
      if (logoutPromiseRef.current === promise) logoutPromiseRef.current = null;
    });
    return promise;
  }, [lifecycle, publish, refresh]);

  const retryLogout = useCallback(async (): Promise<void> => {
    if (!logoutContextRef.current || logoutPromiseRef.current) return;
    setRecoveryBusy(true);
    try {
      await logout();
    } finally {
      if (mountedRef.current) setRecoveryBusy(false);
    }
  }, [logout]);

  const retryBootstrap = useCallback(async (): Promise<void> => {
    if (lifecycle.phase === 'logging-out') return;
    if (lifecycle.signedOutIntent && !lifecycle.retrySignedOutSession()) return;
    setRecoveryBusy(true);
    try {
      await refresh('bootstrap');
    } catch {
      // The recovery state keeps the failure visible and focused.
    } finally {
      if (mountedRef.current) setRecoveryBusy(false);
    }
  }, [lifecycle, refresh]);

  const updatePresentation = useCallback(
    (update: (status: AuthStatus) => AuthStatus) => {
      const previous = snapshotRef.current;
      if (previous.status) publish({ ...previous, status: update(previous.status) });
    },
    [publish],
  );

  useEffect(() => {
    mountedRef.current = true;
    const unsubscribe = lifecycle.subscribe((event) => {
      if (event.type === 'establishing') {
        terminalLoginErrorRef.current = null;
      } else if (event.type === 'expired' || event.type === 'logout-started') {
        if (event.type === 'expired' && event.wasEstablishing) {
          terminalLoginErrorRef.current =
            'Your sign-in session could not be verified. Please sign in again.';
        }
        sequenceRef.current += 1;
        ticketRef.current = null;
        publish({
          phase: event.type === 'logout-started' ? 'logging-out' : 'bootstrapping',
          status: null,
          user: null,
          error: null,
        });
        if (event.type === 'expired') void refresh('expiry').catch(() => undefined);
      } else if (event.type === 'renewed') {
        sequenceRef.current += 1;
        ticketRef.current = null;
        publish(failClosed(snapshotRef.current, null));
        void refresh('login').catch(() => undefined);
      } else if (event.type === 'revalidate') {
        void refresh('step-up').catch(() => undefined);
      }
    });
    void refresh('bootstrap').catch(() => undefined);
    return () => {
      mountedRef.current = false;
      unsubscribe();
    };
  }, [lifecycle, publish, refresh]);

  return {
    authStatus: snapshot.status,
    currentUser: snapshot.user,
    phase: snapshot.phase,
    refresh,
    completeLogin,
    logout,
    retryBootstrap,
    retryLogout,
    recoveryAction: (logoutContextRef.current
      ? 'retry-logout'
      : lifecycle.signedOutIntent
        ? 'check-session'
        : 'retry-bootstrap') as 'retry-bootstrap' | 'retry-logout' | 'check-session',
    recoveryBusy,
    refreshError: snapshot.error,
    generation: lifecycle.generation,
    updatePresentation,
    lifecyclePhase: lifecycle.phase as SessionPhase,
  };
}
