/** Browser-local ownership of authentication request provenance and epochs. */
export type AuthPurpose = 'session' | 'public' | 'challenge';
export type SessionPhase =
  | 'unknown'
  | 'anonymous'
  | 'establishing'
  | 'authenticated'
  | 'logging-out';

export type RequestAuthContext = Readonly<{
  generation: number;
  purpose: AuthPurpose;
  hadSession: boolean;
}>;

export type SessionLifecycleEvent = Readonly<{
  type:
    | 'adopted'
    | 'establishing'
    | 'renewed'
    | 'anonymous'
    | 'logout-started'
    | 'logout-finished'
    | 'expired'
    | 'revalidate';
  context: RequestAuthContext;
  phase: SessionPhase;
  signedOutIntent: boolean;
  userId: string | null;
  wasEstablishing?: boolean;
}>;

export interface SessionLifecycle {
  capture(purpose: AuthPurpose): RequestAuthContext;
  isCurrent(context: RequestAuthContext): boolean;
  beginEstablishment(context: RequestAuthContext): RequestAuthContext | null;
  beginRenewal(context: RequestAuthContext): RequestAuthContext | null;
  abandonRenewal(context: RequestAuthContext): RequestAuthContext | null;
  renewSession(context: RequestAuthContext, userId: string): RequestAuthContext | null;
  adoptSession(context: RequestAuthContext, userId: string): RequestAuthContext | null;
  markAnonymous(context: RequestAuthContext): RequestAuthContext | null;
  beginLogout(): RequestAuthContext;
  finishLogout(context: RequestAuthContext): RequestAuthContext | null;
  expire(context: RequestAuthContext): RequestAuthContext | null;
  retrySignedOutSession(): RequestAuthContext | null;
  revalidate(context: RequestAuthContext): RequestAuthContext | null;
  subscribe(listener: (event: SessionLifecycleEvent) => void): () => void;
  readonly phase: SessionPhase;
  readonly generation: number;
  readonly signedOutIntent: boolean;
}

export function createSessionLifecycle(): SessionLifecycle {
  let generation = 0;
  let phase: SessionPhase = 'unknown';
  let userId: string | null = null;
  let signedOutIntent = false;
  let logoutSettled = false;
  let renewalGeneration: number | null = null;
  let deferredExpiry: RequestAuthContext | null = null;
  const listeners = new Set<(event: SessionLifecycleEvent) => void>();
  const pendingEvents: SessionLifecycleEvent[] = [];
  let deliveringEvents = false;

  const current = (purpose: AuthPurpose): RequestAuthContext => ({
    generation,
    purpose,
    hadSession: phase === 'authenticated' || phase === 'establishing',
  });
  const isCurrent = (context: RequestAuthContext): boolean => context.generation === generation;
  const publish = (
    type: SessionLifecycleEvent['type'],
    context: RequestAuthContext,
    wasEstablishing = false,
  ) => {
    pendingEvents.push({ type, context, phase, signedOutIntent, userId, wasEstablishing });
    if (deliveringEvents) return;
    deliveringEvents = true;
    try {
      while (pendingEvents.length > 0) {
        const event = pendingEvents.shift()!;
        [...listeners].forEach((listener) => {
          try {
            listener(event);
          } catch {
            // Subscribers are presentation observers and must not break lifecycle transitions.
          }
        });
      }
    } finally {
      deliveringEvents = false;
    }
  };
  const advance = (purpose: AuthPurpose): RequestAuthContext => {
    generation += 1;
    return current(purpose);
  };

  return {
    capture: current,
    isCurrent,
    get phase() {
      return phase;
    },
    get generation() {
      return generation;
    },
    get signedOutIntent() {
      return signedOutIntent;
    },
    beginEstablishment(context) {
      if (!isCurrent(context) || phase === 'logging-out') return null;
      signedOutIntent = false;
      logoutSettled = false;
      phase = 'establishing';
      const accepted = advance('session');
      publish('establishing', accepted);
      return accepted;
    },
    beginRenewal(context) {
      if (!isCurrent(context) || phase !== 'authenticated' || !context.hadSession) return null;
      renewalGeneration = generation;
      deferredExpiry = null;
      return current('session');
    },
    abandonRenewal(context) {
      if (renewalGeneration !== context.generation || !isCurrent(context)) return null;
      renewalGeneration = null;
      const pendingExpiry = deferredExpiry;
      deferredExpiry = null;
      if (pendingExpiry) return this.expire(pendingExpiry);
      return this.revalidate(context);
    },
    renewSession(context, nextUserId) {
      if (!isCurrent(context) || phase !== 'authenticated' || userId !== nextUserId) return null;
      renewalGeneration = null;
      deferredExpiry = null;
      const accepted = advance('session');
      publish('renewed', accepted);
      return accepted;
    },
    adoptSession(context, nextUserId) {
      if (!isCurrent(context) || phase === 'logging-out' || signedOutIntent) return null;
      if (phase === 'authenticated' && userId === nextUserId) return current('session');
      if (phase === 'authenticated' && userId !== nextUserId) {
        renewalGeneration = null;
        deferredExpiry = null;
        const accepted = advance('session');
        userId = nextUserId;
        publish('adopted', accepted);
        return accepted;
      }
      phase = 'authenticated';
      userId = nextUserId;
      renewalGeneration = null;
      deferredExpiry = null;
      // Bootstrap/cross-tab adoption must fence work dispatched before identity was known.
      const accepted = advance('session');
      publish('adopted', accepted);
      return accepted;
    },
    markAnonymous(context) {
      if (!isCurrent(context) || phase === 'logging-out') return null;
      const fenceSession = phase === 'authenticated' || phase === 'establishing';
      // An old cookie can report anonymous while account MFA enrollment is
      // replacing it. This is deliberately a rejected transition: callers
      // must not publish an anonymous snapshot or start a second recovery
      // read while the renewal owner still has recovery codes on screen.
      if (renewalGeneration === generation && fenceSession) return null;
      phase = 'anonymous';
      userId = null;
      if (logoutSettled) {
        signedOutIntent = false;
        logoutSettled = false;
      }
      const accepted = fenceSession ? advance('public') : current('public');
      publish('anonymous', accepted);
      return accepted;
    },
    beginLogout() {
      signedOutIntent = true;
      logoutSettled = false;
      renewalGeneration = null;
      deferredExpiry = null;
      phase = 'logging-out';
      userId = null;
      const accepted = advance('public');
      publish('logout-started', accepted);
      return accepted;
    },
    finishLogout(context) {
      if (!isCurrent(context) || phase !== 'logging-out') return null;
      phase = 'anonymous';
      logoutSettled = true;
      const accepted = advance('public');
      publish('logout-finished', accepted);
      return accepted;
    },
    expire(context) {
      if (!isCurrent(context) || context.purpose !== 'session' || !context.hadSession) return null;
      if (renewalGeneration === generation) {
        deferredExpiry = context;
        return current('session');
      }
      const wasEstablishing = phase === 'establishing';
      signedOutIntent = true;
      logoutSettled = false;
      phase = 'anonymous';
      userId = null;
      const accepted = advance('public');
      publish('expired', accepted, wasEstablishing);
      return accepted;
    },
    retrySignedOutSession() {
      if (!signedOutIntent || phase === 'logging-out') return null;
      signedOutIntent = false;
      logoutSettled = false;
      phase = 'unknown';
      const accepted = advance('public');
      publish('revalidate', accepted);
      return accepted;
    },
    revalidate(context) {
      if (!isCurrent(context) || !context.hadSession || phase === 'logging-out') return null;
      const accepted = current('session');
      publish('revalidate', accepted);
      return accepted;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export const sessionLifecycle = createSessionLifecycle();
