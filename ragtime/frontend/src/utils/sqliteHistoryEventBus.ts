import { api } from '@/api';

export type HistoryEventListeners = {
  onHistoryChanged?: () => void;
  onAccessRevoked?: () => void;
};

type Subscription = {
  listeners: Set<HistoryEventListeners>;
  source?: EventSource;
  reconnectAttempts: number;
  reconnectTimer?: ReturnType<typeof setTimeout>;
};

const subscriptions = new Map<string, Subscription>();

function sourceIsClosed(source: EventSource): boolean {
  return source.readyState === 2;
}

function cancelReconnect(subscription: Subscription) {
  if (subscription.reconnectTimer === undefined) return;
  clearTimeout(subscription.reconnectTimer);
  subscription.reconnectTimer = undefined;
}

function reconnectDelay(attempt: number): number {
  const delay = Math.min(1_000 * 2 ** attempt, 30_000);
  return delay * (0.5 + Math.random());
}

function scheduleReconnect(workspaceId: string, subscription: Subscription) {
  if (subscription.reconnectTimer !== undefined || subscription.listeners.size === 0) return;
  const delay = reconnectDelay(subscription.reconnectAttempts++);
  subscription.reconnectTimer = setTimeout(() => {
    subscription.reconnectTimer = undefined;
    if (
      subscriptions.get(workspaceId) === subscription &&
      subscription.listeners.size > 0 &&
      subscription.source &&
      sourceIsClosed(subscription.source)
    ) {
      connect(workspaceId, subscription);
    }
  }, delay);
}

function connect(workspaceId: string, subscription: Subscription) {
  const source = api.subscribeUserSpaceSqliteHistoryEvents(workspaceId);
  subscription.source = source;
  source.addEventListener('open', () => {
    if (subscription.source !== source) return;
    subscription.reconnectAttempts = 0;
    cancelReconnect(subscription);
  });
  source.addEventListener('history_changed', () => {
    if (subscription.source !== source) return;
    for (const listener of subscription.listeners) listener.onHistoryChanged?.();
  });
  source.addEventListener('access_revoked', () => {
    if (subscription.source !== source) return;
    for (const listener of subscription.listeners) listener.onAccessRevoked?.();
    subscription.listeners.clear();
    cancelReconnect(subscription);
    source.close();
    if (subscriptions.get(workspaceId) === subscription) subscriptions.delete(workspaceId);
  });
  source.addEventListener('error', () => {
    if (
      subscription.source !== source ||
      !sourceIsClosed(source) ||
      subscription.listeners.size === 0
    )
      return;
    source.close();
    scheduleReconnect(workspaceId, subscription);
  });
}

/** Subscribe to workspace-scoped SQLite history invalidation events. */
export function subscribeHistoryEvents(
  workspaceId: string,
  listeners: HistoryEventListeners,
): () => void {
  let subscription = subscriptions.get(workspaceId);
  if (!subscription) {
    subscription = { listeners: new Set(), reconnectAttempts: 0 };
    subscriptions.set(workspaceId, subscription);
  }

  subscription.listeners.add(listeners);
  if (!subscription.source) connect(workspaceId, subscription);
  else if (sourceIsClosed(subscription.source)) scheduleReconnect(workspaceId, subscription);
  let subscribed = true;
  return () => {
    if (!subscribed) return;
    subscribed = false;
    subscription!.listeners.delete(listeners);
    if (subscription!.listeners.size === 0 && subscriptions.get(workspaceId) === subscription) {
      cancelReconnect(subscription!);
      subscription!.source?.close();
      subscriptions.delete(workspaceId);
    }
  };
}
