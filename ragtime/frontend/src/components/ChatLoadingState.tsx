export type ChatLoadingStateKind = 'main' | 'history' | 'sidebar' | 'group';
export type ChatLoadingStateStatus = 'loading' | 'error';

export interface ChatLoadingStateProps {
  kind: ChatLoadingStateKind;
  state?: ChatLoadingStateStatus;
  onRetry?: () => void;
  label?: string;
  id?: string;
}

const defaultLabels: Record<ChatLoadingStateKind, Record<ChatLoadingStateStatus, string>> = {
  main: { loading: 'Loading conversation', error: 'Conversation could not load' },
  history: { loading: 'Loading earlier messages', error: 'Earlier messages could not load' },
  sidebar: { loading: 'Loading conversations', error: 'Conversations could not load' },
  group: { loading: 'Loading conversation group', error: 'Conversation group could not load' },
};

function Skeleton({ kind }: Pick<ChatLoadingStateProps, 'kind'>) {
  if (kind === 'main') {
    return (
      <div className="chat-message-skeleton-list" aria-hidden="true">
        <div className="chat-message-skeleton-row chat-message-skeleton-row-assistant">
          <div className="chat-message-skeleton-bubble chat-message-skeleton-bubble-assistant">
            <div className="chat-skeleton-line chat-message-skeleton-line" />
            <div className="chat-skeleton-line chat-message-skeleton-line chat-message-skeleton-line-short" />
          </div>
        </div>
        <div className="chat-message-skeleton-row chat-message-skeleton-row-user">
          <div className="chat-message-skeleton-bubble chat-message-skeleton-bubble-user">
            <div className="chat-skeleton-line chat-message-skeleton-line" />
            <div className="chat-skeleton-line chat-message-skeleton-line chat-message-skeleton-line-user-short" />
          </div>
        </div>
      </div>
    );
  }

  if (kind === 'sidebar') {
    return (
      <div className="chat-conversation-skeleton-list" aria-hidden="true">
        <div className="chat-conversation-skeleton">
          <div className="chat-skeleton-line chat-conversation-skeleton-title" />
          <div className="chat-skeleton-line chat-conversation-skeleton-meta" />
        </div>
        <div className="chat-conversation-skeleton">
          <div className="chat-skeleton-line chat-conversation-skeleton-title" />
          <div className="chat-skeleton-line chat-conversation-skeleton-meta" />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-loading-state-lines" aria-hidden="true">
      <div className="chat-skeleton-line chat-loading-state-line" />
      <div className="chat-skeleton-line chat-loading-state-line chat-loading-state-line-short" />
    </div>
  );
}

export function ChatLoadingState({
  kind,
  state = 'loading',
  onRetry,
  label,
  id,
}: ChatLoadingStateProps) {
  const isLoading = state === 'loading';
  const statusLabel = label ?? defaultLabels[kind][state];
  const Container = kind === 'main' ? 'section' : 'div';

  return (
    <Container
      id={id}
      className={`chat-loading-state chat-loading-state-${kind} chat-loading-state-${state}`}
      data-chat-loading-kind={kind}
      aria-label={kind === 'main' ? 'Conversation messages' : undefined}
      aria-busy={isLoading || undefined}
    >
      {isLoading ? <Skeleton kind={kind} /> : null}
      <p className="chat-loading-state-status" role="status" aria-live="polite">
        {statusLabel}
      </p>
      {!isLoading && onRetry ? (
        <button
          type="button"
          className="btn btn-secondary chat-loading-state-retry"
          onClick={onRetry}
        >
          Retry
        </button>
      ) : null}
    </Container>
  );
}
