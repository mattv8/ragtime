import type { ReceivedConversationShare } from '@/types';
import { useReceivedConversationShares } from '@/hooks/useReceivedConversationShares';
import '../styles/received-conversation-shares.css';

export interface ReceivedConversationSharesProps {
  currentUserId: string;
  searchQuery: string;
}

function shareMatchesQuery(share: ReceivedConversationShare, query: string) {
  if (!query) return true;
  return [share.title, share.label, share.owner_display_name, share.owner_username].some((value) =>
    value?.toLocaleLowerCase().includes(query),
  );
}

function getShareContext(share: ReceivedConversationShare) {
  const role = share.granted_role === 'editor' ? 'Editor' : 'Viewer';
  if (share.scope_anchor_message_idx === null || !share.scope_direction)
    return `${role} · Full chat`;
  const messageNumber = share.scope_anchor_message_idx + 1;
  return `${role} · ${
    share.scope_direction === 'forward'
      ? `From message ${messageNumber} onward`
      : `Through message ${messageNumber}`
  }`;
}

export function ReceivedConversationShares({
  currentUserId,
  searchQuery,
}: ReceivedConversationSharesProps) {
  const { shares, loading, error, refresh } = useReceivedConversationShares({
    userId: currentUserId,
    enabled: true,
  });
  const query = searchQuery.trim().toLocaleLowerCase();
  const visibleShares = shares.filter((share) => shareMatchesQuery(share, query));
  const shouldRender = loading || Boolean(error) || Boolean(query) || shares.length > 0;

  if (!shouldRender) return null;

  return (
    <section
      id="received-conversation-shares"
      className="received-conversation-shares"
      aria-labelledby="received-conversation-shares-heading"
      data-received-conversation-shares-section
    >
      <div className="received-conversation-shares-heading-row">
        <h3 id="received-conversation-shares-heading">Shared with you</h3>
        {loading && shares.length > 0 ? (
          <span
            className="received-conversation-shares-refreshing"
            aria-label="Refreshing shared chats"
          >
            Refreshing
          </span>
        ) : null}
      </div>

      {loading && shares.length === 0 ? (
        <div className="received-conversation-shares-skeletons" aria-label="Loading shared chats">
          <div className="received-conversation-shares-skeleton" />
          <div className="received-conversation-shares-skeleton" />
        </div>
      ) : null}

      {error ? (
        <div className="received-conversation-shares-error" role="alert">
          <span>{error.message || 'Unable to load shared chats.'}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => void refresh()}>
            Retry shared chats
          </button>
        </div>
      ) : null}

      {!loading && !error && visibleShares.length === 0 && query ? (
        <p className="received-conversation-shares-empty">
          No shared chats match &quot;{searchQuery.trim()}&quot;.
        </p>
      ) : null}

      {visibleShares.length > 0 ? (
        <div className="received-conversation-shares-list" data-received-conversation-shares-list>
          {visibleShares.map((share) => {
            const owner = share.owner_display_name
              ? `${share.owner_display_name} (@${share.owner_username})`
              : `@${share.owner_username}`;
            const accessibleName = `${share.title}${share.label ? `, ${share.label}` : ''}, shared by ${owner}, opens in a new tab`;
            return (
              <a
                key={share.id}
                className="received-conversation-share-row"
                data-share-id={share.id}
                href={`/shared/${encodeURIComponent(share.share_token)}`}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={accessibleName}
              >
                <span className="received-conversation-share-title">{share.title}</span>
                {share.label ? (
                  <span className="received-conversation-share-label">{share.label}</span>
                ) : null}
                <span className="received-conversation-share-owner">Shared by {owner}</span>
                <span className="received-conversation-share-context">
                  {getShareContext(share)}
                </span>
              </a>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
