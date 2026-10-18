import { useState } from 'react';
import { ChevronDown, ChevronUp, Clock, Pencil, RefreshCw, X } from 'lucide-react';

import type { QueuedMessageItem } from '@/types';
import type { QueueState } from '@/hooks/useConversationMessageQueue';

interface QueuedMessagesStripProps {
  conversationId: string;
  state: QueueState;
  draftIsEmpty: boolean;
  itemErrors?: Record<string, string>;
  onEdit: (itemId: string) => void;
  onDelete: (itemId: string) => void;
  onDispatch: () => void;
  onRetry: () => void;
}

function canonicalMessagePreview(message: string): string {
  try {
    const content = JSON.parse(message) as unknown;
    if (Array.isArray(content)) {
      return content
        .filter(
          (part): part is { type: 'text'; text: string } =>
            typeof part === 'object' &&
            part !== null &&
            (part as { type?: string }).type === 'text' &&
            typeof (part as { text?: unknown }).text === 'string',
        )
        .map((part) => part.text)
        .join(' ');
    }
  } catch {
    // Plain text messages are not JSON.
  }
  return message;
}

function preview(item: QueuedMessageItem): string {
  const displayPreview = item.segments
    .map((segment) => {
      if (segment?.type === 'text' && typeof segment.text === 'string') return segment.text;
      if (segment?.type === 'ref' && typeof segment.reference?.path === 'string') {
        return `[${segment.reference.path}]`;
      }
      return '[Unsupported queued content]';
    })
    .join(' ');
  return displayPreview || canonicalMessagePreview(item.message);
}

function attachmentCount(item: QueuedMessageItem): number {
  try {
    const content = JSON.parse(item.message) as unknown;
    if (!Array.isArray(content)) return 0;
    return content.filter(
      (part) =>
        typeof part === 'object' &&
        part !== null &&
        ((part as { type?: string }).type === 'file' ||
          (part as { type?: string }).type === 'image_url'),
    ).length;
  } catch {
    return 0;
  }
}

function QueueRow({
  item,
  draftIsEmpty,
  onEdit,
  onDelete,
  itemErrors,
}: Pick<QueuedMessagesStripProps, 'draftIsEmpty' | 'onEdit' | 'onDelete'> & {
  item: QueuedMessageItem;
  itemErrors?: Record<string, string>;
}) {
  const editTitle = draftIsEmpty
    ? 'Edit queued message'
    : 'Clear your draft first to edit a queued message';
  const attachments = attachmentCount(item);
  const disabledReasonId = `queue-edit-disabled-reason-${item.id}`;
  return (
    <div className="queue-strip-item">
      <div className="queue-strip-row" data-queue-item-id={item.id}>
        <span className="queue-strip-preview">
          {preview(item)}
          {attachments > 0 && (
            <span className="queue-strip-attachment-count"> (+{attachments} 📎)</span>
          )}
        </span>
        <button
          type="button"
          className="chat-action-icon-btn queue-strip-edit-btn"
          aria-label="Edit queued message"
          aria-describedby={!draftIsEmpty ? disabledReasonId : undefined}
          title={editTitle}
          disabled={!draftIsEmpty}
          onClick={() => onEdit(item.id)}
        >
          <Pencil size={14} aria-hidden="true" />
        </button>
        {!draftIsEmpty && (
          <span className="queue-strip-visually-hidden" id={disabledReasonId}>
            Clear your draft first to edit a queued message
          </span>
        )}
        <button
          type="button"
          className="chat-action-icon-btn queue-strip-cancel-btn"
          aria-label="Remove queued message"
          title="Remove queued message"
          onClick={() => onDelete(item.id)}
        >
          <X size={16} aria-hidden="true" />
        </button>
      </div>
      {(item.dispatch_error || itemErrors?.[item.id]) && (
        <div className="queue-strip-item-error" role="alert">
          {item.dispatch_error || itemErrors?.[item.id]}
        </div>
      )}
    </div>
  );
}

export function QueuedMessagesStrip({
  conversationId,
  state,
  draftIsEmpty,
  itemErrors,
  onEdit,
  onDelete,
  onDispatch,
  onRetry,
}: QueuedMessagesStripProps) {
  const [expanded, setExpanded] = useState(false);
  if (state.loading)
    return (
      <div
        className="queue-strip queue-strip--loading"
        role="group"
        aria-busy="true"
        aria-label="Loading queued messages"
      >
        <div className="queue-strip-skeleton-row">
          <div className="chat-skeleton-line" style={{ width: '60%' }} />
        </div>
      </div>
    );
  if (state.error)
    return (
      <div className="queue-strip queue-strip--error" role="alert">
        <span className="queue-strip-error-text">Could not load queue</span>
        <button type="button" className="chat-action-icon-btn" aria-label="Retry" onClick={onRetry}>
          <RefreshCw size={14} aria-hidden="true" />
        </button>
      </div>
    );
  if (!state.items.length) return null;
  const multiple = state.items.length > 1;
  const listId = `queue-strip-list-${conversationId}`;
  return (
    <div
      className="queue-strip"
      id={`queue-strip-${conversationId}`}
      role="group"
      aria-label="Queued messages"
    >
      {multiple ? (
        <>
          <div className="queue-strip-header">
            <Clock size={14} aria-hidden="true" />
            <span className="queue-strip-count" role="status" aria-live="polite" aria-atomic="true">
              Queued {state.items.length}
              {state.held && <span className="queue-strip-paused-badge"> · Paused</span>}
            </span>
            <button
              type="button"
              className="chat-action-icon-btn queue-strip-toggle"
              aria-label={expanded ? 'Collapse queued messages' : 'Expand queued messages'}
              aria-expanded={expanded}
              aria-controls={listId}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? (
                <ChevronUp size={16} aria-hidden="true" />
              ) : (
                <ChevronDown size={16} aria-hidden="true" />
              )}
            </button>
            {state.held && (
              <button
                type="button"
                className="btn btn-sm btn-secondary queue-strip-send-now-btn"
                aria-label="Resume queue"
                onClick={onDispatch}
              >
                Send now
              </button>
            )}
          </div>
          {expanded && (
            <div className="queue-strip-list" id={listId}>
              {state.items.map((item) => (
                <QueueRow
                  key={item.id}
                  item={item}
                  draftIsEmpty={draftIsEmpty}
                  onEdit={onEdit}
                  onDelete={onDelete}
                  itemErrors={itemErrors}
                />
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="queue-strip-single">
          <span
            className="queue-strip-single-status"
            role="status"
            aria-live="polite"
            aria-atomic="true"
            aria-label="1 message queued"
          >
            1 message queued
          </span>
          <Clock size={14} aria-hidden="true" />
          <QueueRow
            item={state.items[0]}
            draftIsEmpty={draftIsEmpty}
            onEdit={onEdit}
            onDelete={onDelete}
            itemErrors={itemErrors}
          />
          {state.held && <span className="queue-strip-paused-badge">Paused</span>}
          {state.held && (
            <button
              type="button"
              className="btn btn-sm btn-secondary queue-strip-send-now-btn"
              aria-label="Resume queue"
              onClick={onDispatch}
            >
              Send now
            </button>
          )}
        </div>
      )}
    </div>
  );
}
