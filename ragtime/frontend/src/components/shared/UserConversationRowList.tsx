import { useCallback, useState, type ReactNode } from 'react';
import { Check, Square, Trash2, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import type { Conversation } from '@/types';

interface UserConversationRowListProps {
  conversations: Conversation[];
  loading?: boolean;
  disabled?: boolean;
  onSelect: (conversation: Conversation) => void;
  onDelete: (conversationId: string) => Promise<void>;
  onCancelTask: (conversationId: string, taskId: string) => Promise<void>;
  renderMeta?: (conversation: Conversation) => ReactNode;
  emptyMessage?: string;
}

export function UserConversationRowList({
  conversations,
  loading = false,
  disabled = false,
  onSelect,
  onDelete,
  onCancelTask,
  renderMeta,
  emptyMessage = 'No chats.',
}: UserConversationRowListProps) {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [actionConversationId, setActionConversationId] = useState<string | null>(null);
  const [actionType, setActionType] = useState<'delete' | 'cancel' | null>(null);

  const handleDelete = useCallback(async (conversationId: string) => {
    setActionConversationId(conversationId);
    setActionType('delete');
    try {
      await onDelete(conversationId);
      setDeleteConfirmId(null);
    } finally {
      setActionConversationId(null);
      setActionType(null);
    }
  }, [onDelete]);

  const handleCancelTask = useCallback(async (conversationId: string, taskId: string) => {
    setActionConversationId(conversationId);
    setActionType('cancel');
    try {
      await onCancelTask(conversationId, taskId);
    } finally {
      setActionConversationId(null);
      setActionType(null);
    }
  }, [onCancelTask]);

  if (loading) {
    return (
      <div className="admin-ws-loading" style={{ padding: '10px 8px' }}>
        <MiniLoadingSpinner variant="icon" size={14} />
        <span>Loading chats...</span>
      </div>
    );
  }

  if (conversations.length === 0) {
    return <p className="muted-text" style={{ margin: 0 }}>{emptyMessage}</p>;
  }

  return (
    <div className="admin-ws-group-list">
      {conversations.map((conversation) => {
        const isConfirmingDelete = deleteConfirmId === conversation.id;
        const isActionTarget = actionConversationId === conversation.id;
        const isDeleting = isActionTarget && actionType === 'delete';
        const isCancelling = isActionTarget && actionType === 'cancel';
        const rowBusy = disabled || isDeleting || isCancelling;
        const title = conversation.title?.trim() || 'Untitled Chat';
        const messageCount = conversation.messages?.length ?? 0;
        const updatedAt = new Date(conversation.updated_at).toLocaleDateString();
        const metaContent = renderMeta
          ? renderMeta(conversation)
          : <span className="admin-ws-item-date">{messageCount} msg {updatedAt}</span>;

        return (
          <div key={conversation.id} className="admin-ws-item-wrapper">
            <div className="admin-ws-item">
              <button
                type="button"
                className="admin-ws-item-select"
                disabled={rowBusy}
                onClick={() => onSelect(conversation)}
                title={`Open chat: ${title}`}
              >
                <span className="admin-ws-item-name">{title}</span>
                {metaContent}
              </button>

              <div className="admin-ws-item-actions">
                {conversation.active_task_id && (
                  <button
                    type="button"
                    className="chat-action-btn"
                    disabled={rowBusy}
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleCancelTask(conversation.id, conversation.active_task_id as string);
                    }}
                    title="Cancel running task"
                  >
                    {isCancelling ? <MiniLoadingSpinner variant="icon" size={12} /> : <Square size={12} />}
                  </button>
                )}

                {isConfirmingDelete ? (
                  <>
                    <button
                      type="button"
                      className="chat-action-btn confirm-delete"
                      disabled={rowBusy}
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDelete(conversation.id);
                      }}
                      title="Confirm delete"
                    >
                      {isDeleting ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                    </button>
                    <button
                      type="button"
                      className="chat-action-btn cancel-delete"
                      disabled={rowBusy}
                      onClick={(event) => {
                        event.stopPropagation();
                        setDeleteConfirmId(null);
                      }}
                      title="Cancel"
                    >
                      <X size={12} />
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="chat-action-btn"
                    disabled={rowBusy}
                    onClick={(event) => {
                      event.stopPropagation();
                      setDeleteConfirmId(conversation.id);
                    }}
                    title="Delete chat"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default UserConversationRowList;
