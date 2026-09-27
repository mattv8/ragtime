import type {
  Conversation,
  ConversationSummary,
  User,
  UserSpaceWorkspace,
  WorkspaceConversationStateSummaryItem,
} from '@/types';
import { WorkspaceRowList } from '../shared/WorkspaceRowList';
import { UserConversationRowList } from '../shared/UserConversationRowList';

interface Props {
  user: User;
  users: User[];
  workspaces: UserSpaceWorkspace[];
  chats: ConversationSummary[];
  chatsLoaded: boolean;
  chatsLoading: boolean;
  chatsError: string | null;
  deletingWorkspaceIds: ReadonlySet<string>;
  workspaceStateById: Record<string, WorkspaceConversationStateSummaryItem>;
  workspaceLastMessageAtById: Record<string, string | null>;
  workspaceLastConversationById: Record<string, ConversationSummary | null>;
  workspaceMetaLoading: boolean;
  storageByWorkspaceId: Record<string, number>;
  storageFailuresByWorkspaceId: Record<string, string>;
  storageLoading: boolean;
  onLoad: () => void;
  onComputeStorage: () => void;
  onWorkspaceDelete: (id: string) => Promise<void>;
  onWorkspaceTransfer: (id: string, owner: string) => Promise<void>;
  onOpenWorkspace: (id: string) => void;
  onOpenChat?: (id: string) => void;
  onDeleteChat: (id: string) => Promise<void>;
  onCancelChat: (id: string, task: string) => Promise<void>;
  formatBytes: (bytes: number) => string;
  formatDateTime: (value: string | null | undefined) => string;
  getConversationContextMeta: (
    conversation: Conversation | ConversationSummary | null | undefined,
  ) => string;
}

export function UserResourcesTab(props: Props) {
  const owned = props.workspaces.filter((workspace) => workspace.owner_user_id === props.user.id);
  const memberships = props.workspaces.filter(
    (workspace) =>
      workspace.owner_user_id !== props.user.id &&
      workspace.members.some((member) => member.user_id === props.user.id),
  );
  const knownStorage = owned.filter(
    (workspace) => props.storageByWorkspaceId[workspace.id] !== undefined,
  );
  const failedStorage = owned.filter(
    (workspace) => props.storageFailuresByWorkspaceId[workspace.id],
  );
  const storageBytes = knownStorage.reduce(
    (total, workspace) => total + props.storageByWorkspaceId[workspace.id],
    0,
  );
  const storageLabel =
    owned.length === 0
      ? 'No owned workspaces.'
      : `${props.formatBytes(storageBytes)} sampled for ${knownStorage.length}/${owned.length} workspaces${failedStorage.length ? `; ${failedStorage.length} failed` : ''}.`;

  const workspaceMeta = (workspace: UserSpaceWorkspace) => {
    const state = props.workspaceStateById[workspace.id];
    const latest = props.workspaceLastConversationById[workspace.id];
    const status = state?.has_live_task
      ? 'Live'
      : state?.has_interrupted_task
        ? 'Interrupted'
        : 'Idle';
    const storage = props.storageByWorkspaceId[workspace.id];
    const storageFailure = props.storageFailuresByWorkspaceId[workspace.id];
    return (
      <span className="user-resource-meta">
        {status} · {latest?.model || 'Unknown model'} · {props.getConversationContextMeta(latest)}{' '}
        context · last message{' '}
        {props.workspaceMetaLoading && !(workspace.id in props.workspaceLastMessageAtById)
          ? 'loading…'
          : props.formatDateTime(props.workspaceLastMessageAtById[workspace.id])}{' '}
        ·{' '}
        {storage === undefined
          ? storageFailure
            ? 'storage unavailable'
            : 'storage not sampled'
          : props.formatBytes(storage)}
      </span>
    );
  };

  return (
    <section
      id={`user-resources-${props.user.id}`}
      className="user-resources-tab"
      data-user-resources-tab
    >
      <div className="user-resources-heading">
        <div>
          <h4>Owned workspaces ({owned.length})</h4>
          <p className="field-help">{storageLabel}</p>
        </div>
        {owned.length > 0 && (
          <button
            type="button"
            className="btn btn-secondary"
            disabled={props.storageLoading}
            onClick={props.onComputeStorage}
          >
            {props.storageLoading
              ? 'Sampling storage…'
              : knownStorage.length === owned.length
                ? 'Refresh storage'
                : failedStorage.length
                  ? 'Retry failed storage'
                  : 'Sample storage'}
          </button>
        )}
      </div>
      <WorkspaceRowList
        workspaces={owned}
        users={props.users}
        deletingWorkspaceIds={props.deletingWorkspaceIds}
        onTransfer={props.onWorkspaceTransfer}
        onDelete={props.onWorkspaceDelete}
        onSelect={(workspace) => props.onOpenWorkspace(workspace.id)}
        renderMeta={workspaceMeta}
        emptyMessage="No owned workspaces."
      />
      <h4>Workspace memberships ({memberships.length})</h4>
      <WorkspaceRowList
        workspaces={memberships}
        users={props.users}
        disabled
        onTransfer={props.onWorkspaceTransfer}
        onDelete={props.onWorkspaceDelete}
        onSelect={(workspace) => props.onOpenWorkspace(workspace.id)}
        renderMeta={workspaceMeta}
        emptyMessage="No workspace memberships."
      />
      <div className="user-resources-heading">
        <h4>Standalone chats {props.chatsLoaded ? `(${props.chats.length})` : ''}</h4>
        {!props.chatsLoading && (!props.chatsLoaded || props.chatsError) && (
          <button type="button" className="btn btn-secondary" onClick={props.onLoad}>
            {props.chatsError ? 'Retry chats' : 'Load chats'}
          </button>
        )}
      </div>
      {props.chatsError ? (
        <p className="form-error" role="alert">
          Chats could not be loaded: {props.chatsError}
        </p>
      ) : (
        <UserConversationRowList
          conversations={props.chats}
          loading={!props.chatsLoaded || props.chatsLoading}
          onSelect={(conversation) => props.onOpenChat?.(conversation.id)}
          onDelete={props.onDeleteChat}
          onCancelTask={props.onCancelChat}
          renderMeta={(conversation) => (
            <span className="user-resource-meta">
              {conversation.model || 'Unknown model'} · {conversation.message_count} msg ·{' '}
              {props.getConversationContextMeta(conversation)} context · last message{' '}
              {props.formatDateTime(conversation.updated_at)}
            </span>
          )}
          emptyMessage="No standalone chats."
        />
      )}
    </section>
  );
}
