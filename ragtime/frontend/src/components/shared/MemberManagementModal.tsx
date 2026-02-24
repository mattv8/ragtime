import { useCallback, useEffect, useState } from 'react';
import { X } from 'lucide-react';
import type { User } from '@/types';

export type MemberRole = 'owner' | 'editor' | 'viewer';

export interface Member {
  user_id: string;
  role: MemberRole;
}

interface MemberManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
  members: Member[];
  onSave: (members: Member[]) => Promise<void>;
  allUsers: User[];
  ownerId: string;
  entityType?: 'workspace' | 'conversation';
  formatUserLabel: (user?: Pick<User, 'username' | 'display_name'> | null, fallbackId?: string) => string;
  saving?: boolean;
}

export function MemberManagementModal({
  isOpen,
  onClose,
  members,
  onSave,
  allUsers,
  ownerId,
  entityType = 'workspace',
  formatUserLabel,
  saving = false,
}: MemberManagementModalProps) {
  const [pendingMembers, setPendingMembers] = useState<Member[]>([]);

  // Initialize pending members when modal opens or members prop changes
  useEffect(() => {
    if (isOpen) {
      // Filter out owner from members list (shown separately)
      setPendingMembers(members.filter((m) => m.user_id !== ownerId));
    }
  }, [isOpen, members, ownerId]);

  const handleAddMember = useCallback((userId: string) => {
    if (pendingMembers.some((m) => m.user_id === userId)) return;
    setPendingMembers((prev) => [...prev, { user_id: userId, role: 'viewer' }]);
  }, [pendingMembers]);

  const handleRemoveMember = useCallback((userId: string) => {
    setPendingMembers((prev) => prev.filter((m) => m.user_id !== userId));
  }, []);

  const handleChangeMemberRole = useCallback((userId: string, role: MemberRole) => {
    setPendingMembers((prev) => prev.map((m) => m.user_id === userId ? { ...m, role } : m));
  }, []);

  const handleSave = useCallback(async () => {
    await onSave(pendingMembers);
  }, [onSave, pendingMembers]);

  const handleClose = useCallback(() => {
    if (!saving) {
      onClose();
    }
  }, [onClose, saving]);

  if (!isOpen) return null;

  const ownerUser = allUsers.find((u) => u.id === ownerId);
  const entityLabel = entityType === 'workspace' ? 'Workspace' : 'Conversation';

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content modal-small userspace-members-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Manage {entityLabel} Members</h3>
          <button className="modal-close" onClick={handleClose} disabled={saving}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="userspace-members-list">
            {/* Owner row (cannot be modified) */}
            <div className="userspace-member-row userspace-member-owner">
              <span>{formatUserLabel(ownerUser, ownerId)}</span>
              <small className="userspace-muted">owner</small>
            </div>

            {/* Member rows */}
            {pendingMembers.map((member) => {
              const user = allUsers.find((u) => u.id === member.user_id);
              return (
                <div key={member.user_id} className="userspace-member-row">
                  <span>{formatUserLabel(user, member.user_id)}</span>
                  <div className="userspace-member-role-toggle" role="group" aria-label="Member role">
                    <button
                      type="button"
                      className={`userspace-member-role-option ${member.role === 'editor' ? 'active' : ''}`}
                      onClick={() => handleChangeMemberRole(member.user_id, 'editor')}
                      disabled={saving}
                    >
                      Editor
                    </button>
                    <button
                      type="button"
                      className={`userspace-member-role-option ${member.role === 'viewer' ? 'active' : ''}`}
                      onClick={() => handleChangeMemberRole(member.user_id, 'viewer')}
                      disabled={saving}
                    >
                      Viewer
                    </button>
                  </div>
                  <button
                    className="chat-action-btn"
                    onClick={() => handleRemoveMember(member.user_id)}
                    title="Remove member"
                    disabled={saving}
                  >
                    <X size={14} />
                  </button>
                </div>
              );
            })}
          </div>

          {/* Add member dropdown */}
          {allUsers.length > 0 && (
            <div className="userspace-add-member">
              <select
                id="userspace-add-member-select"
                defaultValue=""
                onChange={(e) => {
                  if (e.target.value) {
                    handleAddMember(e.target.value);
                    e.target.value = '';
                  }
                }}
                disabled={saving}
              >
                <option value="" disabled>Add a member...</option>
                {allUsers
                  .filter((u) => u.id !== ownerId && !pendingMembers.some((m) => m.user_id === u.id))
                  .map((u) => (
                    <option key={u.id} value={u.id}>{formatUserLabel(u, u.id)}</option>
                  ))}
              </select>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={handleClose} disabled={saving}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
