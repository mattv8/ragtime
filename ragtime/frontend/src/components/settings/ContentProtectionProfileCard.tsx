import { useEffect, useRef, useState, type KeyboardEvent, type RefObject } from 'react';
import { Pencil, Shield } from 'lucide-react';

import type { ContentProtectionProfile } from '@/api/contentProtection';

export interface ContentProtectionProfileCardProps {
  profile: ContentProtectionProfile;
  affectedGroups: number;
  onUpdate: (id: string, change: Partial<ContentProtectionProfile>) => void;
  onDelete: (profile: ContentProtectionProfile) => void;
}

type EditableField = 'name' | 'scope' | 'level';

export function ContentProtectionProfileCard({
  profile,
  affectedGroups,
  onUpdate,
  onDelete,
}: ContentProtectionProfileCardProps): JSX.Element {
  const [editingField, setEditingField] = useState<EditableField | null>(null);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const cancellingRef = useRef(false);

  useEffect(() => {
    if (!editingField) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [editingField]);

  const valueFor = (field: EditableField) => {
    if (field === 'name') return profile.name;
    if (field === 'scope') return profile.scope;
    return String(profile.level);
  };
  const startEdit = (field: EditableField) => {
    cancellingRef.current = false;
    setEditValue(valueFor(field));
    setEditingField(field);
  };
  const cancelEdit = () => {
    cancellingRef.current = true;
    setEditingField(null);
  };
  const commitEdit = () => {
    if (cancellingRef.current) {
      cancellingRef.current = false;
      return;
    }
    if (!editingField) return;

    if (editingField === 'name') {
      const name = editValue.trim();
      if (name && name !== profile.name) onUpdate(profile.id, { name });
    } else if (editingField === 'scope') {
      if (editValue !== profile.scope) onUpdate(profile.id, { scope: editValue });
    } else {
      const value = Number(editValue);
      if (Number.isFinite(value)) {
        const level = Math.max(0, Math.min(2, Math.round(value)));
        if (level !== profile.level) onUpdate(profile.id, { level });
      }
    }
    setEditingField(null);
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      cancelEdit();
    } else if (
      event.key === 'Enter' &&
      (editingField === 'name' || editingField === 'level' || event.ctrlKey)
    ) {
      event.preventDefault();
      commitEdit();
    }
  };
  const groupLabel = affectedGroups
    ? `${affectedGroups} group${affectedGroups === 1 ? '' : 's'}`
    : 'Unassigned';

  return (
    <article className="tool-card content-protection-profile" data-profile-id={profile.id}>
      <div className="tool-card-header" data-profile-header>
        <div className="tool-card-icon">
          <Shield aria-hidden="true" size={28} />
        </div>
        <div className="tool-card-header-content">
          <div className="tool-card-header-main">
            {editingField === 'name' ? (
              <div className="tool-card-title inline-edit-field">
                <input
                  ref={inputRef as RefObject<HTMLInputElement>}
                  className="inline-edit-input"
                  type="text"
                  aria-label={`${profile.name} name`}
                  value={editValue}
                  onChange={(event) => setEditValue(event.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={commitEdit}
                />
              </div>
            ) : (
              <div
                className="tool-card-title editable-field-wrapper name-wrapper"
                onClick={() => startEdit('name')}
              >
                <h3>{profile.name}</h3>
                <button
                  type="button"
                  className="inline-edit-btn"
                  aria-label={`Edit ${profile.name} name`}
                  title={`Edit ${profile.name} name`}
                  onClick={(event) => {
                    event.stopPropagation();
                    startEdit('name');
                  }}
                >
                  <Pencil size={14} aria-hidden="true" />
                </button>
              </div>
            )}
            <div className="tool-card-heartbeat content-protection-profile-level">
              {editingField === 'level' ? (
                <input
                  ref={inputRef as RefObject<HTMLInputElement>}
                  className="content-protection-profile-level-input"
                  type="number"
                  min="0"
                  max="2"
                  step="1"
                  aria-label={`${profile.name} level`}
                  value={editValue}
                  onChange={(event) => setEditValue(event.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={commitEdit}
                />
              ) : (
                <button
                  type="button"
                  className="tool-badge content-protection-profile-level-badge"
                  aria-label={`Edit ${profile.name} level`}
                  onClick={() => startEdit('level')}
                >
                  Level {profile.level}
                </button>
              )}
            </div>
          </div>
          <div className="tool-card-meta-row">
            <div className="tool-card-badges">
              <span className="tool-badge">{groupLabel}</span>
            </div>
          </div>
        </div>
      </div>

      {editingField === 'scope' ? (
        <div className="inline-edit-field description-edit">
          <textarea
            ref={inputRef as RefObject<HTMLTextAreaElement>}
            className="inline-edit-textarea"
            aria-label={`${profile.name} scope`}
            rows={3}
            value={editValue}
            onChange={(event) => setEditValue(event.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={commitEdit}
          />
        </div>
      ) : (
        <div
          className="editable-field-wrapper description-wrapper"
          onClick={() => startEdit('scope')}
        >
          <p className={`tool-card-description${profile.scope ? '' : ' placeholder'}`}>
            {profile.scope || 'Describe the information this profile permits...'}
          </p>
          <button
            type="button"
            className="inline-edit-btn"
            aria-label={`Edit ${profile.name} permitted information`}
            title={`Edit ${profile.name} permitted information`}
            onClick={(event) => {
              event.stopPropagation();
              startEdit('scope');
            }}
          >
            <Pencil size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      <div className="tool-card-footer" data-profile-footer>
        <div className="tool-card-actions">
          <button type="button" className="btn btn-sm btn-danger" onClick={() => onDelete(profile)}>
            Delete{affectedGroups ? ` (${affectedGroups} groups)` : ''}
          </button>
        </div>
      </div>
    </article>
  );
}
