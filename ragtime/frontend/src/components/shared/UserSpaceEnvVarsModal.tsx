import { useEffect, useMemo, useState } from 'react';
import { Check, Copy, Pencil, Trash2, X } from 'lucide-react';

import type { UpsertUserSpaceWorkspaceEnvVarRequest, UserSpaceWorkspaceEnvVar } from '@/types';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';

interface UserSpaceEnvVarsModalProps {
  isOpen: boolean;
  title?: string;
  onClose: () => void;
  envVars: UserSpaceWorkspaceEnvVar[];
  loading: boolean;
  saving: boolean;
  canManage: boolean;
  showReadonlyAsCompact?: boolean;
  helperText?: string;
  addLabel?: string;
  onCreateEnvVar: (request: UpsertUserSpaceWorkspaceEnvVarRequest) => Promise<void>;
  onUpdateEnvVar: (request: UpsertUserSpaceWorkspaceEnvVarRequest) => Promise<void>;
  onDeleteEnvVar: (key: string) => Promise<void>;
}

function normalizeDraftKey(value: string): string {
  return value
    .toUpperCase()
    .replace(/[^A-Z0-9_]/g, '_')
    .replace(/^[0-9]/, '_$&')
    .replace(/_+/g, '_');
}

export function UserSpaceEnvVarsModal({
  isOpen,
  title = 'Environment Variables',
  onClose,
  envVars,
  loading,
  saving,
  canManage,
  showReadonlyAsCompact = false,
  helperText,
  addLabel = 'Add variable',
  onCreateEnvVar,
  onUpdateEnvVar,
  onDeleteEnvVar,
}: UserSpaceEnvVarsModalProps) {
  const [draftEnvKey, setDraftEnvKey] = useState('');
  const [draftEnvValue, setDraftEnvValue] = useState('');
  const [draftEnvDescription, setDraftEnvDescription] = useState('');
  const [editingEnvKey, setEditingEnvKey] = useState<string | null>(null);
  const [editingEnvValueDraft, setEditingEnvValueDraft] = useState('');
  const [editingEnvDescKey, setEditingEnvDescKey] = useState<string | null>(null);
  const [editingEnvDescriptionDraft, setEditingEnvDescriptionDraft] = useState('');
  const [deletingEnvKey, setDeletingEnvKey] = useState<string | null>(null);
  const [confirmDeleteEnvKey, setConfirmDeleteEnvKey] = useState<string | null>(null);
  const [copiedEnvKey, setCopiedEnvKey] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setDraftEnvKey('');
    setDraftEnvValue('');
    setDraftEnvDescription('');
    setEditingEnvKey(null);
    setEditingEnvValueDraft('');
    setEditingEnvDescKey(null);
    setEditingEnvDescriptionDraft('');
    setDeletingEnvKey(null);
    setConfirmDeleteEnvKey(null);
    setCopiedEnvKey(null);
  }, [isOpen]);

  const sortedVars = useMemo(() => {
    return [...envVars].sort((a, b) => a.key.localeCompare(b.key));
  }, [envVars]);

  if (!isOpen) {
    return null;
  }

  const handleCreate = async () => {
    if (!canManage) {
      return;
    }
    const key = draftEnvKey.trim().toUpperCase();
    if (!key) {
      return;
    }
    await onCreateEnvVar({
      key,
      value: draftEnvValue || undefined,
      description: draftEnvDescription.trim() || undefined,
    });
    setDraftEnvKey('');
    setDraftEnvValue('');
    setDraftEnvDescription('');
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <p className="userspace-muted" style={{ marginBottom: 12 }}>
            {helperText || (
              <>
                Variables are encrypted at rest and injected into the devserver at runtime startup.
                Terminal and PTY sessions only expose redacted <code>printenv</code> and <code>env</code> output for configured keys.
                Reference them as <code>process.env.KEY</code> (Node.js) or <code>os.environ[&quot;KEY&quot;]</code> (Python).
              </>
            )}
          </p>
          {loading ? (
            <p className="userspace-muted">Loading...</p>
          ) : (
            <>
              {sortedVars.length > 0 && (
                <div className="userspace-env-var-list">
                  {sortedVars.map((envVar) => {
                    const rowReadOnly = !canManage || envVar.read_only === true;
                    const isEditingValue = editingEnvKey === envVar.key;
                    const isEditingDesc = editingEnvDescKey === envVar.key;
                    const showCompactReadonly = rowReadOnly && showReadonlyAsCompact;

                    return (
                      <div
                        key={`${envVar.source || 'workspace'}::${envVar.key}`}
                        className={`userspace-env-var-row${showCompactReadonly ? ' userspace-env-var-row-compact userspace-env-var-row-readonly' : ''}`}
                      >
                        <div className="userspace-env-var-primary">
                          <span className="userspace-env-var-key">
                            {envVar.key}
                            {envVar.source === 'global' && (
                              <span className="userspace-env-var-badge">Global</span>
                            )}
                            <button
                              className="userspace-env-var-copy-btn"
                              title="Copy key"
                              onClick={async () => {
                                await navigator.clipboard.writeText(envVar.key);
                                setCopiedEnvKey(envVar.key);
                                setTimeout(() => setCopiedEnvKey((c) => c === envVar.key ? null : c), 1500);
                              }}
                            >
                              {copiedEnvKey === envVar.key ? <Check size={13} /> : <Copy size={13} />}
                            </button>
                          </span>

                          {isEditingValue && !rowReadOnly ? (
                            <>
                              <input
                                type="password"
                                className="form-input userspace-env-var-value-input"
                                placeholder="New value (leave blank to keep current)"
                                value={editingEnvValueDraft}
                                onChange={(e) => setEditingEnvValueDraft(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') {
                                    void onUpdateEnvVar({
                                      key: envVar.key,
                                      value: editingEnvValueDraft || undefined,
                                    }).then(() => {
                                      setEditingEnvKey(null);
                                      setEditingEnvValueDraft('');
                                    });
                                  }
                                  if (e.key === 'Escape') {
                                    setEditingEnvKey(null);
                                    setEditingEnvValueDraft('');
                                  }
                                }}
                                autoFocus
                              />
                              <div className="userspace-env-var-actions">
                                <button
                                  className="btn btn-primary btn-sm"
                                  onClick={async () => {
                                    await onUpdateEnvVar({
                                      key: envVar.key,
                                      value: editingEnvValueDraft || undefined,
                                    });
                                    setEditingEnvKey(null);
                                    setEditingEnvValueDraft('');
                                  }}
                                  disabled={saving}
                                  title="Save"
                                >
                                  {saving ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                </button>
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => {
                                    setEditingEnvKey(null);
                                    setEditingEnvValueDraft('');
                                  }}
                                  title="Cancel"
                                >
                                  <X size={12} />
                                </button>
                              </div>
                            </>
                          ) : (
                            <>
                              <span className="userspace-env-var-value">
                                {envVar.has_value ? '••••••' : <em>not set</em>}
                              </span>
                              {!rowReadOnly && (
                                <div className="userspace-env-var-actions">
                                  {confirmDeleteEnvKey === envVar.key ? (
                                    <>
                                      <button
                                        className="btn btn-danger btn-sm"
                                        onClick={async () => {
                                          setDeletingEnvKey(envVar.key);
                                          try {
                                            await onDeleteEnvVar(envVar.key);
                                          } finally {
                                            setDeletingEnvKey(null);
                                          }
                                          setConfirmDeleteEnvKey(null);
                                        }}
                                        disabled={deletingEnvKey === envVar.key}
                                        title="Confirm delete"
                                      >
                                        {deletingEnvKey === envVar.key ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                      </button>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => setConfirmDeleteEnvKey(null)}
                                        title="Cancel"
                                      >
                                        <X size={12} />
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => {
                                          setEditingEnvKey(envVar.key);
                                          setEditingEnvValueDraft('');
                                        }}
                                        title="Edit value"
                                      >
                                        <Pencil size={12} />
                                      </button>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => setConfirmDeleteEnvKey(envVar.key)}
                                        title="Delete"
                                      >
                                        <Trash2 size={12} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              )}
                            </>
                          )}
                        </div>

                        {!showCompactReadonly && (
                          <div className="userspace-env-var-desc-row">
                            {isEditingDesc && !rowReadOnly ? (
                              <div className="userspace-env-var-desc-edit">
                                <input
                                  type="text"
                                  className="form-input userspace-env-var-desc-input"
                                  placeholder="Description (optional)"
                                  value={editingEnvDescriptionDraft}
                                  onChange={(e) => setEditingEnvDescriptionDraft(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') {
                                      void onUpdateEnvVar({
                                        key: envVar.key,
                                        description: editingEnvDescriptionDraft.trim() || undefined,
                                      }).then(() => {
                                        setEditingEnvDescKey(null);
                                        setEditingEnvDescriptionDraft('');
                                      });
                                    }
                                    if (e.key === 'Escape') {
                                      setEditingEnvDescKey(null);
                                      setEditingEnvDescriptionDraft('');
                                    }
                                  }}
                                  autoFocus
                                />
                                <button
                                  className="btn btn-primary btn-sm"
                                  onClick={async () => {
                                    await onUpdateEnvVar({
                                      key: envVar.key,
                                      description: editingEnvDescriptionDraft.trim() || undefined,
                                    });
                                    setEditingEnvDescKey(null);
                                    setEditingEnvDescriptionDraft('');
                                  }}
                                  disabled={saving}
                                  title="Save description"
                                >
                                  {saving ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                </button>
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => {
                                    setEditingEnvDescKey(null);
                                    setEditingEnvDescriptionDraft('');
                                  }}
                                  title="Cancel"
                                >
                                  <X size={12} />
                                </button>
                              </div>
                            ) : (
                              <div
                                className="userspace-env-var-desc-display"
                                onClick={() => {
                                  if (rowReadOnly) {
                                    return;
                                  }
                                  setEditingEnvDescKey(envVar.key);
                                  setEditingEnvDescriptionDraft(envVar.description ?? '');
                                }}
                              >
                                <span className="userspace-env-var-desc">
                                  {envVar.description || <em>No description</em>}
                                </span>
                                {!rowReadOnly && (
                                  <button
                                    className="inline-edit-btn userspace-env-var-desc-edit-btn"
                                    title="Edit description"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setEditingEnvDescKey(envVar.key);
                                      setEditingEnvDescriptionDraft(envVar.description ?? '');
                                    }}
                                  >
                                    <Pencil size={11} />
                                  </button>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {canManage && (
                <div className="userspace-env-var-form">
                  <strong className="userspace-env-var-form-title">{addLabel}</strong>
                  <div className="userspace-env-var-primary-row">
                    <input
                      type="text"
                      className="form-input userspace-env-var-key-input"
                      placeholder="KEY_NAME"
                      value={draftEnvKey}
                      onChange={(e) => setDraftEnvKey(e.target.value.toUpperCase())}
                      onBlur={() => setDraftEnvKey((k) => normalizeDraftKey(k))}
                    />
                    <input
                      className="form-input"
                      placeholder="Value (optional, leave blank for placeholder)"
                      type="password"
                      value={draftEnvValue}
                      onChange={(e) => setDraftEnvValue(e.target.value)}
                    />
                  </div>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Description (optional)"
                    value={draftEnvDescription}
                    onChange={(e) => setDraftEnvDescription(e.target.value)}
                  />
                  <div className="userspace-env-var-form-actions">
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => { void handleCreate(); }}
                      disabled={saving || !draftEnvKey.trim()}
                    >
                      {saving ? <MiniLoadingSpinner variant="icon" size={14} /> : <Check size={14} />}
                      Add
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
