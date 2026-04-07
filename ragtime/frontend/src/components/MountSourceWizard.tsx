import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { Icon } from './Icon';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';
import { ToolWizard } from './ToolWizard';
import { X } from 'lucide-react';
import { api } from '@/api';
import type {
  UserSpaceObjectStorageConfig,
  UserSpaceObjectStorageBucket,
  UserspaceMountSource,
  ToolConfig,
  ToolType,
  FilesystemConnectionConfig,
  SSHShellConnectionConfig,
} from '@/types';

// ---------------------------------------------------------------------------
// Draft model — simplified, no connection fields (tool provides them)
// ---------------------------------------------------------------------------

export type MountSourceDraft = {
  id: string | null;
  name: string;
  description: string;
  enabled: boolean;
  tool_config_id: string | null;
  approved_paths: string[];
  sync_interval_seconds: number;
};

export function createEmptyMountSourceDraft(): MountSourceDraft {
  return {
    id: null,
    name: '',
    description: '',
    enabled: true,
    tool_config_id: null,
    approved_paths: ['.'],
    sync_interval_seconds: 30,
  };
}

export function mountSourceToDraft(source: UserspaceMountSource): MountSourceDraft {
  return {
    id: source.id,
    name: source.name,
    description: source.description || '',
    enabled: source.enabled,
    tool_config_id: source.tool_config_id,
    approved_paths: source.approved_paths.length > 0 ? [...source.approved_paths] : ['.'],
    sync_interval_seconds: source.sync_interval_seconds ?? 30,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MOUNT_TOOL_TYPES: ToolType[] = ['ssh_shell', 'filesystem_indexer'];

// Sync interval slider: exponential scale from 1 second to ~1 month
const SYNC_INTERVAL_MIN = 1;
const SYNC_INTERVAL_MAX = 2592000; // 30 days in seconds
const SYNC_INTERVAL_SCALE = Math.log(SYNC_INTERVAL_MAX / SYNC_INTERVAL_MIN);

function syncIntervalToSlider(seconds: number): number {
  if (seconds >= SYNC_INTERVAL_MAX) return 100;
  if (seconds <= SYNC_INTERVAL_MIN) return 0;
  return Math.max(0, Math.min(100, (Math.log(seconds / SYNC_INTERVAL_MIN) / SYNC_INTERVAL_SCALE) * 100));
}

function sliderToSyncInterval(slider: number): number {
  if (slider >= 100) return SYNC_INTERVAL_MAX;
  if (slider <= 0) return SYNC_INTERVAL_MIN;
  return Math.round(SYNC_INTERVAL_MIN * Math.exp((slider / 100) * SYNC_INTERVAL_SCALE));
}

function formatSyncInterval(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  if (d >= 7 && h === 0) {
    const w = Math.floor(d / 7);
    const rd = d % 7;
    return rd > 0 ? `${w}w ${rd}d` : `${w}w`;
  }
  return h > 0 ? `${d}d ${h}h` : `${d}d`;
}

function isMountTool(tool: ToolConfig): boolean {
  return MOUNT_TOOL_TYPES.includes(tool.tool_type);
}

function toolTypeLabel(tool: ToolConfig): string {
  if (tool.tool_type === 'ssh_shell') return 'SSH';
  const config = tool.connection_config as FilesystemConnectionConfig | undefined;
  const mt = config?.mount_type;
  if (mt === 'docker_volume') return 'Docker Volume';
  if (mt === 'smb') return 'SMB';
  if (mt === 'nfs') return 'NFS';
  if (mt === 'local') return 'Local Path';
  return 'Filesystem';
}

function toolTypeIcon(tool: ToolConfig): 'terminal' | 'database' | 'folder' | 'harddrive' | 'server' {
  if (tool.tool_type === 'ssh_shell') return 'terminal';
  const config = tool.connection_config as FilesystemConnectionConfig | undefined;
  const mt = config?.mount_type;
  if (mt === 'docker_volume') return 'harddrive';
  if (mt === 'local') return 'folder';
  if (mt === 'smb') return 'harddrive';
  if (mt === 'nfs') return 'server';
  return 'harddrive';
}

function toolSummary(tool: ToolConfig): string {
  if (tool.tool_type === 'ssh_shell') {
    const cfg = tool.connection_config as SSHShellConnectionConfig | undefined;
    if (cfg?.host) return `${cfg.user || 'root'}@${cfg.host}${cfg.port && cfg.port !== 22 ? ':' + cfg.port : ''}`;
    return 'SSH connection';
  }
  const cfg = tool.connection_config as FilesystemConnectionConfig | undefined;
  return cfg?.base_path || 'Filesystem';
}

function normalizeMountBrowserPath(value: string): string {
  const normalizedParts: string[] = [];
  for (const part of (value || '/').replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') { normalizedParts.pop(); continue; }
    normalizedParts.push(part);
  }
  return `/${normalizedParts.join('/')}`;
}

function browserPathToSourcePath(browserPath: string): string {
  const normalized = normalizeMountBrowserPath(browserPath);
  return normalized === '/' ? '.' : normalized.slice(1);
}

function sourcePathToBrowserPath(sourcePath: string): string {
  const normalized = (sourcePath || '').trim();
  if (!normalized || normalized === '.') return '/';
  return normalizeMountBrowserPath(`/${normalized}`);
}

// ---------------------------------------------------------------------------
// Wizard steps
// ---------------------------------------------------------------------------

type MountSourceWizardStep = 'select_tool' | 'mount_details' | 'review';

const WIZARD_STEPS: MountSourceWizardStep[] = ['select_tool', 'mount_details', 'review'];
const EDIT_WIZARD_STEPS: MountSourceWizardStep[] = ['mount_details', 'review'];

function getStepTitle(step: MountSourceWizardStep): string {
  switch (step) {
    case 'select_tool': return 'Select Backing Tool';
    case 'mount_details': return 'Mount Configuration';
    case 'review': return 'Review & Save';
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function deduplicateName(baseName: string, existingNames: string[]): string {
  const lowerNames = new Set(existingNames.map((n) => n.toLowerCase()));
  if (!lowerNames.has(baseName.toLowerCase())) return baseName;
  for (let i = 1; ; i++) {
    const candidate = `${baseName} (${i})`;
    if (!lowerNames.has(candidate.toLowerCase())) return candidate;
  }
}

interface MountSourceWizardProps {
  existingSource: UserspaceMountSource | null;
  existingNames?: string[];
  onClose: () => void;
  onSaved: (source: UserspaceMountSource) => void;
  embedded?: boolean;
}

export function MountSourceWizard({ existingSource, existingNames = [], onClose, onSaved, embedded = false }: MountSourceWizardProps) {
  const isEditing = existingSource !== null;
  const progressRef = useRef<HTMLDivElement>(null);

  const [draft, setDraft] = useState<MountSourceDraft>(() =>
    existingSource ? mountSourceToDraft(existingSource) : createEmptyMountSourceDraft()
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browserPath, setBrowserPath] = useState('');
  const [stagedDirectories, setStagedDirectories] = useState<string[]>([]);
  const [editingName, setEditingName] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Tool selection state
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [loadingTools, setLoadingTools] = useState(true);
  const [showToolWizard, setShowToolWizard] = useState(false);
  const [newToolType, setNewToolType] = useState<ToolType | undefined>(undefined);

  const wizardSteps = isEditing ? EDIT_WIZARD_STEPS : WIZARD_STEPS;
  const [currentStep, setCurrentStep] = useState<MountSourceWizardStep>(
    isEditing ? 'mount_details' : 'select_tool'
  );

  // Load available mount-compatible tools
  const loadTools = useCallback(async () => {
    setLoadingTools(true);
    try {
      const allTools = await api.listToolConfigs();
      setTools(allTools.filter(isMountTool));
    } catch {
      // Silently handle - empty list shown
    } finally {
      setLoadingTools(false);
    }
  }, []);

  useEffect(() => { void loadTools(); }, [loadTools]);

  // Auto-scroll active step into view
  useEffect(() => {
    const activeStep = progressRef.current?.querySelector('.wizard-step.active');
    if (activeStep) {
      activeStep.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentStep]);

  const selectedTool = tools.find((t) => t.id === draft.tool_config_id) ?? null;
  const isSSHSource = selectedTool?.tool_type === 'ssh_shell' || existingSource?.source_type === 'ssh';

  // Auto-fill name from tool when tool is selected and name is empty/default
  const namesForDedup = useMemo(() => existingNames, [existingNames]);

  useEffect(() => {
    if (!selectedTool) return;
    // Only auto-fill if the name is empty or was previously auto-derived from another tool
    const currentName = draft.name.trim();
    const wasAutoNamed = !currentName || tools.some((t) => {
      const base = t.name;
      return currentName === base || /^.+ \(\d+\)$/.test(currentName) && currentName.startsWith(base);
    });
    if (wasAutoNamed) {
      const derived = deduplicateName(selectedTool.name, namesForDedup);
      setDraft((d) => ({ ...d, name: derived }));
    }
  }, [selectedTool?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------

  const getCurrentStepIndex = () => wizardSteps.indexOf(currentStep);

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'select_tool':
        return draft.tool_config_id !== null;
      case 'mount_details':
        return draft.name.trim().length > 0 && draft.approved_paths.length > 0;
      case 'review':
        return true;
    }
  };

  const canNavigateToStep = (stepIndex: number): boolean => {
    if (stepIndex <= getCurrentStepIndex()) return true;
    if (stepIndex === getCurrentStepIndex() + 1 && canProceed()) return true;
    return false;
  };

  const goToStep = (step: MountSourceWizardStep) => {
    const stepIndex = wizardSteps.indexOf(step);
    if (canNavigateToStep(stepIndex)) {
      setCurrentStep(step);
      setError(null);
    }
  };

  const goToNextStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex < wizardSteps.length - 1) {
      setCurrentStep(wizardSteps[currentIndex + 1]);
      setError(null);
    }
  };

  const goToPreviousStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex > 0) {
      setCurrentStep(wizardSteps[currentIndex - 1]);
      setError(null);
    }
  };

  // ---------------------------------------------------------------------------
  // Approved paths helpers
  // ---------------------------------------------------------------------------

  const handleAddApprovedPath = useCallback(() => {
    const nextPath = browserPathToSourcePath(browserPath);
    setDraft((current) =>
      current.approved_paths.includes(nextPath)
        ? current
        : { ...current, approved_paths: [...current.approved_paths, nextPath].sort((a, b) => a.localeCompare(b)) }
    );
  }, [browserPath]);

  const handleRemoveApprovedPath = useCallback((path: string) => {
    setDraft((current) => {
      const remaining = current.approved_paths.filter((item) => item !== path);
      return { ...current, approved_paths: remaining.length > 0 ? remaining : ['.'] };
    });
  }, []);

  const browseMountSourcePath = useCallback(async (path: string) => {
    if (draft.id) {
      return api.browseUserspaceMountSource(draft.id, { path });
    }
    if (draft.tool_config_id) {
      return api.browseToolConfig(draft.tool_config_id, { path });
    }
    return { path, entries: [], error: 'Select a backing tool first.' };
  }, [draft.id, draft.tool_config_id]);

  const handleStageDirectory = useCallback((path: string) => {
    setStagedDirectories((prev) => prev.includes(path) ? prev : [...prev, path]);
  }, []);

  // ---------------------------------------------------------------------------
  // Tool wizard callback — after creating a new tool, select it
  // ---------------------------------------------------------------------------

  const handleToolWizardSaved = useCallback(async () => {
    setShowToolWizard(false);
    // Reload tools and auto-select the newest one
    try {
      const allTools = await api.listToolConfigs();
      const mountTools = allTools.filter(isMountTool);
      setTools(mountTools);
      // Select the most recently created tool (highest created_at)
      if (mountTools.length > 0) {
        const newest = mountTools.reduce((a, b) =>
          new Date(a.created_at) > new Date(b.created_at) ? a : b
        );
        setDraft((d) => ({ ...d, tool_config_id: newest.id }));
      }
    } catch {
      // Silently handle
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Save
  // ---------------------------------------------------------------------------

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      const approvedPaths = Array.from(
        new Set(draft.approved_paths.map((v) => v.trim()).filter(Boolean))
      );

      if (draft.id) {
        // Update existing
        const saved = await api.updateUserspaceMountSource(draft.id, {
          name: draft.name.trim(),
          description: null,
          enabled: true,
          approved_paths: approvedPaths.length > 0 ? approvedPaths : ['.'],
          sync_interval_seconds: draft.sync_interval_seconds,
        });
        onSaved(saved);
      } else {
        // Create new — pass tool_config_id so backend resolves source_type + connection
        const saved = await api.createUserspaceMountSource({
          name: draft.name.trim(),
          description: null,
          enabled: true,
          tool_config_id: draft.tool_config_id ?? undefined,
          approved_paths: approvedPaths.length > 0 ? approvedPaths : ['.'],
          sync_interval_seconds: draft.sync_interval_seconds,
        });
        onSaved(saved);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save mount source');
    } finally {
      setSaving(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Step renderers
  // ---------------------------------------------------------------------------

  const renderSelectTool = () => {
    if (showToolWizard) {
      return (
        <div className="wizard-step-content">
          <ToolWizard
            existingTool={null}
            onClose={() => setShowToolWizard(false)}
            onSave={handleToolWizardSaved}
            defaultToolType={newToolType}
            embedded={true}
            mountOnly={newToolType === 'filesystem_indexer'}
          />
        </div>
      );
    }

    return (
      <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
        <p className="field-help" style={{ margin: 0 }}>
          Choose an existing SSH or filesystem tool to back this mount source. The tool provides the connection credentials.
        </p>

        {loadingTools ? (
          <p className="muted">Loading tools...</p>
        ) : tools.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '24px 0' }}>
            <p className="muted">No SSH or filesystem tools configured yet.</p>
            <p className="muted" style={{ fontSize: '0.85rem' }}>Create one below to get started.</p>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
            {tools.map((tool) => (
              <button
                key={tool.id}
                type="button"
                className={`tool-type-option ${draft.tool_config_id === tool.id ? 'selected' : ''}`}
                style={!tool.enabled ? { opacity: 0.6 } : undefined}
                onClick={() => setDraft((d) => ({ ...d, tool_config_id: tool.id }))}
              >
                <div className="tool-type-option-icon">
                  <Icon name={toolTypeIcon(tool)} size={24} />
                </div>
                <div>
                  <span className="tool-type-option-name">{tool.name}{!tool.enabled && (
                      <span style={{ fontStyle: 'italic', fontWeight: 400, opacity: 0.7 }}> (disabled)</span>
                    )}</span>
                  <span className="tool-type-option-desc">
                    {toolTypeLabel(tool)}{' '}{toolSummary(tool)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => { setNewToolType('ssh_shell'); setShowToolWizard(true); }}
          >
            <Icon name="terminal" size={14} />
            New SSH Tool
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => { setNewToolType('filesystem_indexer'); setShowToolWizard(true); }}
          >
            <Icon name="folder" size={14} />
            New Filesystem Tool
          </button>
        </div>
      </div>
    );
  };

  const renderMountDetails = () => (
    <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
      {selectedTool && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', background: 'var(--color-bg-tertiary)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
          <Icon name={toolTypeIcon(selectedTool)} size={16} />
          <span className="muted" style={{ fontSize: '0.8rem' }}>{toolTypeLabel(selectedTool)}</span>
          <span className="muted" style={{ fontSize: '0.8rem' }}>{toolSummary(selectedTool)}</span>
          <span style={{ marginLeft: 'auto' }} />
          {editingName ? (
            <input
              ref={nameInputRef}
              type="text"
              value={draft.name}
              onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
              onBlur={() => setEditingName(false)}
              onKeyDown={(e) => { if (e.key === 'Enter') setEditingName(false); }}
              style={{ fontWeight: 500, fontSize: '0.9rem', padding: '2px 6px', border: '1px solid var(--color-border)', borderRadius: 4, background: 'var(--color-bg-primary)', color: 'inherit', width: 200 }}
              autoFocus
            />
          ) : (
            <span
              className="mount-source-name-display"
              style={{ fontWeight: 500, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6 }}
              onClick={() => { setEditingName(true); setTimeout(() => nameInputRef.current?.select(), 0); }}
              title="Click to rename"
            >
              {draft.name || '(unnamed)'}
              <Icon name="pencil" size={12} />
            </span>
          )}
        </div>
      )}

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <strong>Allowed Mount Roots</strong>
          <span className="muted" style={{ fontSize: '0.85rem' }}>
            Selections are relative to the source root.
          </span>
        </div>

        <div style={{ display: 'grid', gap: 12 }}>
          <ConstrainedPathBrowser
            currentPath={browserPath}
            rootPath="/"
            rootLabel="/"
            defaultExpanded={false}
            cacheKey={`mount-source-wizard:${draft.id ?? draft.tool_config_id ?? 'draft'}`}
            stagedDirectories={stagedDirectories}
            onStageDirectory={handleStageDirectory}
            emptyMessage="No directories found"
            onSelectPath={(selectedPath) => setBrowserPath(normalizeMountBrowserPath(selectedPath))}
            onBrowsePath={browseMountSourcePath}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <button type="button" className="btn btn-secondary" onClick={handleAddApprovedPath} disabled={!browserPath} style={{ padding: '6px 12px' }}>
              Add Selected Path
            </button>
            {draft.approved_paths.map((path) => (
              <div key={path} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: 6, fontSize: '0.85rem' }}>
                <code>{sourcePathToBrowserPath(path)}</code>
                <button type="button" className="btn btn-secondary" onClick={() => handleRemoveApprovedPath(path)} style={{ padding: '4px' }}>
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sync interval slider — SSH sources only */}
      {isSSHSource && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <strong>Auto-Sync Polling Interval</strong>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              {formatSyncInterval(draft.sync_interval_seconds)}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="muted" style={{ fontSize: '0.75rem', whiteSpace: 'nowrap' }}>1s</span>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              style={{ flex: 1 }}
              value={syncIntervalToSlider(draft.sync_interval_seconds)}
              onChange={(e) => {
                const val = sliderToSyncInterval(parseInt(e.target.value, 10));
                setDraft((d) => ({ ...d, sync_interval_seconds: val }));
              }}
            />
            <span className="muted" style={{ fontSize: '0.75rem', whiteSpace: 'nowrap' }}>30d</span>
          </div>
          <p className="field-help" style={{ marginTop: 4 }}>
            How often workspaces using this source check for changes when auto-sync is enabled.
            Lower values increase responsiveness but use more resources. Uses rsync for efficient delta transfers.
          </p>
        </div>
      )}
    </div>
  );

  const renderReview = () => (
    <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
      <table className="review-table">
        <tbody>
          <tr>
            <td className="review-label">Name</td>
            <td>{draft.name || <span className="muted">(not set)</span>}</td>
          </tr>
          <tr>
            <td className="review-label">Backing Tool</td>
            <td>
              {selectedTool ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Icon name={toolTypeIcon(selectedTool)} size={14} />
                  {selectedTool.name}
                  <span className="muted" style={{ fontSize: '0.8rem' }}>({toolTypeLabel(selectedTool)})</span>
                </span>
              ) : existingSource?.tool_name ? (
                <span>{existingSource.tool_name}</span>
              ) : (
                <span className="muted">(none selected)</span>
              )}
            </td>
          </tr>
          <tr>
            <td className="review-label">Allowed Mount Roots</td>
            <td>
              {draft.approved_paths.map((path) => (
                <code key={path} style={{ display: 'inline-block', marginRight: 8, marginBottom: 4 }}>
                  {sourcePathToBrowserPath(path)}
                </code>
              ))}
            </td>
          </tr>
          {isSSHSource && (
            <tr>
              <td className="review-label">Sync Interval</td>
              <td style={{ fontFamily: 'var(--font-mono)' }}>{formatSyncInterval(draft.sync_interval_seconds)}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );

  const renderStepContent = () => {
    switch (currentStep) {
      case 'select_tool': return renderSelectTool();
      case 'mount_details': return renderMountDetails();
      case 'review': return renderReview();
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {!embedded && (
        <div className="modal-header" style={{ borderBottom: 'none' }}>
          <h3>{isEditing ? 'Edit Mount Source' : 'New Mount Source'}</h3>
          <button type="button" className="modal-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
      )}

      {!showToolWizard && (
        <div className="wizard-progress" ref={progressRef} style={{ padding: '0 var(--space-lg)' }}>
          {wizardSteps.map((step, index) => {
            const stepIndex = wizardSteps.indexOf(step);
            const isNavigable = canNavigateToStep(stepIndex);
            return (
              <button
                key={step}
                type="button"
                className={`wizard-step ${currentStep === step ? 'active' : ''} ${getCurrentStepIndex() > index ? 'completed' : ''
                  } ${isNavigable ? 'navigable' : ''}`}
                onClick={() => goToStep(step)}
                disabled={!isNavigable}
              >
                <span className="step-number">{index + 1}</span>
                <span className="step-title">{getStepTitle(step)}</span>
              </button>
            );
          })}
        </div>
      )}

      {error && <div className="error-banner" style={{ margin: '0 var(--space-lg)' }}>{error}</div>}

      <div className="modal-body" style={{ flex: 1 }}>{renderStepContent()}</div>

      {!showToolWizard && (
        <div className="modal-footer" style={{ justifyContent: 'space-between' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={getCurrentStepIndex() === 0 ? onClose : goToPreviousStep}
          >
            {getCurrentStepIndex() === 0 ? 'Cancel' : 'Back'}
          </button>

          {currentStep === 'review' ? (
            <button
              type="button"
              className="btn"
              onClick={handleSave}
              disabled={saving || !draft.name.trim()}
            >
              {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Mount Source'}
            </button>
          ) : (
            <button
              type="button"
              className="btn"
              onClick={goToNextStep}
              disabled={!canProceed()}
            >
              Continue
            </button>
          )}
        </div>
      )}
    </div>
  );
}

type WorkspaceObjectStorageWizardStep = 'bucket_details' | 'review';

const WORKSPACE_OBJECT_STORAGE_WIZARD_STEPS: WorkspaceObjectStorageWizardStep[] = ['bucket_details', 'review'];

function getWorkspaceObjectStorageStepTitle(step: WorkspaceObjectStorageWizardStep): string {
  switch (step) {
    case 'bucket_details': return 'Bucket Details';
    case 'review': return 'Review & Save';
  }
}

interface WorkspaceObjectStorageWizardProps {
  workspaceId: string;
  existingBucket: UserSpaceObjectStorageBucket | null;
  existingBucketNames?: string[];
  onClose: () => void;
  onSaved: (config: UserSpaceObjectStorageConfig) => void;
  embedded?: boolean;
}

export function WorkspaceObjectStorageWizard({
  workspaceId,
  existingBucket,
  existingBucketNames = [],
  onClose,
  onSaved,
  embedded = false,
}: WorkspaceObjectStorageWizardProps) {
  const isEditing = existingBucket !== null;
  const progressRef = useRef<HTMLDivElement>(null);
  const [name, setName] = useState(existingBucket?.name ?? '');
  const [description, setDescription] = useState(existingBucket?.description ?? '');
  const [makeDefault, setMakeDefault] = useState(existingBucket?.is_default ?? !existingBucketNames.length);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<WorkspaceObjectStorageWizardStep>('bucket_details');

  useEffect(() => {
    const activeStep = progressRef.current?.querySelector('.wizard-step.active');
    if (activeStep) {
      activeStep.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentStep]);

  const normalizedName = name.trim().toLowerCase();
  const nameConflict = !isEditing && existingBucketNames.some((bucketName) => bucketName.toLowerCase() === normalizedName);
  const nameValid = /^[a-z0-9][a-z0-9-]*[a-z0-9]$/.test(normalizedName) && normalizedName.length >= 3 && normalizedName.length <= 63;
  const canProceed = currentStep === 'bucket_details'
    ? (isEditing || (nameValid && !nameConflict))
    : true;

  const getCurrentStepIndex = () => WORKSPACE_OBJECT_STORAGE_WIZARD_STEPS.indexOf(currentStep);

  const goToStep = (step: WorkspaceObjectStorageWizardStep) => {
    const targetIndex = WORKSPACE_OBJECT_STORAGE_WIZARD_STEPS.indexOf(step);
    if (targetIndex <= getCurrentStepIndex() || (targetIndex === getCurrentStepIndex() + 1 && canProceed)) {
      setCurrentStep(step);
      setError(null);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const saved = isEditing
        ? await api.updateUserSpaceObjectStorageBucket(workspaceId, existingBucket.name, {
          description: description.trim() || undefined,
          make_default: makeDefault,
        })
        : await api.createUserSpaceObjectStorageBucket(workspaceId, {
          name: normalizedName,
          description: description.trim() || undefined,
          make_default: makeDefault,
        });
      onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save object storage bucket');
    } finally {
      setSaving(false);
    }
  };

  const renderDetails = () => (
    <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
      <div style={{ display: 'grid', gap: 8 }}>
        <label style={{ display: 'grid', gap: 6 }}>
          <strong>Bucket Name</strong>
          <input
            type="text"
            className="form-input"
            value={name}
            disabled={isEditing}
            onChange={(event) => setName(event.target.value.replace(/[^a-zA-Z0-9-]/g, '-').toLowerCase())}
            placeholder="workspace-assets"
          />
        </label>
        {!isEditing && !nameValid && normalizedName.length > 0 && (
          <span className="field-help">Use 3-63 lowercase letters, numbers, or hyphens.</span>
        )}
        {!isEditing && nameConflict && (
          <span className="field-help" style={{ color: 'var(--color-error)' }}>A bucket with this name already exists in the workspace.</span>
        )}
      </div>

      <label style={{ display: 'grid', gap: 6 }}>
        <strong>Description</strong>
        <input
          type="text"
          className="form-input"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="Optional note for this bucket"
        />
      </label>

      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          checked={makeDefault}
          onChange={(event) => setMakeDefault(event.target.checked)}
        />
        <span>Use as the workspace default bucket</span>
      </label>

      <div style={{ padding: '12px 14px', borderRadius: 8, border: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)', display: 'grid', gap: 6 }}>
        <strong>Compatibility paths</strong>
        <span className="muted" style={{ fontSize: '0.85rem' }}>Public objects: <code>/{normalizedName || 'bucket'}/public</code></span>
        <span className="muted" style={{ fontSize: '0.85rem' }}>Private objects: <code>/{normalizedName || 'bucket'}/private</code></span>
      </div>
    </div>
  );

  const renderReview = () => (
    <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
      <table className="review-table">
        <tbody>
          <tr>
            <td className="review-label">Bucket</td>
            <td>{existingBucket?.name ?? normalizedName}</td>
          </tr>
          <tr>
            <td className="review-label">Description</td>
            <td>{description.trim() || <span className="muted">(none)</span>}</td>
          </tr>
          <tr>
            <td className="review-label">Default</td>
            <td>{makeDefault ? 'Yes' : 'No'}</td>
          </tr>
          <tr>
            <td className="review-label">Env Contract</td>
            <td>
              <code>RAGTIME_OBJECT_STORAGE_ENDPOINT</code>
              {' '}
              <code>RAGTIME_OBJECT_STORAGE_ACCESS_KEY_ID</code>
              {' '}
              <code>RAGTIME_OBJECT_STORAGE_SECRET_ACCESS_KEY</code>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {!embedded && (
        <div className="modal-header" style={{ borderBottom: 'none' }}>
          <h3>{isEditing ? 'Edit Object Storage Bucket' : 'New Object Storage Bucket'}</h3>
          <button type="button" className="modal-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
      )}

      <div className="wizard-progress" ref={progressRef} style={{ padding: '0 var(--space-lg)' }}>
        {WORKSPACE_OBJECT_STORAGE_WIZARD_STEPS.map((step, index) => {
          const isNavigable = index <= getCurrentStepIndex() || (index === getCurrentStepIndex() + 1 && canProceed);
          return (
            <button
              key={step}
              type="button"
              className={`wizard-step ${currentStep === step ? 'active' : ''} ${getCurrentStepIndex() > index ? 'completed' : ''} ${isNavigable ? 'navigable' : ''}`}
              onClick={() => goToStep(step)}
              disabled={!isNavigable}
            >
              <span className="step-number">{index + 1}</span>
              <span className="step-title">{getWorkspaceObjectStorageStepTitle(step)}</span>
            </button>
          );
        })}
      </div>

      {error && <div className="error-banner" style={{ margin: '0 var(--space-lg)' }}>{error}</div>}

      <div className="modal-body" style={{ flex: 1 }}>
        {currentStep === 'bucket_details' ? renderDetails() : renderReview()}
      </div>

      <div className="modal-footer" style={{ justifyContent: 'space-between' }}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={getCurrentStepIndex() === 0 ? onClose : () => setCurrentStep('bucket_details')}
        >
          {getCurrentStepIndex() === 0 ? 'Cancel' : 'Back'}
        </button>
        {currentStep === 'review' ? (
          <button type="button" className="btn" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : isEditing ? 'Save Bucket' : 'Create Bucket'}
          </button>
        ) : (
          <button type="button" className="btn" onClick={() => setCurrentStep('review')} disabled={!canProceed}>
            Continue
          </button>
        )}
      </div>
    </div>
  );
}
