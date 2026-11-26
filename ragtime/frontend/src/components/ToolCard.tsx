import { Children, Fragment, type ReactNode, useEffect, useRef, useState } from 'react';
import { MoreHorizontal } from 'lucide-react';
import type {
  ToolConfig,
  HeartbeatStatus,
  PdmIndexStats,
  SchemaIndexJob,
  SchemaIndexStats,
  InstructionSkillCount,
  SolidworksHostAutoUpdate,
  TunnelClientState,
} from '@/types';
import { TOOL_TYPE_INFO } from '@/types';
import {
  hostUpdateBadgeKind,
  hostUpdateFailureText,
  hostUpdateWaitingText,
  type HostUpdateBadgeKind,
} from '@/utils/solidworksHostUpdate';
import {
  getToolConnectionSummary,
  formatToolCount,
  formatToolFactDate,
  getToolWorkingDirectory,
  hasSchemaIndexingEnabled,
  isPairedWindowsTool,
} from '@/utils/toolConfigDisplay';
import {
  describePdmLinkedNote,
  formatPdmSchedule,
  getPdmLinkRelations,
} from '@/utils/solidworksPdmLink';
import { useIsTruncated } from '@/utils/useIsTruncated';
import { useFitLineClamp } from '@/utils/useFitLineClamp';
import type { SolidworksConnectionConfig } from '@/api/solidworks';
import { Icon, getToolIconType } from './Icon';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { IndexingPill } from './IndexingPill';
import { Popover } from './Popover';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

// Inline field being edited
type EditingField = 'name' | 'description' | null;

const HEARTBEAT_ERROR_SOFT_WRAP_LENGTH = 30;
const HEARTBEAT_ERROR_LONG_TOKEN_PATTERN = new RegExp(
  `(\\S{${HEARTBEAT_ERROR_SOFT_WRAP_LENGTH + 1},})`,
);
const HEARTBEAT_ERROR_CHUNK_PATTERN = new RegExp(`.{1,${HEARTBEAT_ERROR_SOFT_WRAP_LENGTH}}`, 'g');

function addHeartbeatErrorSoftWraps(error: string) {
  return error.split(HEARTBEAT_ERROR_LONG_TOKEN_PATTERN).map((part, partIndex) => {
    if (!HEARTBEAT_ERROR_LONG_TOKEN_PATTERN.test(part)) {
      return part;
    }

    const chunks = part.match(HEARTBEAT_ERROR_CHUNK_PATTERN);
    if (!chunks) {
      return part;
    }

    return chunks.map((chunk, chunkIndex) => (
      <Fragment key={`${partIndex}-${chunkIndex}`}>
        {chunk}
        {chunkIndex < chunks.length - 1 && <wbr />}
      </Fragment>
    ));
  });
}

export interface ToolCardProps {
  tool: ToolConfig;
  tools: ToolConfig[];
  targetId?: string;
  heartbeat: HeartbeatStatus | null;
  showFooterActions: boolean;
  onEdit: (tool: ToolConfig) => void;
  onDelete: (tool: ToolConfig) => void;
  onTest: (toolId: string) => void;
  testing: boolean;
  onPdmReindex?: (toolId: string, fullReindex: boolean) => void;
  pdmIndexing?: boolean;
  pdmStats?: PdmIndexStats | null;
  onSchemaReindex?: (toolId: string, fullReindex: boolean) => void;
  schemaIndexing?: boolean;
  activeSchemaJob?: SchemaIndexJob | null;
  schemaStats?: SchemaIndexStats | null;
  onInlineUpdate?: (
    toolId: string,
    updates: { name?: string; description?: string },
  ) => Promise<void>;
  classified?: boolean;
  skillCount?: InstructionSkillCount;
  onOpenSkills: (tool: ToolConfig) => void;
  onOpenActions?: (tool: ToolConfig, anchor: HTMLElement) => void;
  actionsMenuOpen?: boolean;
}

function formatInstructionSkillCount(count: InstructionSkillCount): string {
  return count.total === 1 ? '1 skill' : `${count.total} skills`;
}

function hostUpdatePendingText(update: SolidworksHostAutoUpdate): string {
  const { installed_version: installed, target_version: target } = update;
  const versions =
    installed && target
      ? `Version ${installed} is running; ${target} will be installed automatically. `
      : '';
  return `${versions}${hostUpdateWaitingText(update)}`;
}

function hostUpdateBadgeText(update: SolidworksHostAutoUpdate, kind: HostUpdateBadgeKind): string {
  if (kind === 'pending') return hostUpdatePendingText(update);
  if (kind === 'updating') {
    const target = update.target_version ? ` to version ${update.target_version}` : '';
    return `Ragtime is updating the agent${target}. Requests wait until it finishes.`;
  }
  return `${hostUpdateFailureText(update)} Open the tool to update the agent yourself.`;
}

function tunnelClientOutdatedText(client: TunnelClientState): string {
  const prefix = 'The tunnel keeps working.';
  const suffix = 'Ragtime cannot update tunnel clients yet.';
  if (client.version === null) {
    return `${prefix} This machine's tunnel client did not report a version, so it is older than version ${client.current_version}. ${suffix}`;
  }
  return `${prefix} This machine's tunnel client is version ${client.version}; the newest is version ${client.current_version}. ${suffix}`;
}

interface DetailBadgeProps {
  hook: string;
  className: string;
  icon: ReactNode;
  label: string;
  detail: string;
}

/**
 * A focusable badge whose popover detail is also rendered as visually hidden text, so keyboard and
 * screen-reader users get it from the badge itself (the shared Popover has no aria-describedby).
 */
function DetailBadge({ hook, className, icon, label, detail }: DetailBadgeProps) {
  return (
    <Popover content={detail} position="top" style={{ display: 'inline-flex' }}>
      <span className={className} data-tool-badge={hook} tabIndex={0}>
        {icon}
        <span className="path-text">{label}</span>
        <span className="rc-visually-hidden">{detail}</span>
      </span>
    </Popover>
  );
}

interface HostUpdateBadgeProps {
  update: SolidworksHostAutoUpdate;
  kind: HostUpdateBadgeKind;
}

function BadgeRow({ children }: { children: ReactNode }) {
  const badges = Children.toArray(children);
  if (badges.length === 0) return null;
  return (
    <div className="tool-card-meta-row">
      <div className="tool-card-badges">{badges}</div>
    </div>
  );
}

function HostUpdateBadge({ update, kind }: HostUpdateBadgeProps) {
  const badge = {
    pending: {
      label: 'Agent update pending',
      hook: 'host-update-pending',
      className: 'tool-badge',
      icon: <Icon name="refresh" size={12} />,
    },
    updating: {
      label: 'Updating agent…',
      hook: 'host-updating',
      className: 'tool-badge',
      icon: <span className="rc-polling-indicator" aria-hidden="true" />,
    },
    failed: {
      label: 'Agent update failed',
      hook: 'host-update-failed',
      className: 'tool-badge badge-warning',
      icon: <Icon name="alert-triangle" size={12} />,
    },
  }[kind];
  return (
    <DetailBadge
      hook={badge.hook}
      className={badge.className}
      icon={badge.icon}
      label={badge.label}
      detail={hostUpdateBadgeText(update, kind)}
    />
  );
}

export function ToolCard({
  tool,
  tools,
  targetId,
  heartbeat,
  showFooterActions,
  onEdit,
  onDelete,
  onTest,
  testing,
  onPdmReindex,
  pdmIndexing,
  pdmStats,
  onSchemaReindex,
  schemaIndexing,
  activeSchemaJob,
  schemaStats,
  onInlineUpdate,
  classified = false,
  skillCount,
  onOpenSkills,
  onOpenActions,
  actionsMenuOpen,
}: ToolCardProps) {
  const typeInfo = TOOL_TYPE_INFO[tool.tool_type];
  const hasSchemaIndexing = hasSchemaIndexingEnabled(tool);
  const workingDirectory = getToolWorkingDirectory(tool);
  const hasActiveIndexingJob = Boolean(
    activeSchemaJob && ['pending', 'processing', 'indexing'].includes(activeSchemaJob.status),
  );
  const toolSupportsWorkingDir =
    tool.tool_type === 'ssh_shell' ||
    tool.tool_type === 'windows_shell' ||
    tool.tool_type === 'odoo_shell';
  const pdmSchedule = formatPdmSchedule(
    (tool.connection_config as { reindex_interval_hours?: number }).reindex_interval_hours,
  );
  const solidworksOptions =
    tool.tool_type === 'solidworks' ? (tool.connection_config as SolidworksConnectionConfig) : null;
  const linkedPdmTool = tools.find(
    (candidate) =>
      candidate.id === solidworksOptions?.pdm_tool_id && candidate.tool_type === 'solidworks_pdm',
  );
  const pdmLink = solidworksOptions
    ? (tool.pdm_link ??
      (solidworksOptions.pdm_tool_id
        ? {
            tool_id: solidworksOptions.pdm_tool_id,
            tool_name: linkedPdmTool?.name ?? null,
            enabled: linkedPdmTool?.enabled ?? false,
            exists: Boolean(linkedPdmTool),
          }
        : null))
    : null;
  const pdmUnavailable = pdmLink && (!pdmLink.exists || !pdmLink.enabled);
  const pdmDatabase = (linkedPdmTool?.connection_config as { database?: string } | undefined)
    ?.database;
  const hostUpdate = tool.tool_type === 'solidworks' ? tool.solidworks_host_update : null;
  const hostUpdateKind = hostUpdateBadgeKind(hostUpdate);
  const pdmLinkDescription = pdmLink
    ? !pdmLink.exists
      ? 'The linked PDM tool no longer exists.'
      : !pdmLink.enabled
        ? `“${pdmLink.tool_name}” is disabled, so the AI cannot search it. Enable it or choose another tool.`
        : `Linked to SolidWorks PDM tool ${pdmLink.tool_name}, vault ${pdmDatabase ?? 'unknown'}`
    : '';
  const approvedFolderCount = solidworksOptions?.pdm_approved_folders?.length ?? 0;
  const linkedFrom =
    tool.tool_type === 'solidworks_pdm'
      ? getPdmLinkRelations(tools).filter((relation) => relation.pdmToolId === tool.id)
      : [];
  const linkedNote = describePdmLinkedNote(linkedFrom);

  // Inline editing state
  const [editingField, setEditingField] = useState<EditingField>(null);
  const [editName, setEditName] = useState(tool.name);
  const [editDescription, setEditDescription] = useState(tool.description || '');
  const [saving, setSaving] = useState(false);
  const [descriptionOpen, setDescriptionOpen] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const descTextareaRef = useRef<HTMLTextAreaElement>(null);
  const moreButtonRef = useRef<HTMLButtonElement>(null);
  const moreContentRef = useRef<HTMLSpanElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const regionRef = useRef<HTMLDivElement>(null);
  const descriptionRef = useRef<HTMLParagraphElement>(null);
  const { isClamped } = useFitLineClamp({
    cardRef,
    regionRef,
    textRef: descriptionRef,
    enabled: editingField !== 'description',
  });
  const summary = getToolConnectionSummary(tool);
  const [summaryRef, isSummaryTruncated] = useIsTruncated<HTMLElement>(summary);
  // Only show the title popover when the name is actually clipped by the
  // responsive tool card (otherwise it would just repeat visible text).
  const [titleRef, isTitleTruncated] = useIsTruncated<HTMLHeadingElement>(tool.name);

  useEffect(() => {
    if (!isClamped || editingField === 'description') setDescriptionOpen(false);
  }, [isClamped, editingField]);

  useEffect(() => {
    if (!descriptionOpen) return;
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setDescriptionOpen(false);
      moreButtonRef.current?.focus();
    };
    const handleOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        !moreButtonRef.current?.contains(target) &&
        !moreContentRef.current?.closest('.popover')?.contains(target)
      ) {
        setDescriptionOpen(false);
      }
    };
    document.addEventListener('keydown', handleEscape);
    document.addEventListener('mousedown', handleOutside);
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('mousedown', handleOutside);
    };
  }, [descriptionOpen]);

  // Format memory size for display
  const formatMemory = (mb: number): string => {
    if (mb < 1) return `${Math.round(mb * 1024)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  // Determine heartbeat display status
  const getHeartbeatDisplay = () => {
    if (testing) {
      return {
        status: 'checking',
        label: 'Testing...',
        node: <MiniLoadingSpinner variant="css" />,
      };
    }
    if (!tool.enabled) {
      return { status: 'disabled', label: 'Disabled', node: <span className="status-dot" /> };
    }
    if (!heartbeat) {
      return { status: 'checking', label: 'Checking...', node: <span className="status-dot" /> };
    }
    if (heartbeat.alive) {
      const latency = heartbeat.latency_ms ? `${Math.round(heartbeat.latency_ms)}ms` : '';
      return {
        status: 'alive',
        label: latency || 'Connected',
        node: <span className="status-dot" />,
      };
    }
    return {
      status: 'dead',
      label: heartbeat.error || 'Disconnected',
      node: <span className="status-dot" />,
    };
  };

  const heartbeatDisplay = getHeartbeatDisplay();

  // Focus input/textarea when entering edit mode
  useEffect(() => {
    if (editingField === 'name' && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    } else if (editingField === 'description' && descTextareaRef.current) {
      descTextareaRef.current.focus();
      descTextareaRef.current.select();
    }
  }, [editingField]);

  // Sync local state when tool prop changes
  useEffect(() => {
    setEditName(tool.name);
    setEditDescription(tool.description || '');
  }, [tool.name, tool.description]);

  const handleStartEdit = (field: EditingField) => {
    setEditingField(field);
    if (field === 'name') {
      setEditName(tool.name);
    } else if (field === 'description') {
      setEditDescription(tool.description || '');
    }
  };

  const handleCancelEdit = () => {
    setEditingField(null);
    setEditName(tool.name);
    setEditDescription(tool.description || '');
  };

  const handleSaveEdit = async () => {
    if (!onInlineUpdate || saving) return;

    const updates: { name?: string; description?: string } = {};

    if (editingField === 'name' && editName.trim() !== tool.name) {
      if (!editName.trim()) {
        // Don't allow empty names
        handleCancelEdit();
        return;
      }
      updates.name = editName.trim();
    } else if (editingField === 'description' && editDescription !== (tool.description || '')) {
      updates.description = editDescription;
    }

    if (Object.keys(updates).length === 0) {
      setEditingField(null);
      return;
    }

    setSaving(true);
    try {
      await onInlineUpdate(tool.id, updates);
      setEditingField(null);
    } catch (err) {
      console.error('Failed to save:', err);
      // Keep edit mode open on error
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter' && (editingField === 'name' || e.ctrlKey)) {
      e.preventDefault();
      handleSaveEdit();
    }
  };

  return (
    <div
      ref={cardRef}
      id={targetId}
      tabIndex={-1}
      data-tool-type={tool.tool_type}
      className={`tool-card ${!tool.enabled ? 'disabled' : ''}`}
    >
      <div className="tool-card-header">
        <div className="tool-card-icon">
          <Icon name={getToolIconType(typeInfo?.icon)} size={22} />
        </div>
        <div className="tool-card-header-content">
          <div className="tool-card-header-main">
            {editingField === 'name' ? (
              <div className="tool-card-title inline-edit-field">
                <input
                  ref={nameInputRef}
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={handleSaveEdit}
                  disabled={saving}
                  className="inline-edit-input"
                />
              </div>
            ) : (
              <div
                className="tool-card-title editable-field-wrapper name-wrapper"
                onClick={() => handleStartEdit('name')}
              >
                <Popover
                  className="tool-title-popover"
                  content={tool.name}
                  position="top"
                  trigger="hover"
                  disabled={!isTitleTruncated}
                >
                  <h3 ref={titleRef} title={isTitleTruncated ? tool.name : undefined}>
                    {tool.name}
                  </h3>
                </Popover>
                <button
                  type="button"
                  className="inline-edit-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleStartEdit('name');
                  }}
                  title="Edit name"
                >
                  <Icon name="pencil" size={14} />
                </button>
              </div>
            )}
            <div className="tool-card-heartbeat">
              <Popover
                content={heartbeatDisplay.label}
                position="top"
                style={{ display: 'inline-flex' }}
              >
                <span className={`heartbeat-indicator ${heartbeatDisplay.status}`}>
                  {heartbeatDisplay.node}
                </span>
              </Popover>
            </div>
            {onOpenActions && (
              <button
                type="button"
                className="tool-card-menu-button"
                data-tool-actions-button
                aria-label={`Tool actions for ${tool.name}`}
                aria-haspopup="menu"
                aria-expanded={actionsMenuOpen ?? false}
                onClick={(event) => onOpenActions(tool, event.currentTarget)}
              >
                <MoreHorizontal size={18} />
              </button>
            )}
          </div>
          <Popover
            className="tool-card-summary-popover"
            content={summary}
            disabled={!isSummaryTruncated}
          >
            <code ref={summaryRef} className="tool-card-connection-sub" data-tool-summary>
              {summary}
            </code>
          </Popover>
        </div>
      </div>
      <BadgeRow>
        {!tool.enabled && (
          <span className="tool-badge" data-tool-badge="disabled">
            Disabled
          </span>
        )}
        {tool.undecryptable_fields.length > 0 && (
          <Popover
            content={`Credential ${tool.undecryptable_fields.length === 1 ? 'field' : 'fields'} ${tool.undecryptable_fields.join(', ')} cannot be decrypted with the current server key.`}
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span
              className="tool-badge badge-warning"
              title="Some credentials cannot be decrypted with the current server key."
            >
              <Icon name="alert-circle" size={12} />
              <span className="path-text">Key Mismatch</span>
            </span>
          </Popover>
        )}
        {hostUpdate && hostUpdateKind === 'failed' && (
          <HostUpdateBadge update={hostUpdate} kind={hostUpdateKind} />
        )}
        {pdmLink && pdmUnavailable && (
          <Popover content={pdmLinkDescription} position="top" style={{ display: 'inline-flex' }}>
            <span className="tool-badge badge-warning" data-solidworks-pdm-badge>
              <Icon name="database" size={12} />
              <span className="path-text">PDM unavailable</span>
            </span>
          </Popover>
        )}
        {tool.allow_write && (tool.tool_type !== 'solidworks' || pdmLink) && (
          <Popover
            content={
              solidworksOptions
                ? approvedFolderCount
                  ? `Can add files, check out and check in in ${approvedFolderCount} approved vault folders`
                  : 'No approved folders: changes are refused'
                : 'Write access enabled'
            }
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span
              className="tool-badge badge-warning"
              data-solidworks-write-badge={solidworksOptions ? true : undefined}
            >
              <Icon name="alert-triangle" size={12} />
              <span className="path-text">Write</span>
            </span>
          </Popover>
        )}
        {solidworksOptions?.allow_scripts === true && (
          <Popover
            content="Can run C# scripts on the SolidWorks computer as its Windows user. Not sandboxed."
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span className="tool-badge badge-warning" data-badge="scripts-allowed">
              <Icon name="alert-triangle" size={12} />
              <span className="path-text">Scripts on</span>
            </span>
          </Popover>
        )}
        {hasActiveIndexingJob && (
          <IndexingPill activeJob={activeSchemaJob} progressLabelPrefix="Indexing" />
        )}
        {hostUpdate && hostUpdateKind && hostUpdateKind !== 'failed' && (
          <HostUpdateBadge update={hostUpdate} kind={hostUpdateKind} />
        )}
        {tool.tunnel_client?.outdated === true && (
          <DetailBadge
            hook="tunnel-client-outdated"
            className="tool-badge"
            icon={<Icon name="plug" size={12} />}
            label="Tunnel client outdated"
            detail={tunnelClientOutdatedText(tool.tunnel_client)}
          />
        )}
        {classified && (
          <span className="tool-badge" data-tool-classified-badge>
            Classified
          </span>
        )}
        {skillCount && skillCount.total > 0 && (
          <button
            type="button"
            className="tool-badge tool-skills-badge"
            data-tool-skills-badge
            title={`${skillCount.enabled} of ${skillCount.total} skills enabled`}
            aria-label={`${formatInstructionSkillCount(skillCount)}, ${skillCount.enabled} enabled. Open skills.`}
            onClick={(event) => {
              event.stopPropagation();
              onOpenSkills(tool);
            }}
          >
            <Icon name="book" size={12} />
            <span className="path-text">{formatInstructionSkillCount(skillCount)}</span>
          </button>
        )}
        {solidworksOptions?.instance_mode === 'attach' && (
          <Popover
            content="Attaches to the SolidWorks the user has open"
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span className="tool-badge" data-solidworks-mode-badge>
              Attach
            </span>
          </Popover>
        )}
        {solidworksOptions?.solidworks_version != null && (
          <Popover
            content={`Pinned to SolidWorks ${solidworksOptions.solidworks_version}`}
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span className="tool-badge" data-solidworks-version-badge>
              SolidWorks {solidworksOptions.solidworks_version}
            </span>
          </Popover>
        )}
        {pdmLink && !pdmUnavailable && (
          <Popover content={pdmLinkDescription} position="top" style={{ display: 'inline-flex' }}>
            <span className="tool-badge" data-solidworks-pdm-badge>
              <Icon name="database" size={12} />
              <span className="path-text">PDM: {pdmLink.tool_name}</span>
            </span>
          </Popover>
        )}
        {toolSupportsWorkingDir && (
          <Popover
            content={
              workingDirectory ? `Constrained to ${workingDirectory}` : 'Working directory not set'
            }
            position="top"
            style={{ display: 'inline-flex' }}
          >
            <span className="tool-badge">
              <Icon name="folder" size={12} />
              <span className="path-text">{workingDirectory || '/'}</span>
            </span>
          </Popover>
        )}
      </BadgeRow>

      {heartbeat && !heartbeat.alive && tool.enabled && !testing && (
        <div className="tool-card-heartbeat-error">
          <span className="error-icon">
            <Icon name="alert-circle" size={16} />
          </span>
          <span title={heartbeat.error || 'Connection failed'}>
            {addHeartbeatErrorSoftWraps(heartbeat.error || 'Connection failed')}
          </span>
        </div>
      )}

      <div ref={regionRef} className="tool-card-description-region">
        {editingField === 'description' ? (
          <div className="inline-edit-field description-edit">
            <textarea
              ref={descTextareaRef}
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleSaveEdit}
              disabled={saving}
              className="inline-edit-textarea"
              rows={1}
              placeholder="Description for AI..."
            />
          </div>
        ) : (
          <div
            className="editable-field-wrapper description-wrapper"
            onClick={(event) => {
              if (!event.currentTarget.contains(event.target as Node)) return;
              handleStartEdit('description');
            }}
          >
            {tool.description ? (
              <p ref={descriptionRef} className="tool-card-description">
                {tool.description}
              </p>
            ) : (
              <p ref={descriptionRef} className="tool-card-description placeholder">
                Add description for AI...
              </p>
            )}
            <button
              type="button"
              className="inline-edit-btn"
              onClick={(e) => {
                e.stopPropagation();
                handleStartEdit('description');
              }}
              title="Edit description"
            >
              <Icon name="pencil" size={14} />
            </button>
            {isClamped && tool.description && (
              <Popover
                className="tool-card-more-popover"
                content={<span ref={moreContentRef}>{tool.description}</span>}
                trigger="click"
                position="bottom"
                show={descriptionOpen}
              >
                <button
                  ref={moreButtonRef}
                  type="button"
                  className="tool-card-more"
                  aria-label="Show full description"
                  aria-expanded={descriptionOpen}
                  onClick={(event) => {
                    event.stopPropagation();
                    setDescriptionOpen((open) => !open);
                  }}
                >
                  More
                </button>
              </Popover>
            )}
          </div>
        )}
      </div>

      {((hasSchemaIndexing && schemaStats && schemaStats.embedding_count > 0) ||
        tool.tool_type === 'solidworks_pdm' ||
        linkedFrom.length > 0) && (
        <dl className="tool-card-facts">
          {hasSchemaIndexing && schemaStats && schemaStats.embedding_count > 0 && (
            <div className="tool-card-schema-stats">
              <dt className="schema-stats-label">Schema Index:</dt>
              <dd className="schema-stats-value">
                {formatToolCount(schemaStats.embedding_count)} tables
                {schemaStats.estimated_memory_mb != null && (
                  <span className="schema-stats-memory" title="Estimated pgvector storage size">
                    ({formatMemory(schemaStats.estimated_memory_mb)})
                  </span>
                )}
              </dd>
            </div>
          )}

          {tool.tool_type === 'solidworks_pdm' && pdmStats && pdmStats.document_count > 0 && (
            <div className="tool-card-schema-stats">
              <dt className="schema-stats-label">PDM Index:</dt>
              <dd className="schema-stats-value">
                {formatToolCount(pdmStats.document_count)} documents
              </dd>
            </div>
          )}

          {tool.tool_type === 'solidworks_pdm' && (
            <div className="tool-card-schema-stats" data-testid={`pdm-tool-summary-${tool.id}`}>
              <dt className="schema-stats-label">PDM schedule:</dt>
              <dd className="schema-stats-value">{pdmSchedule}</dd>
              {pdmStats?.last_indexed_at && (
                <dd className="schema-stats-memory" title={pdmStats.last_indexed_at}>
                  Last success: {formatToolFactDate(pdmStats.last_indexed_at)}
                </dd>
              )}
            </div>
          )}

          {linkedFrom.length > 0 && (
            <div className="tool-card-schema-stats" data-testid={`pdm-linked-from-${tool.id}`}>
              <dt className="schema-stats-label">Used by:</dt>
              <dd className="schema-stats-value">
                {linkedFrom.map((relation) => relation.solidworksToolName).join(', ')}
              </dd>
              {linkedNote && (
                <dd
                  className="schema-stats-memory"
                  data-pdm-linked-note
                  id={`tool-${tool.id}-pdm-linked-note`}
                >
                  {linkedNote}
                </dd>
              )}
            </div>
          )}
        </dl>
      )}

      {showFooterActions && (
        <div className="tool-card-footer">
          <div className="tool-card-actions">
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onTest(tool.id)}
              disabled={testing}
            >
              {testing ? 'Testing...' : 'Test'}
            </button>
            {tool.tool_type === 'solidworks_pdm' && onPdmReindex && (
              <>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onPdmReindex(tool.id, false)}
                  disabled={pdmIndexing}
                  title="Index new and changed documents"
                >
                  {pdmIndexing ? 'Indexing...' : 'Index'}
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onPdmReindex(tool.id, true)}
                  disabled={pdmIndexing}
                  title="Re-index all documents from scratch"
                >
                  Full Re-index
                </button>
              </>
            )}
            {hasSchemaIndexing && onSchemaReindex && (
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => onSchemaReindex(tool.id, true)}
                disabled={schemaIndexing}
                title="Re-index database schema"
              >
                {schemaIndexing ? 'Indexing...' : 'Re-index Schema'}
              </button>
            )}
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              data-tool-skills-button
              onClick={() => onOpenSkills(tool)}
            >
              <Icon name="book" size={14} />
              Skills
            </button>
            <button type="button" className="btn btn-sm" onClick={() => onEdit(tool)}>
              Edit
            </button>
            {isPairedWindowsTool(tool) || linkedFrom.length > 0 ? (
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={() => onDelete(tool)}
                title="Delete tool"
              >
                Delete
              </button>
            ) : (
              <DeleteConfirmButton
                onDelete={() => onDelete(tool)}
                className="btn btn-sm btn-danger"
                title="Delete tool"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
