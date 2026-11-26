import { useState, useEffect, useCallback, useRef, useMemo, type RefObject } from 'react';
import { api } from '@/api';
import type {
  ToolConfig,
  ToolGroup,
  User,
  HeartbeatStatus,
  SchemaIndexStats,
  SchemaIndexJob,
  PdmIndexStats,
  UserspaceMountSource,
  MountSourceAffectedWorkspacesResponse,
  UserCloudOAuthAccount,
  InstructionSkill,
  InstructionSkillOwner,
  InstructionSkillSummary,
} from '@/types';
import { TOOL_TYPE_INFO } from '@/types';
import { ToolWizard } from './ToolWizard';
import { InstructionSkillsDialog } from './InstructionSkillsDialog';
import { MountSourceWizard } from './MountSourceWizard';
import { Icon } from './Icon';
import { ToolCard } from './ToolCard';
import { ToolGroupSection } from './ToolGroupSection';
import { ToolGroupDialog } from './ToolGroupDialog';
import { ToolGroupDeleteDialog } from './ToolGroupDeleteDialog';
import { MoveToGroupDialog, type MoveToGroupTarget } from './MoveToGroupDialog';
import { AnimatedCreateButton } from './AnimatedCreateButton';
import { useToast, ToastContainer } from './shared/Toast';
import {
  ContextMenu,
  type ContextMenuAnchorRect,
  type ContextMenuPlacement,
} from './shared/ContextMenu';
import {
  SearchFilterBar,
  searchFilterTextMatchesQuery,
  useUrlSearchFilterState,
} from './shared/SearchFilterBar';
import { HardDrive, Trash2, Pencil, X, Upload } from 'lucide-react';
import { resolveSourceDisplayPath } from '@/utils/mountPaths';
import { toToolAccessGroupOptions, toToolAccessUserOptions } from '@/utils/toolAccessOptions';
import {
  EMPTY_INSTRUCTION_SKILL_SUMMARY,
  countInstructionSkills,
  describeInstructionSkillCount,
  restoreFocusIfLost,
} from '@/utils/instructionSkills';
import {
  getExportPasswordPolicy,
  passwordMeetsRequirements,
  type ExportPasswordPolicy,
} from '@/utils/exportPasswordPolicy';
import { PasswordRequirementsChecklist } from './shared/PasswordRequirementsChecklist';
import { ToolAccessModal } from './ToolAccessModal';
import { ToolDeleteDialog } from './ToolDeleteDialog';
import type { ToolAccessPolicy } from '@/types';
import { contentProtectionApi, type ContentProtectionConfig } from '@/api/contentProtection';
import { getPdmLinkRelations } from '@/utils/solidworksPdmLink';
import {
  getToolConnectionSummary,
  hasSchemaIndexingEnabled,
  isPairedWindowsTool,
} from '@/utils/toolConfigDisplay';
import {
  UNGROUPED_SECTION_ID,
  buildToolSections,
  computeToolMove,
  moveToolWithinSection,
  type ToolSection,
} from '@/utils/toolGroupSections';
import { summarizeToolGroupHealth } from '@/utils/toolGroupHealth';
import { getCollapsedToolSections, setCollapsedToolSections } from '@/utils/toolSectionPreferences';

const DRAG_REORDER_PREVIEW_DELAY_MS = 80;
type DragPreviewTarget = { toolId: string; insertBefore: boolean };

interface ToolHealthEventPayload {
  statuses?: Record<string, HeartbeatStatus>;
}

function formatMountSourceInterval(seconds: number): string {
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
  return h > 0 ? `${d}d ${h}h` : `${d}d`;
}

function isAutoSyncMountSource(source: UserspaceMountSource): boolean {
  return (
    source.source_type === 'ssh' ||
    source.source_type === 'microsoft_drive' ||
    source.source_type === 'google_drive'
  );
}

function isCloudMountSource(
  source: UserspaceMountSource,
): source is UserspaceMountSource & { source_type: 'microsoft_drive' | 'google_drive' } {
  return source.source_type === 'microsoft_drive' || source.source_type === 'google_drive';
}

function getCloudOAuthCallbackUrl(): string {
  return new URL('/indexes/userspace/cloud-oauth/callback', window.location.origin).toString();
}

function pickLatestConnectedCloudAccount(
  accounts: UserCloudOAuthAccount[],
  provider: 'microsoft_drive' | 'google_drive',
): UserCloudOAuthAccount | null {
  const sorted = [...accounts]
    .filter((account) => account.provider === provider && account.connected)
    .sort(
      (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
    );
  return sorted[0] ?? null;
}

const SORT_ORDER_STEP = 100;
const UNGROUPED_LABEL = 'Ungrouped';
const UNGROUPED_DROP_TITLE = 'Ungrouped — drop to remove from group';
const NEW_GROUP_BUTTON_ID = 'tools-new-group-button';
const GROUP_EMPTY_MESSAGE =
  "No tools in this group yet. Drag a tool here, or choose Move to group… from a tool's actions.";

type GroupDialogState = { mode: 'create' } | { mode: 'edit'; group: ToolGroup };
type SectionDropHandlers = Pick<
  React.HTMLAttributes<HTMLElement>,
  'onDragOver' | 'onDragLeave' | 'onDrop'
>;
interface VisibleToolSection {
  section: ToolSection;
  shownTools: ToolConfig[];
}

interface PendingFocus {
  elementId: string;
  /** The group whose element is requested; when it no longer exists the request is dropped. */
  groupId: string | null;
}

/** Copies the rect's edges; DOMRect itself is not a plain placement value. */
function toAnchorRect(rect: DOMRect): ContextMenuAnchorRect {
  return { top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right };
}

function sectionToggleId(sectionId: string): string {
  return `tools-section-${sectionId}-toggle`;
}

function toolMatchesFilter(tool: ToolConfig, groupName: string, queries: string[]): boolean {
  return searchFilterTextMatchesQuery(`${getToolSearchText(tool)} ${groupName}`, queries);
}

function withSetMembers(
  current: ReadonlySet<string>,
  ids: Iterable<string>,
  present: boolean,
): Set<string> {
  const next = new Set(current);
  for (const id of ids) {
    if (present) next.add(id);
    else next.delete(id);
  }
  return next;
}

function pluralize(count: number, singular: string): string {
  return `${count} ${count === 1 ? singular : `${singular}s`}`;
}

function getToolsSummaryText(input: {
  toolCount: number;
  matchedCount: number;
  groupCount: number;
  filtering: boolean;
}): string {
  if (input.filtering) return `${input.matchedCount} of ${pluralize(input.toolCount, 'tool')}`;
  const toolsText = pluralize(input.toolCount, 'tool');
  return input.groupCount > 0
    ? `${toolsText} in ${pluralize(input.groupCount, 'group')}`
    : toolsText;
}

// Drop position within a card: before when the pointer is in its left half.
function isPointerBeforeMidpoint(event: React.DragEvent<HTMLElement>): boolean {
  const rect = event.currentTarget.getBoundingClientRect();
  return rect.width > 0 ? event.clientX - rect.left < rect.width / 2 : true;
}

// Brief highlight on a collapsed section header that received a drop.
function flashSectionHeader(sectionId: string): void {
  const header = document
    .getElementById(`tools-section-${sectionId}`)
    ?.querySelector<HTMLElement>('.tools-section-header');
  if (!header) return;
  header.classList.add('tools-section-flash');
  window.setTimeout(() => header.classList.remove('tools-section-flash'), 1000);
}

// Closes an open menu on outside pointer press or Escape; the trigger itself is not "outside".
function useMenuDismissal({
  open,
  menuRef,
  triggerRef,
  onDismiss,
}: {
  open: boolean;
  menuRef: RefObject<HTMLElement | null>;
  triggerRef: RefObject<HTMLElement | null>;
  onDismiss: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (menuRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      onDismiss();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onDismiss();
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open, menuRef, triggerRef, onDismiss]);
}
function getDuplicateToolName(originalName: string, tools: ToolConfig[]): string {
  const existingNames = new Set(tools.map((tool) => tool.name.trim()));
  const candidate = `${originalName} Copy`;
  if (!existingNames.has(candidate)) {
    return candidate;
  }

  for (let suffix = 2; ; suffix += 1) {
    const numberedCandidate = `${originalName} Copy ${suffix}`;
    if (!existingNames.has(numberedCandidate)) {
      return numberedCandidate;
    }
  }
}

function getToolSearchText(tool: ToolConfig): string {
  const typeInfo = TOOL_TYPE_INFO[tool.tool_type];
  return [
    tool.name,
    tool.description,
    typeInfo?.name,
    tool.tool_type,
    getToolConnectionSummary(tool),
  ]
    .filter(Boolean)
    .join(' ');
}

interface ToolsPanelProps {
  onSchemaJobTriggered?: () => void;
  schemaJobs?: SchemaIndexJob[];
  highlightSection?: string | null;
  onHighlightComplete?: () => void;
  currentUser?: User | null;
}

export function ToolsPanel({
  onSchemaJobTriggered,
  schemaJobs = [],
  highlightSection,
  onHighlightComplete,
  currentUser,
}: ToolsPanelProps) {
  const resolveHighlightTargetId = useCallback((section: string) => {
    if (section.startsWith('tool:')) {
      return `tools-tool-${section.slice('tool:'.length)}`;
    }
    return `tools-${section}`;
  }, []);

  const [tools, setTools] = useState<ToolConfig[]>([]);
  const pdmLinkRelations = useMemo(() => getPdmLinkRelations(tools), [tools]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [toasts, toast] = useToast();
  const [showWizard, setShowWizard] = useState(false);
  const [editingTool, setEditingTool] = useState<ToolConfig | null>(null);
  const [testingToolId, setTestingToolId] = useState<string | null>(null);
  const [pdmIndexingToolId, setPdmIndexingToolId] = useState<string | null>(null);
  const [schemaIndexingToolId, setSchemaIndexingToolId] = useState<string | null>(null);
  const [heartbeats, setHeartbeats] = useState<Record<string, HeartbeatStatus>>({});
  const [schemaStats, setSchemaStats] = useState<Record<string, SchemaIndexStats>>({});
  const [pdmStats, setPdmStats] = useState<Record<string, PdmIndexStats>>({});
  const [showFooterActions, setShowFooterActions] = useState(false);
  const [contentProtectionConfig, setContentProtectionConfig] =
    useState<ContentProtectionConfig | null>(null);
  const cpEnabled = contentProtectionConfig?.enabled === true;
  const requiredToolIds = useMemo(
    () =>
      new Set(
        contentProtectionConfig?.requirements
          .filter(
            (requirement) => requirement.scope_kind === 'tool' && requirement.mode === 'require',
          )
          .map((requirement) => requirement.scope_key),
      ),
    [contentProtectionConfig],
  );

  // Mount source state
  const [mountSources, setMountSources] = useState<UserspaceMountSource[]>([]);
  const [mountSourceDeletingId, setMountSourceDeletingId] = useState<string | null>(null);
  const [showMountSourceWizard, setShowMountSourceWizard] = useState(false);
  const [editingMountSource, setEditingMountSource] = useState<UserspaceMountSource | null>(null);
  const [reconnectingMountSource, setReconnectingMountSource] = useState<{
    mountSourceId: string;
    provider: 'microsoft_drive' | 'google_drive';
  } | null>(null);
  const [reconnectStartingMountSourceId, setReconnectStartingMountSourceId] = useState<
    string | null
  >(null);
  const [disableConfirmation, setDisableConfirmation] = useState<{
    source: UserspaceMountSource;
    affected: MountSourceAffectedWorkspacesResponse | null;
    loading: boolean;
  } | null>(null);
  const [writeConfirmTool, setWriteConfirmTool] = useState<ToolConfig | null>(null);
  const [deleteConfirmTool, setDeleteConfirmTool] = useState<ToolConfig | null>(null);
  const [windowsDeleteTool, setWindowsDeleteTool] = useState<ToolConfig | null>(null);
  const [clearConfirmTool, setClearConfirmTool] = useState<ToolConfig | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [toolAccessModalTool, setToolAccessModalTool] = useState<ToolConfig | null>(null);
  const [toolAccessPolicy, setToolAccessPolicy] = useState<ToolAccessPolicy | null>(null);
  const [toolAccessUserOptions, setToolAccessUserOptions] = useState<
    Array<{ id: string; username: string; display_name: string | null; is_admin: boolean }>
  >([]);
  const [toolAccessGroupOptions, setToolAccessGroupOptions] = useState<
    Array<{ id: string; key: string; display_name: string; provider: string; member_count: number }>
  >([]);
  const [toolAccessLoading, setToolAccessLoading] = useState(false);
  const [toolAccessSaving, setToolAccessSaving] = useState(false);
  const [instructionSkillSummary, setInstructionSkillSummary] = useState<InstructionSkillSummary>(
    EMPTY_INSTRUCTION_SKILL_SUMMARY,
  );
  const [skillsDialogOpen, setSkillsDialogOpen] = useState(false);
  const [skillsDialogTarget, setSkillsDialogTarget] = useState<{
    owner: InstructionSkillOwner;
    ownerName: string;
  } | null>(null);

  const [toolPasswordModal, setToolPasswordModal] = useState<
    { mode: 'export'; toolId: string } | { mode: 'import' } | null
  >(null);
  const [toolPassword, setToolPassword] = useState('');
  const [toolPasswordConfirm, setToolPasswordConfirm] = useState('');
  const [importFileContent, setImportFileContent] = useState<string | null>(null);
  const [passwordModalBusy, setPasswordModalBusy] = useState(false);
  const [exportPasswordPolicy, setExportPasswordPolicy] = useState<ExportPasswordPolicy>(
    getExportPasswordPolicy(null),
  );
  const importFileInputRef = useRef<HTMLInputElement>(null);

  // Tool sections: in-memory filter/reveal overrides, persisted collapse state, dialogs and menus
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(() =>
    getCollapsedToolSections(currentUser?.id),
  );
  const [groupsLoaded, setGroupsLoaded] = useState(false);
  const [filterCollapsedIds, setFilterCollapsedIds] = useState<Set<string>>(() => new Set());
  const [forcedOpenIds, setForcedOpenIds] = useState<Set<string>>(() => new Set());
  const [groupDialog, setGroupDialog] = useState<GroupDialogState | null>(null);
  const [deleteGroupTarget, setDeleteGroupTarget] = useState<ToolGroup | null>(null);
  const [moveDialogToolId, setMoveDialogToolId] = useState<string | null>(null);
  const [toolMenu, setToolMenu] = useState<{
    toolId: string;
    placement: ContextMenuPlacement;
  } | null>(null);
  const [groupMenu, setGroupMenu] = useState<{
    sectionId: string;
    placement: ContextMenuPlacement;
  } | null>(null);
  const [highlightRequest, setHighlightRequest] = useState<{ elementId: string } | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const toolMenuRef = useRef<HTMLDivElement>(null);
  const toolMenuTriggerRef = useRef<HTMLElement | null>(null);
  const groupMenuRef = useRef<HTMLDivElement>(null);
  const groupMenuTriggerRef = useRef<HTMLElement | null>(null);
  const pendingFocusRef = useRef<PendingFocus | null>(null);
  // Serialises ordering writes: a second move starts only after the first settles.
  const orderLockRef = useRef(false);
  const [orderPending, setOrderPending] = useState(false);
  const handledHighlightRef = useRef<string | null>(null);
  const toolFilter = useUrlSearchFilterState();
  const toolFilterInputRef = useRef<HTMLInputElement | null>(null);

  // Drag-and-drop state
  const [dragToolId, setDragToolId] = useState<string | null>(null);
  const [dragOverToolId, setDragOverToolId] = useState<string | null>(null);
  const [dragInsertBefore, setDragInsertBefore] = useState(true);
  const [bodyDropSectionId, setBodyDropSectionId] = useState<string | null>(null);
  const [headerDropSectionId, setHeaderDropSectionId] = useState<string | null>(null);
  const dragPreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingDragPreviewRef = useRef<DragPreviewTarget | null>(null);

  const activeSchemaJobsByToolId = useMemo(() => {
    const jobsByToolId: Record<string, SchemaIndexJob> = {};

    for (const job of schemaJobs) {
      if (
        (job.status === 'pending' || job.status === 'indexing') &&
        !jobsByToolId[job.tool_config_id]
      ) {
        jobsByToolId[job.tool_config_id] = job;
      }
    }

    return jobsByToolId;
  }, [schemaJobs]);

  const isToolFiltering = toolFilter.queries.length > 0;
  const dragDisabled = isToolFiltering || orderPending;
  const knownGroupIds = useMemo(() => new Set(groups.map((group) => group.id)), [groups]);
  const knownSectionIds = useMemo(
    () => new Set([UNGROUPED_SECTION_ID, ...knownGroupIds]),
    [knownGroupIds],
  );
  // Every section in display order; the Ungrouped section is included while dragging.
  const sections = useMemo(
    () => buildToolSections(tools, groups, { includeEmptyUngrouped: dragToolId !== null }),
    [tools, groups, dragToolId],
  );
  const orderedGroups = useMemo(
    () => sections.flatMap((section) => (section.group ? [section.group] : [])),
    [sections],
  );
  const visibleSections = useMemo(
    () =>
      sections.flatMap((section): VisibleToolSection[] => {
        const groupName = section.group?.name ?? '';
        const shownTools = isToolFiltering
          ? section.tools.filter((tool) => toolMatchesFilter(tool, groupName, toolFilter.queries))
          : section.tools;
        return isToolFiltering && shownTools.length === 0 ? [] : [{ section, shownTools }];
      }),
    [sections, isToolFiltering, toolFilter.queries],
  );
  const matchedTools = visibleSections.flatMap(({ shownTools }) => shownTools);
  const plainSection: ToolSection = sections[0] ?? {
    id: UNGROUPED_SECTION_ID,
    group: null,
    tools: [],
  };
  const summaryText = getToolsSummaryText({
    toolCount: tools.length,
    matchedCount: matchedTools.length,
    groupCount: groups.length,
    filtering: isToolFiltering,
  });
  const moveDialogTool = tools.find((tool) => tool.id === moveDialogToolId) ?? null;
  const moveTargetGroups = sections.flatMap((section) =>
    section.group
      ? [
          {
            id: section.group.id,
            name: section.group.name,
            description: section.group.description,
            toolCount: section.tools.length,
          },
        ]
      : [],
  );

  const currentUserId = currentUser?.id ?? null;
  useEffect(() => {
    // Per-user state: neither the saved collapse set nor in-memory overrides carry over.
    setCollapsedIds(getCollapsedToolSections(currentUserId));
    setForcedOpenIds(new Set());
    setFilterCollapsedIds(new Set());
  }, [currentUserId]);

  useEffect(() => {
    if (!isToolFiltering) setFilterCollapsedIds((current) => (current.size ? new Set() : current));
  }, [isToolFiltering]);

  const isSectionCollapsed = (sectionId: string): boolean => {
    if (forcedOpenIds.has(sectionId)) return false;
    return isToolFiltering ? filterCollapsedIds.has(sectionId) : collapsedIds.has(sectionId);
  };
  const allSectionsCollapsed =
    visibleSections.length > 0 &&
    visibleSections.every(({ section }) => isSectionCollapsed(section.id));

  // Groups must be loaded first: the setter prunes ids it does not know.
  const persistCollapsedSections = (next: ReadonlySet<string>) => {
    if (groupsLoaded) setCollapsedToolSections(currentUserId, next, knownSectionIds);
  };

  const toggleSection = (sectionId: string) => {
    const collapseNow = !isSectionCollapsed(sectionId);
    setForcedOpenIds((current) => withSetMembers(current, [sectionId], false));
    if (isToolFiltering) {
      setFilterCollapsedIds((current) => withSetMembers(current, [sectionId], collapseNow));
      return;
    }
    const next = withSetMembers(collapsedIds, [sectionId], collapseNow);
    setCollapsedIds(next);
    persistCollapsedSections(next);
  };

  const toggleAllSections = () => {
    const collapse = !allSectionsCollapsed;
    const sectionIds = visibleSections.map(({ section }) => section.id);
    setForcedOpenIds(new Set());
    if (isToolFiltering) {
      setFilterCollapsedIds((current) => withSetMembers(current, sectionIds, collapse));
      return;
    }
    const next = withSetMembers(collapsedIds, sectionIds, collapse);
    setCollapsedIds(next);
    persistCollapsedSections(next);
  };

  // Opens the section holding a tool (in memory only) and clears a filter that would hide it.
  const revealToolInSection = useCallback(
    (sectionId: string, tool: ToolConfig, groupName: string) => {
      if (isToolFiltering && !toolMatchesFilter(tool, groupName, toolFilter.queries)) {
        toolFilter.clear();
      }
      setForcedOpenIds((current) => withSetMembers(current, [sectionId], true));
    },
    [isToolFiltering, toolFilter],
  );

  const loadTools = useCallback(async () => {
    try {
      setLoading(true);
      const [data, groupData, sources, settingsResponse, protectionConfig, skillSummary] =
        await Promise.all([
          api.listToolConfigs(),
          api.listToolGroups(),
          api.listUserspaceMountSources(),
          api.getSettings().catch(() => null),
          contentProtectionApi.getConfig().catch(() => null),
          api.getInstructionSkillSummary().catch(() => null),
        ]);
      // Filter out filesystem_indexer tools - they're shown in the Indexer tab
      const connectionTools = data.filter((t) => t.tool_type !== 'filesystem_indexer');
      setTools(connectionTools);
      setGroups(groupData);
      setGroupsLoaded(true);
      setMountSources(sources);
      setShowFooterActions(Boolean(settingsResponse?.settings?.show_tool_card_footer_actions));
      setExportPasswordPolicy(getExportPasswordPolicy(settingsResponse?.settings ?? null));
      setContentProtectionConfig(protectionConfig);
      if (skillSummary) setInstructionSkillSummary(skillSummary);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load tools');
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // Load schema stats for tools with schema indexing enabled
  const loadSchemaStats = useCallback(async (toolList: ToolConfig[]) => {
    const schemaTools = toolList.filter(hasSchemaIndexingEnabled);

    if (schemaTools.length === 0) return;

    const statsMap: Record<string, SchemaIndexStats> = {};
    await Promise.all(
      schemaTools.map(async (tool) => {
        try {
          const stats = await api.getSchemaIndexStats(tool.id);
          statsMap[tool.id] = stats;
        } catch (err) {
          console.warn(`Failed to load schema stats for ${tool.name}:`, err);
        }
      }),
    );
    setSchemaStats(statsMap);
  }, []);

  // Load PDM stats for SolidWorks PDM tools
  const loadPdmStats = useCallback(async (toolList: ToolConfig[]) => {
    const pdmTools = toolList.filter((tool) => tool.tool_type === 'solidworks_pdm');

    if (pdmTools.length === 0) return;

    const statsMap: Record<string, PdmIndexStats> = {};
    await Promise.all(
      pdmTools.map(async (tool) => {
        try {
          const stats = await api.getPdmIndexStats(tool.id);
          statsMap[tool.id] = stats;
        } catch (err) {
          console.warn(`Failed to load PDM stats for ${tool.name}:`, err);
        }
      }),
    );
    setPdmStats(statsMap);
  }, []);

  // Fetch cached heartbeat status for all enabled tools
  const fetchHeartbeats = useCallback(async () => {
    try {
      const response = await api.getToolHeartbeats();
      setHeartbeats(response.statuses);
    } catch (err) {
      // Silently fail on heartbeat errors - don't disrupt the UI
      console.warn('Heartbeat check failed:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadTools();
  }, [loadTools]);

  // Navigation target (tool:<id> or a section id): open its section, then highlight it once rendered
  useEffect(() => {
    if (loading) return;
    const requested =
      highlightSection || (window.location.hash === '#tools-connections' ? 'connections' : null);
    if (!requested) {
      handledHighlightRef.current = null;
      return;
    }
    if (requested === handledHighlightRef.current) return;
    handledHighlightRef.current = requested;
    if (requested.startsWith('tool:')) {
      const toolId = requested.slice('tool:'.length);
      const section = sections.find((candidate) =>
        candidate.tools.some((tool) => tool.id === toolId),
      );
      const tool = section?.tools.find((candidate) => candidate.id === toolId);
      if (section && tool) revealToolInSection(section.id, tool, section.group?.name ?? '');
    }
    setHighlightRequest({ elementId: resolveHighlightTargetId(requested) });
  }, [highlightSection, loading, sections, resolveHighlightTargetId, revealToolInSection]);

  useEffect(() => {
    if (!highlightRequest) return;
    const element = document.getElementById(highlightRequest.elementId);
    if (!element) return;
    element.classList.add('highlight-setting');
    element.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
    const timer = setTimeout(() => {
      element.classList.remove('highlight-setting');
      setHighlightRequest(null);
      onHighlightComplete?.();
    }, 2000);
    return () => clearTimeout(timer);
  }, [highlightRequest, onHighlightComplete]);

  // Load schema stats when tools change
  useEffect(() => {
    if (tools.length > 0) {
      loadSchemaStats(tools);
      loadPdmStats(tools);
    }
  }, [tools, loadSchemaStats, loadPdmStats]);

  // Cached heartbeat snapshot + passive updates. Fresh checks happen in the
  // backend monitor; this panel should not add remote SSH/Docker probe load.
  useEffect(() => {
    if (loading || tools.length === 0) return;

    void fetchHeartbeats();
    const source = api.subscribeToolHealthEvents();

    const handleMessage = (event: MessageEvent) => {
      if (!event.data) return;
      let payload: ToolHealthEventPayload;
      try {
        payload = JSON.parse(event.data) as ToolHealthEventPayload;
      } catch {
        return;
      }

      const statuses = payload.statuses ?? {};
      if (Object.keys(statuses).length === 0) return;
      setHeartbeats((current) => ({
        ...current,
        ...statuses,
      }));
    };

    source.addEventListener('snapshot', handleMessage as EventListener);
    source.addEventListener('delta', handleMessage as EventListener);
    source.onmessage = handleMessage;

    return () => {
      source.close();
    };
  }, [loading, tools.length, fetchHeartbeats]);

  const handleAddTool = () => {
    setEditingTool(null);
    setShowWizard(true);
  };

  const handleImportClick = () => {
    importFileInputRef.current?.click();
  };

  const handleImportFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      setImportFileContent(text);
      setToolPasswordModal({ mode: 'import' });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to read file');
    } finally {
      // Reset the input so the same file can be selected again if needed.
      e.target.value = '';
    }
  };

  const closeToolPasswordModal = (force = false) => {
    if (passwordModalBusy && !force) return;
    setToolPasswordModal(null);
    setToolPassword('');
    setToolPasswordConfirm('');
    setImportFileContent(null);
  };

  const handleToolPasswordModalSubmit = async () => {
    if (!toolPasswordModal) return;

    if (toolPasswordModal.mode === 'export') {
      if (!toolPassword) {
        toast.error('Password is required');
        return;
      }
      if (!passwordMeetsRequirements(toolPassword, exportPasswordPolicy)) {
        toast.error('Password does not meet the export password requirements');
        return;
      }
      if (toolPassword !== toolPasswordConfirm) {
        toast.error('Passwords do not match');
        return;
      }

      const tool = tools.find((t) => t.id === toolPasswordModal.toolId);
      if (!tool) return;

      setPasswordModalBusy(true);
      try {
        await api.exportToolConfig(tool.id, toolPassword);
        closeToolPasswordModal(true);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Export failed');
      } finally {
        setPasswordModalBusy(false);
      }
    } else {
      if (!toolPassword) {
        toast.error('Password is required');
        return;
      }
      if (!importFileContent) {
        toast.error('No import file selected');
        return;
      }

      setPasswordModalBusy(true);
      try {
        const importedTool = await api.importToolConfig({
          password: toolPassword,
          file_content: importFileContent,
        });
        toast.success(`Imported ${importedTool.name}`);
        await loadTools();
        closeToolPasswordModal(true);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Import failed');
      } finally {
        setPasswordModalBusy(false);
      }
    }
  };

  const handleEditTool = useCallback((tool: ToolConfig) => {
    setEditingTool(tool);
    setShowWizard(true);
  }, []);

  const handleWizardClose = () => {
    setShowWizard(false);
    setEditingTool(null);
  };

  const handleWizardSave = async () => {
    setShowWizard(false);
    setEditingTool(null);
    await loadTools();
    toast.success('Tool configuration saved successfully');
  };

  const openSkillsDialog = useCallback((tool: ToolConfig) => {
    setSkillsDialogTarget({ owner: { kind: 'tool', id: tool.id }, ownerName: tool.name });
    setSkillsDialogOpen(true);
  }, []);

  const closeSkillsDialog = useCallback(() => {
    setSkillsDialogOpen(false);
    if (skillsDialogTarget) {
      restoreFocusIfLost(`tools-tool-${skillsDialogTarget.owner.id}`);
    }
  }, [skillsDialogTarget]);

  const handleSkillsDialogChanged = useCallback(
    (skills: InstructionSkill[]) => {
      if (!skillsDialogTarget) return;
      const toolId = skillsDialogTarget.owner.id;
      setInstructionSkillSummary((current) => ({
        ...current,
        tools: { ...current.tools, [toolId]: countInstructionSkills(skills) },
      }));
    },
    [skillsDialogTarget],
  );

  const copyToolSkillsForDuplicate = useCallback(
    async (targetToolId: string, sourceToolId: string): Promise<boolean> => {
      try {
        const copied = await api.copyToolInstructionSkills(targetToolId, sourceToolId);
        setInstructionSkillSummary((current) => ({
          ...current,
          tools: { ...current.tools, [targetToolId]: countInstructionSkills(copied) },
        }));
        return true;
      } catch {
        return false;
      }
    },
    [],
  );

  const closeToolAccessModal = useCallback(() => {
    if (toolAccessSaving) {
      return;
    }
    setToolAccessModalTool(null);
    setToolAccessPolicy(null);
    setToolAccessLoading(false);
  }, [toolAccessSaving]);

  const handleOpenToolAccess = useCallback(
    async (tool: ToolConfig) => {
      setToolAccessModalTool(tool);
      setToolAccessPolicy(null);
      setToolAccessLoading(true);
      try {
        const [policy, users, groups] = await Promise.all([
          api.getToolAccessPolicy(tool.id),
          api.listUsers(),
          api.listAuthGroups(),
        ]);
        setToolAccessPolicy(policy);
        setToolAccessUserOptions(toToolAccessUserOptions(users));
        setToolAccessGroupOptions(toToolAccessGroupOptions(groups));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to load tool access');
        setToolAccessModalTool(null);
      } finally {
        setToolAccessLoading(false);
      }
    },
    [toast],
  );

  const handleSaveToolAccess = useCallback(
    async (policy: ToolAccessPolicy) => {
      if (!toolAccessModalTool) {
        return;
      }
      setToolAccessSaving(true);
      try {
        const savedPolicy = await api.updateToolAccessPolicy(toolAccessModalTool.id, policy);
        setToolAccessPolicy(savedPolicy);
        toast.success('Tool access saved');
        setToolAccessModalTool(null);
        setToolAccessPolicy(null);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to save tool access');
        throw err;
      } finally {
        setToolAccessSaving(false);
      }
    },
    [toast, toolAccessModalTool],
  );

  const handleContentProtectionModeChange = useCallback(
    (toolId: string, mode: 'inherit' | 'require') => {
      setContentProtectionConfig((current) => {
        if (!current) return current;
        return {
          ...current,
          requirements: [
            ...current.requirements.filter(
              (requirement) =>
                requirement.scope_kind !== 'tool' || requirement.scope_key !== toolId,
            ),
            ...(mode === 'require'
              ? [{ scope_kind: 'tool' as const, scope_key: toolId, mode }]
              : []),
          ],
        };
      });
    },
    [],
  );

  const patchToolInState = useCallback((toolId: string, updates: Partial<ToolConfig>) => {
    setTools((current) =>
      current.map((tool) => (tool.id === toolId ? { ...tool, ...updates } : tool)),
    );
  }, []);

  const replaceToolInState = useCallback((updatedTool: ToolConfig) => {
    setTools((current) => current.map((tool) => (tool.id === updatedTool.id ? updatedTool : tool)));
  }, []);

  const handleDeleteTool = async (
    toolId: string,
    options?: Parameters<typeof api.deleteToolConfig>[1],
  ) => {
    const response = options
      ? await api.deleteToolConfig(toolId, options)
      : await api.deleteToolConfig(toolId);
    setTools((current) =>
      current
        .filter((tool) => tool.id !== toolId)
        .map((tool) =>
          tool.pdm_link?.tool_id === toolId
            ? { ...tool, pdm_link: { ...tool.pdm_link, exists: false, enabled: false } }
            : tool,
        ),
    );
    setHeartbeats((current) => {
      const next = { ...current };
      delete next[toolId];
      return next;
    });
    setSchemaStats((current) => {
      const next = { ...current };
      delete next[toolId];
      return next;
    });
    return response;
  };

  const requestToolDeletion = (tool: ToolConfig) => {
    if (isPairedWindowsTool(tool)) {
      setWindowsDeleteTool(tool);
      return;
    }
    if (
      tool.tool_type === 'solidworks_pdm' &&
      pdmLinkRelations.some((relation) => relation.pdmToolId === tool.id)
    ) {
      setDeleteConfirmTool(tool);
      return;
    }
    void handleDeleteTool(tool.id)
      .then(() => toast.success('Tool deleted successfully'))
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Failed to delete tool'));
  };

  const handleToggleTool = async (toolId: string, enabled: boolean) => {
    try {
      const result = await api.toggleToolConfig(toolId, enabled);
      patchToolInState(toolId, { enabled: result.enabled });
      const pdmTool = tools.find(
        (tool) => tool.id === toolId && tool.tool_type === 'solidworks_pdm',
      );
      if (pdmTool) {
        setTools((current) =>
          current.map((tool) =>
            tool.pdm_link?.tool_id === toolId
              ? { ...tool, pdm_link: { ...tool.pdm_link, enabled: result.enabled } }
              : tool,
          ),
        );
        if (!result.enabled) {
          pdmLinkRelations
            .filter((relation) => relation.pdmToolId === toolId)
            .forEach((relation) => {
              toast.info(
                `“${pdmTool.name}” is disabled. The SolidWorks tool “${relation.solidworksToolName}” cannot search the vault until you enable it again.`,
              );
            });
        }
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to toggle tool');
    }
  };

  const handleToggleWrite = async (toolId: string, allowWrite: boolean) => {
    if (allowWrite) {
      const tool = tools.find((t) => t.id === toolId);
      if (tool) {
        closeToolMenu();
        setWriteConfirmTool(tool);
      }
      return;
    }

    try {
      const updatedTool = await api.updateToolConfig(toolId, { allow_write: false });
      replaceToolInState(updatedTool);
      toast.success('Write access disabled');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to disable write access');
    }
  };

  const handleConfirmWriteEnable = useCallback(async () => {
    if (!writeConfirmTool) return;
    const toolId = writeConfirmTool.id;
    setWriteConfirmTool(null);

    try {
      const updatedTool = await api.updateToolConfig(toolId, { allow_write: true });
      replaceToolInState(updatedTool);
      toast.success('Write access enabled');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to enable write access');
    }
  }, [writeConfirmTool, replaceToolInState, toast]);

  const handleClearUndecryptableCredentials = useCallback(async () => {
    if (!clearConfirmTool) return;
    const toolId = clearConfirmTool.id;
    setClearConfirmTool(null);

    try {
      const updatedTool = await api.clearToolUndecryptableCredentials(toolId);
      replaceToolInState(updatedTool);
      toast.success('Broken tool credentials cleared. Re-enter them to restore the connection.');
      handleEditTool(updatedTool);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to clear broken credentials');
    }
  }, [clearConfirmTool, handleEditTool, replaceToolInState, toast]);

  const handleDuplicateTool = useCallback(
    async (toolId: string) => {
      const tool = tools.find((candidate) => candidate.id === toolId);
      if (!tool) return;

      try {
        const newName = getDuplicateToolName(tool.name, tools);
        let newTool = await api.createToolConfig({
          name: newName,
          tool_type: tool.tool_type,
          description: tool.description || undefined,
          connection_config: tool.connection_config,
          max_results: tool.max_results,
          timeout_max_seconds: tool.timeout_max_seconds,
          allow_write: tool.allow_write,
          group_id: tool.group_id,
        });

        if (newTool.enabled !== tool.enabled) {
          newTool = await api.updateToolConfig(newTool.id, { enabled: tool.enabled });
        }

        const toolIdx = tools.findIndex((candidate) => candidate.id === toolId);
        if (toolIdx >= 0) {
          const nextTools = [...tools];
          nextTools.splice(toolIdx + 1, 0, newTool);
          setTools(nextTools);
          await api.reorderTools({ tool_ids: nextTools.map((candidate) => candidate.id) });
        } else {
          setTools((current) => [...current, newTool]);
          await api.reorderTools({
            tool_ids: [...tools.map((candidate) => candidate.id), newTool.id],
          });
        }

        const skillsCopied = await copyToolSkillsForDuplicate(newTool.id, toolId);
        if (skillsCopied) {
          toast.success('Tool duplicated');
        } else {
          toast.error('Tool duplicated, but its skills could not be copied');
        }
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to duplicate tool');
        await loadTools();
      }
    },
    [tools, toast, loadTools, copyToolSkillsForDuplicate],
  );

  const handleTestTool = async (toolId: string) => {
    try {
      setTestingToolId(toolId);
      const result = await api.testSavedToolConnection(toolId);
      await loadTools();
      setHeartbeats((current) => ({
        ...current,
        [toolId]: {
          tool_id: toolId,
          alive: result.success,
          latency_ms: null,
          error: result.success ? null : result.message,
          checked_at: new Date().toISOString(),
        },
      }));
      if (result.success) {
        toast.success('Connection test successful');
      } else {
        toast.error(`Connection test failed: ${result.message}`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setTestingToolId(null);
    }
  };

  const handlePdmReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setPdmIndexingToolId(toolId);
      toast.success(fullReindex ? 'Starting full PDM re-index...' : 'Starting PDM index update...');
      await api.triggerPdmIndex(toolId, fullReindex);
      toast.success(fullReindex ? 'Full PDM re-index started' : 'PDM index update started');
      setTimeout(() => loadPdmStats(tools), 2000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to trigger PDM index');
    } finally {
      setPdmIndexingToolId(null);
    }
  };

  const handleSchemaReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setSchemaIndexingToolId(toolId);
      toast.success('Starting schema re-index...');
      await api.triggerSchemaIndex(toolId, fullReindex);
      toast.success('Schema re-index started');
      // Notify parent to refresh schema jobs list
      onSchemaJobTriggered?.();
      // Refresh schema stats after a short delay to allow job to start
      setTimeout(() => loadSchemaStats(tools), 2000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to trigger schema index');
    } finally {
      setSchemaIndexingToolId(null);
    }
  };

  const handleInlineUpdate = async (
    toolId: string,
    updates: { name?: string; description?: string },
  ) => {
    try {
      const updatedTool = await api.updateToolConfig(toolId, updates);
      replaceToolInState(updatedTool);
      if (updates.name) {
        toast.success('Tool name updated');
      } else {
        toast.success('Description updated');
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update');
      throw err; // Re-throw to let ToolCard know the save failed
    }
  };

  // ---- Tool and group menus ----

  const closeToolMenu = useCallback(() => {
    const trigger = toolMenuTriggerRef.current;
    toolMenuTriggerRef.current = null;
    setToolMenu(null);
    trigger?.focus();
  }, []);

  const closeGroupMenu = useCallback(() => {
    const trigger = groupMenuTriggerRef.current;
    groupMenuTriggerRef.current = null;
    setGroupMenu(null);
    trigger?.focus();
  }, []);

  useMenuDismissal({
    open: toolMenu !== null,
    menuRef: toolMenuRef,
    triggerRef: toolMenuTriggerRef,
    onDismiss: closeToolMenu,
  });
  useMenuDismissal({
    open: groupMenu !== null,
    menuRef: groupMenuRef,
    triggerRef: groupMenuTriggerRef,
    onDismiss: closeGroupMenu,
  });

  const openToolMenuFromButton = (tool: ToolConfig, anchor: HTMLElement) => {
    if (toolMenuTriggerRef.current === anchor) {
      closeToolMenu();
      return;
    }
    const rect = anchor.getBoundingClientRect();
    toolMenuTriggerRef.current = anchor;
    setToolMenu({ toolId: tool.id, placement: { anchorRect: toAnchorRect(rect) } });
  };

  const openToolMenuAtPointer = (tool: ToolConfig, event: React.MouseEvent) => {
    event.preventDefault();
    toolMenuTriggerRef.current = null;
    setToolMenu({ toolId: tool.id, placement: { x: event.clientX, y: event.clientY } });
  };

  const openGroupMenuFromButton = (sectionId: string, anchor: HTMLElement) => {
    if (groupMenuTriggerRef.current === anchor) {
      closeGroupMenu();
      return;
    }
    const rect = anchor.getBoundingClientRect();
    groupMenuTriggerRef.current = anchor;
    setGroupMenu({ sectionId, placement: { anchorRect: toAnchorRect(rect) } });
  };

  const openGroupMenuAtPointer = (sectionId: string, event: React.MouseEvent) => {
    event.preventDefault();
    groupMenuTriggerRef.current = null;
    setGroupMenu({ sectionId, placement: { x: event.clientX, y: event.clientY } });
  };

  // Focus the element requested by the last state change. The request waits across renders until
  // its element exists; it is dropped only when its group no longer exists.
  useEffect(() => {
    const pending = pendingFocusRef.current;
    if (!pending) return;
    if (pending.groupId && !knownGroupIds.has(pending.groupId)) {
      pendingFocusRef.current = null;
      return;
    }
    const element = document.getElementById(pending.elementId);
    if (!element) return;
    pendingFocusRef.current = null;
    element.scrollIntoView?.({ block: 'center' });
    element.focus();
  });

  // One ordering write at a time: while it runs, moves and group moves are disabled.
  const runOrderChange = async <T,>(task: () => Promise<T>): Promise<T | null> => {
    if (orderLockRef.current) return null;
    orderLockRef.current = true;
    setOrderPending(true);
    try {
      return await task();
    } finally {
      orderLockRef.current = false;
      setOrderPending(false);
    }
  };

  // ---- Tool moves: one path for drag and drop, dialogs and keyboard steps ----

  const nextGroupSortOrder = groups.length
    ? Math.max(...groups.map((group) => group.sort_order)) + SORT_ORDER_STEP
    : 0;

  const canStepTool = (toolId: string, direction: 'earlier' | 'later'): boolean =>
    moveToolWithinSection({ tools, knownGroupIds, toolId, direction }) !== null;

  const commitToolOrder = async (input: {
    toolId: string;
    orderedIds: string[];
    targetGroupId: string | null;
    groupChanged: boolean;
    groupList: ToolGroup[];
  }): Promise<boolean> => (await runOrderChange(() => saveToolOrder(input))) ?? false;

  const saveToolOrder = async (input: {
    toolId: string;
    orderedIds: string[];
    targetGroupId: string | null;
    groupChanged: boolean;
    groupList: ToolGroup[];
  }): Promise<boolean> => {
    const tool = tools.find((candidate) => candidate.id === input.toolId);
    if (!tool) return false;
    const knownIds = new Set(input.groupList.map((group) => group.id));
    const staleGroup = Boolean(tool.group_id) && !knownIds.has(tool.group_id ?? '');
    const needsGroupWrite = input.groupChanged || (staleGroup && input.targetGroupId === null);
    const targetName = input.groupList.find((group) => group.id === input.targetGroupId)?.name;

    setTools((current) => {
      const byId = new Map(current.map((candidate) => [candidate.id, candidate]));
      return input.orderedIds.flatMap((id, index) => {
        const existing = byId.get(id);
        if (!existing) return [];
        const sort_order = index * SORT_ORDER_STEP;
        if (id === input.toolId && needsGroupWrite) {
          return [
            {
              ...existing,
              group_id: input.targetGroupId,
              group_name: targetName ?? null,
              sort_order,
            },
          ];
        }
        return [{ ...existing, sort_order }];
      });
    });

    try {
      if (needsGroupWrite) {
        const updated = await api.updateToolConfig(input.toolId, {
          group_id: input.targetGroupId ?? '',
        });
        setTools((current) =>
          current.map((candidate) =>
            candidate.id === updated.id
              ? { ...updated, sort_order: candidate.sort_order }
              : candidate,
          ),
        );
      }
      // Always the full global order: the API reorders the whole list, never a subset.
      await api.reorderTools({ tool_ids: input.orderedIds });
      return true;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to move tool');
      await loadTools();
      return false;
    }
  };

  const applyToolMove = async (input: {
    toolId: string;
    targetGroupId: string | null;
    insertIndex: number;
    groupList: ToolGroup[];
  }): Promise<boolean> => {
    const tool = tools.find((candidate) => candidate.id === input.toolId);
    if (!tool) return false;
    const knownIds = new Set(input.groupList.map((group) => group.id));
    const targetGroupId =
      input.targetGroupId && knownIds.has(input.targetGroupId) ? input.targetGroupId : null;
    const move = computeToolMove({
      tools,
      knownGroupIds: knownIds,
      toolId: tool.id,
      targetGroupId,
      insertIndex: input.insertIndex,
    });
    const staleGroup = Boolean(tool.group_id) && !knownIds.has(tool.group_id ?? '');
    if (!move && !(staleGroup && targetGroupId === null)) return true;
    return commitToolOrder({
      toolId: tool.id,
      orderedIds: move?.orderedIds ?? tools.map((candidate) => candidate.id),
      targetGroupId,
      groupChanged: move?.groupChanged ?? false,
      groupList: input.groupList,
    });
  };

  const moveToolOneStep = (toolId: string, direction: 'earlier' | 'later') => {
    const move = moveToolWithinSection({ tools, knownGroupIds, toolId, direction });
    if (!move) return;
    const section = sections.find((candidate) =>
      candidate.tools.some((tool) => tool.id === toolId),
    );
    void commitToolOrder({
      toolId,
      orderedIds: move.orderedIds,
      targetGroupId: section?.group?.id ?? null,
      groupChanged: false,
      groupList: groups,
    });
  };

  const handleMoveToGroup = async (target: MoveToGroupTarget) => {
    const tool = moveDialogTool;
    if (!tool) return;
    if (orderLockRef.current) {
      throw new Error('Another move is still saving. Try again when it finishes.');
    }
    let groupList = groups;
    let targetGroupId: string | null;
    let insertIndex: number;
    let groupLabel: string;
    if (target.kind === 'new') {
      const created = await api.createToolGroup({
        name: target.name,
        description: '',
        sort_order: nextGroupSortOrder,
      });
      groupList = [...groups, created];
      setGroups(groupList);
      targetGroupId = created.id;
      insertIndex = 0;
      groupLabel = created.name;
    } else {
      const targetSection = sections.find(
        (section) => section.id === (target.groupId ?? UNGROUPED_SECTION_ID),
      );
      targetGroupId = target.groupId;
      insertIndex = targetSection?.tools.length ?? 0;
      groupLabel = targetSection?.group?.name ?? UNGROUPED_LABEL;
    }
    const moved = await applyToolMove({ toolId: tool.id, targetGroupId, insertIndex, groupList });
    if (!moved) throw new Error('The tool was not moved. The list has been reloaded.');
    toast.success(`Moved “${tool.name}” to “${groupLabel}”`);
    setMoveDialogToolId(null);
    revealToolInSection(
      targetGroupId ?? UNGROUPED_SECTION_ID,
      tool,
      targetGroupId ? groupLabel : '',
    );
  };

  // ---- Group handlers ----

  const handleSaveGroupDialog = async (value: { name: string; description: string }) => {
    if (groupDialog?.mode === 'edit') {
      const updated = await api.updateToolGroup(groupDialog.group.id, value);
      setGroups((current) => current.map((group) => (group.id === updated.id ? updated : group)));
      setGroupDialog(null);
      return;
    }
    const created = await api.createToolGroup({ ...value, sort_order: nextGroupSortOrder });
    // A filter would hide the empty new section, so it is cleared to show the new group.
    if (isToolFiltering) toolFilter.clear();
    setGroups((current) => [...current, created]);
    const expanded = withSetMembers(collapsedIds, [created.id], false);
    setCollapsedIds(expanded);
    persistCollapsedSections(expanded);
    setGroupDialog(null);
    pendingFocusRef.current = { elementId: sectionToggleId(created.id), groupId: created.id };
  };

  const moveGroup = (groupId: string, offset: 1 | -1): Promise<unknown> =>
    runOrderChange(() => saveGroupOrder(groupId, offset));

  const saveGroupOrder = async (groupId: string, offset: 1 | -1) => {
    const index = orderedGroups.findIndex((group) => group.id === groupId);
    const nextIndex = index + offset;
    const moved = orderedGroups[index];
    if (!moved || nextIndex < 0 || nextIndex >= orderedGroups.length) return;

    const nextOrder = [...orderedGroups];
    nextOrder.splice(index, 1);
    nextOrder.splice(nextIndex, 0, moved);
    const sortOrders = new Map<string, number>(
      nextOrder.map((group, position): [string, number] => [group.id, position * SORT_ORDER_STEP]),
    );
    setGroups((current) =>
      current.map((group) => ({
        ...group,
        sort_order: sortOrders.get(group.id) ?? group.sort_order,
      })),
    );
    try {
      await api.reorderToolGroups({ group_ids: nextOrder.map((group) => group.id) });
      setAnnouncement(
        `Group “${moved.name}” moved to position ${nextIndex + 1} of ${nextOrder.length}.`,
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to move group');
      const freshGroups = await api.listToolGroups().catch(() => null);
      if (freshGroups) setGroups(freshGroups);
    }
  };

  const handleConfirmDeleteGroup = async () => {
    const group = deleteGroupTarget;
    if (!group) return;
    await api.deleteToolGroup(group.id);
    // Focus goes to the next group section that is actually rendered, else the New group button.
    const renderedGroups = visibleSections.flatMap(({ section }) =>
      section.group ? [section.group] : [],
    );
    const deletedIndex = renderedGroups.findIndex((candidate) => candidate.id === group.id);
    const nextRendered = deletedIndex >= 0 ? renderedGroups[deletedIndex + 1] : undefined;
    setGroups((current) => current.filter((candidate) => candidate.id !== group.id));
    setTools((current) =>
      current.map((tool) =>
        tool.group_id === group.id ? { ...tool, group_id: null, group_name: null } : tool,
      ),
    );
    setForcedOpenIds((current) => withSetMembers(current, [group.id], false));
    const nextCollapsed = withSetMembers(collapsedIds, [group.id], false);
    setCollapsedIds(nextCollapsed);
    persistCollapsedSections(nextCollapsed);
    setDeleteGroupTarget(null);
    pendingFocusRef.current = nextRendered
      ? { elementId: sectionToggleId(nextRendered.id), groupId: nextRendered.id }
      : { elementId: NEW_GROUP_BUTTON_ID, groupId: null };
    toast.success('Group deleted');
  };

  // ---------------------------------------------------------------------------
  // Mount source handlers
  // ---------------------------------------------------------------------------

  const handleMountSourceWizardSaved = useCallback(
    (saved: UserspaceMountSource) => {
      setMountSources((current) => {
        const next = current.some((source) => source.id === saved.id)
          ? current.map((source) => (source.id === saved.id ? saved : source))
          : [...current, saved];
        return [...next].sort((left, right) => left.name.localeCompare(right.name));
      });
      setShowMountSourceWizard(false);
      setEditingMountSource(null);
      toast.success(
        saved.id === editingMountSource?.id ? 'Mount source updated.' : 'Mount source created.',
        5000,
      );
    },
    [editingMountSource, toast],
  );

  const handleDeleteMountSource = useCallback(
    async (mountSourceId: string) => {
      setMountSourceDeletingId(mountSourceId);

      try {
        await api.deleteUserspaceMountSource(mountSourceId);
        setMountSources((current) => current.filter((source) => source.id !== mountSourceId));
        toast.success('Mount source deleted.', 5000);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to delete mount source');
      } finally {
        setMountSourceDeletingId(null);
      }
    },
    [toast],
  );

  const handleStartReconnectMountSource = useCallback(
    async (source: UserspaceMountSource & { source_type: 'microsoft_drive' | 'google_drive' }) => {
      setReconnectStartingMountSourceId(source.id);
      try {
        const response = await api.startUserCloudOAuth({
          provider: source.source_type,
          redirect_uri: getCloudOAuthCallbackUrl(),
        });
        setReconnectingMountSource({ mountSourceId: source.id, provider: source.source_type });
        window.open(response.auth_url, 'ragtime-cloud-oauth', 'popup,width=720,height=820');
      } catch (reconnectErr) {
        toast.error(
          reconnectErr instanceof Error
            ? reconnectErr.message
            : 'Failed to start cloud OAuth reconnect',
        );
        setReconnectingMountSource(null);
      } finally {
        setReconnectStartingMountSourceId(null);
      }
    },
    [toast],
  );

  const handleToggleMountSourceEnabled = useCallback(
    async (source: UserspaceMountSource) => {
      const nextEnabled = !source.enabled;

      if (!nextEnabled) {
        setDisableConfirmation({ source, affected: null, loading: true });
        try {
          const affected = await api.getMountSourceAffectedWorkspaces(source.id);
          if (affected.total_mounts > 0) {
            setDisableConfirmation({ source, affected, loading: false });
            return;
          }
          setDisableConfirmation(null);
        } catch {
          setDisableConfirmation(null);
        }
      }

      setMountSources((current) =>
        current.map((s) => (s.id === source.id ? { ...s, enabled: nextEnabled } : s)),
      );
      try {
        await api.updateUserspaceMountSource(source.id, { enabled: nextEnabled });
      } catch (err) {
        setMountSources((current) =>
          current.map((s) => (s.id === source.id ? { ...s, enabled: source.enabled } : s)),
        );
        if (source.source_unavailable_kind === 'cloud_auth' && isCloudMountSource(source)) {
          toast.error(
            <>
              Cloud authentication is unavailable for this mount source.{' '}
              <a
                href="#"
                onClick={(event) => {
                  event.preventDefault();
                  void handleStartReconnectMountSource(source);
                }}
                style={{ color: 'inherit', textDecoration: 'underline' }}
              >
                Reconnect now
              </a>
            </>,
            10000,
          );
        } else {
          toast.error(err instanceof Error ? err.message : 'Failed to update mount source');
        }
      }
    },
    [handleStartReconnectMountSource, toast],
  );

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (event.origin !== window.location.origin || !reconnectingMountSource) {
        return;
      }
      if (event.data?.type === 'ragtime-cloud-oauth-error') {
        toast.error(
          typeof event.data.message === 'string'
            ? event.data.message
            : 'Cloud OAuth reconnect failed',
        );
        setReconnectingMountSource(null);
        return;
      }
      if (event.data?.type !== 'ragtime-cloud-oauth-complete') {
        return;
      }

      void (async () => {
        try {
          const accounts = await api.listUserCloudOAuthAccounts();
          const callbackAccountId =
            typeof event.data?.account_id === 'string' ? event.data.account_id : null;
          const account = callbackAccountId
            ? (accounts.find(
                (item) =>
                  item.id === callbackAccountId &&
                  item.provider === reconnectingMountSource.provider &&
                  item.connected,
              ) ?? null)
            : pickLatestConnectedCloudAccount(accounts, reconnectingMountSource.provider);
          if (!account) {
            toast.error(
              'Connected account was not found after OAuth completion. Try reconnecting again.',
            );
            return;
          }

          const updated = await api.updateUserspaceMountSource(
            reconnectingMountSource.mountSourceId,
            {
              enabled: true,
              connection_config: {
                oauth_account_id: account.id,
              },
            },
          );

          setMountSources((current) =>
            current.map((source) =>
              source.id === updated.id ? { ...updated, usage_count: source.usage_count } : source,
            ),
          );
          toast.success('Cloud account reconnected and mount source re-enabled.', 5000);
        } catch (err) {
          toast.error(err instanceof Error ? err.message : 'Failed to reconnect mount source');
        } finally {
          setReconnectingMountSource(null);
        }
      })();
    };

    window.addEventListener('message', listener);
    return () => window.removeEventListener('message', listener);
  }, [reconnectingMountSource, toast]);

  const handleConfirmDisableMountSource = useCallback(async () => {
    if (!disableConfirmation) return;
    const { source } = disableConfirmation;
    setDisableConfirmation(null);
    setMountSources((current) =>
      current.map((s) => (s.id === source.id ? { ...s, enabled: false } : s)),
    );
    try {
      const updated = await api.updateUserspaceMountSource(source.id, { enabled: false });
      setMountSources((current) =>
        current.map((s) => (s.id === source.id ? { ...updated, usage_count: s.usage_count } : s)),
      );
    } catch (err) {
      setMountSources((current) =>
        current.map((s) => (s.id === source.id ? { ...s, enabled: true } : s)),
      );
      toast.error(err instanceof Error ? err.message : 'Failed to disable mount source');
    }
  }, [disableConfirmation, toast]);

  // ---- Drag and drop (disabled while filtering) ----

  const clearDragPreviewTimer = useCallback(() => {
    if (dragPreviewTimerRef.current) {
      clearTimeout(dragPreviewTimerRef.current);
      dragPreviewTimerRef.current = null;
    }
  }, []);

  const clearDragPreview = () => {
    clearDragPreviewTimer();
    pendingDragPreviewRef.current = null;
    setDragOverToolId(null);
  };

  const resetDragState = () => {
    clearDragPreview();
    setDragToolId(null);
    setBodyDropSectionId(null);
    setHeaderDropSectionId(null);
  };

  const scheduleDragPreview = (target: DragPreviewTarget) => {
    clearDragPreviewTimer();
    pendingDragPreviewRef.current = target;
    setDragOverToolId(null);

    dragPreviewTimerRef.current = setTimeout(() => {
      const pending = pendingDragPreviewRef.current;
      dragPreviewTimerRef.current = null;
      pendingDragPreviewRef.current = null;
      if (!pending) return;
      setDragOverToolId(pending.toolId);
      setDragInsertBefore(pending.insertBefore);
    }, DRAG_REORDER_PREVIEW_DELAY_MS);
  };

  const handleDragStart = (event: React.DragEvent, toolId: string) => {
    event.dataTransfer.setData('text/plain', toolId);
    event.dataTransfer.effectAllowed = 'move';
    setDragToolId(toolId);
  };

  // Drop slot shown in one section: index is "insert before the card at this index".
  const previewSlot = useMemo(() => {
    if (!dragToolId || !dragOverToolId) return null;
    const targetSection = sections.find((section) =>
      section.tools.some((tool) => tool.id === dragOverToolId),
    );
    if (!targetSection) return null;
    const targetIndex = targetSection.tools.findIndex((tool) => tool.id === dragOverToolId);
    const index = dragInsertBefore ? targetIndex : targetIndex + 1;
    const sourceSection = sections.find((section) =>
      section.tools.some((tool) => tool.id === dragToolId),
    );
    if (sourceSection?.id === targetSection.id) {
      const fromIndex = sourceSection.tools.findIndex((tool) => tool.id === dragToolId);
      if (index === fromIndex || index === fromIndex + 1) return null;
    }
    return { sectionId: targetSection.id, index };
  }, [dragToolId, dragOverToolId, dragInsertBefore, sections]);

  const handleToolCardDragOver = (event: React.DragEvent<HTMLElement>, targetToolId: string) => {
    if (!dragToolId || dragDisabled) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'move';
    setBodyDropSectionId(null);
    setHeaderDropSectionId(null);
    if (dragToolId === targetToolId) return;

    const insertBefore = isPointerBeforeMidpoint(event);
    if (dragOverToolId === targetToolId && dragInsertBefore === insertBefore) return;
    const pending = pendingDragPreviewRef.current;
    if (pending?.toolId === targetToolId && pending.insertBefore === insertBefore) return;
    scheduleDragPreview({ toolId: targetToolId, insertBefore });
  };

  const handleToolCardDragLeave = (event: React.DragEvent<HTMLElement>) => {
    const related = event.relatedTarget as HTMLElement | null;
    if (!event.currentTarget.contains(related) && !related?.closest('.tool-reorder-drop-slot')) {
      clearDragPreview();
    }
  };

  const handleToolCardDrop = async (event: React.DragEvent<HTMLElement>, targetToolId: string) => {
    event.preventDefault();
    event.stopPropagation();
    const draggedId = event.dataTransfer.getData('text/plain') || dragToolId;
    const insertBefore = isPointerBeforeMidpoint(event);
    resetDragState();
    if (!draggedId || draggedId === targetToolId || dragDisabled) return;

    const section = sections.find((candidate) =>
      candidate.tools.some((tool) => tool.id === targetToolId),
    );
    if (!section) return;
    const targetIndex = section.tools.findIndex((tool) => tool.id === targetToolId);
    await applyToolMove({
      toolId: draggedId,
      targetGroupId: section.group?.id ?? null,
      insertIndex: insertBefore ? targetIndex : targetIndex + 1,
      groupList: groups,
    });
  };

  const handleDropSlotDragOver = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'move';
  };

  const handleSlotDrop = async (
    event: React.DragEvent<HTMLElement>,
    section: ToolSection,
    index: number,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    const draggedId = event.dataTransfer.getData('text/plain') || dragToolId;
    resetDragState();
    if (!draggedId || dragDisabled) return;
    await applyToolMove({
      toolId: draggedId,
      targetGroupId: section.group?.id ?? null,
      insertIndex: index,
      groupList: groups,
    });
  };

  const handleSectionBodyDragOver = (event: React.DragEvent<HTMLElement>, section: ToolSection) => {
    if (!dragToolId || dragDisabled) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    setBodyDropSectionId(section.id);
    setHeaderDropSectionId(null);

    const lastTool = section.tools[section.tools.length - 1];
    if (!lastTool || lastTool.id === dragToolId) {
      clearDragPreview();
      return;
    }
    const pending = pendingDragPreviewRef.current;
    if (pending?.toolId === lastTool.id && !pending.insertBefore) return;
    if (dragOverToolId === lastTool.id && !dragInsertBefore) return;
    scheduleDragPreview({ toolId: lastTool.id, insertBefore: false });
  };

  const handleSectionBodyDragLeave = (event: React.DragEvent<HTMLElement>, sectionId: string) => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setBodyDropSectionId((current) => (current === sectionId ? null : current));
    clearDragPreview();
  };

  const handleSectionBodyDrop = async (
    event: React.DragEvent<HTMLElement>,
    section: ToolSection,
  ) => {
    event.preventDefault();
    const draggedId = event.dataTransfer.getData('text/plain') || dragToolId;
    resetDragState();
    if (!draggedId || dragDisabled) return;
    await applyToolMove({
      toolId: draggedId,
      targetGroupId: section.group?.id ?? null,
      insertIndex: section.tools.length,
      groupList: groups,
    });
  };

  const handleSectionHeaderDragOver = (
    event: React.DragEvent<HTMLElement>,
    section: ToolSection,
  ) => {
    if (!dragToolId || dragDisabled) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    clearDragPreview();
    setBodyDropSectionId(null);
    setHeaderDropSectionId(section.id);
  };

  const handleSectionHeaderDragLeave = (event: React.DragEvent<HTMLElement>, sectionId: string) => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setHeaderDropSectionId((current) => (current === sectionId ? null : current));
  };

  const handleSectionHeaderDrop = async (
    event: React.DragEvent<HTMLElement>,
    section: ToolSection,
  ) => {
    event.preventDefault();
    const draggedId = event.dataTransfer.getData('text/plain') || dragToolId;
    const wasCollapsed = isSectionCollapsed(section.id);
    resetDragState();
    const draggedTool = tools.find((tool) => tool.id === draggedId);
    if (!draggedId || !draggedTool || dragDisabled) return;

    const groupLabel = section.group?.name ?? UNGROUPED_LABEL;
    const moved = await applyToolMove({
      toolId: draggedId,
      targetGroupId: section.group?.id ?? null,
      insertIndex: section.tools.length,
      groupList: groups,
    });
    if (moved && wasCollapsed) {
      flashSectionHeader(section.id);
      toast.success(`Moved “${draggedTool.name}” to “${groupLabel}”`);
    }
  };

  const getSectionBodyHandlers = (section: ToolSection): SectionDropHandlers => ({
    onDragOver: (event) => handleSectionBodyDragOver(event, section),
    onDragLeave: (event) => handleSectionBodyDragLeave(event, section.id),
    onDrop: (event) => void handleSectionBodyDrop(event, section),
  });

  const getSectionHeaderHandlers = (section: ToolSection): SectionDropHandlers => ({
    onDragOver: (event) => handleSectionHeaderDragOver(event, section),
    onDragLeave: (event) => handleSectionHeaderDragLeave(event, section.id),
    onDrop: (event) => void handleSectionHeaderDrop(event, section),
  });

  useEffect(() => {
    return () => {
      clearDragPreviewTimer();
      pendingDragPreviewRef.current = null;
    };
  }, [clearDragPreviewTimer]);

  const renderToolCard = (tool: ToolConfig) => (
    <div
      key={tool.id}
      role="listitem"
      className={`tool-card-drag-wrap${dragToolId === tool.id ? ' dragging' : ''}`}
      draggable={!dragDisabled}
      onDragStart={(event) => handleDragStart(event, tool.id)}
      onDragEnd={resetDragState}
      onDragOver={(event) => handleToolCardDragOver(event, tool.id)}
      onDragLeave={handleToolCardDragLeave}
      onDrop={(event) => handleToolCardDrop(event, tool.id)}
      onContextMenu={(event) => openToolMenuAtPointer(tool, event)}
    >
      <ToolCard
        tool={tool}
        tools={tools}
        targetId={`tools-tool-${tool.id}`}
        heartbeat={heartbeats[tool.id] || null}
        showFooterActions={showFooterActions}
        onEdit={handleEditTool}
        onDelete={requestToolDeletion}
        onTest={handleTestTool}
        testing={testingToolId === tool.id}
        onPdmReindex={handlePdmReindex}
        pdmIndexing={pdmIndexingToolId === tool.id}
        pdmStats={pdmStats[tool.id] || null}
        onSchemaReindex={handleSchemaReindex}
        schemaIndexing={schemaIndexingToolId === tool.id}
        activeSchemaJob={activeSchemaJobsByToolId[tool.id] || null}
        schemaStats={schemaStats[tool.id] || null}
        onInlineUpdate={handleInlineUpdate}
        classified={cpEnabled && requiredToolIds.has(tool.id)}
        skillCount={instructionSkillSummary.tools[tool.id]}
        onOpenSkills={openSkillsDialog}
        onOpenActions={openToolMenuFromButton}
        actionsMenuOpen={toolMenu?.toolId === tool.id}
      />
    </div>
  );

  const renderDropSlot = (section: ToolSection, index: number) => (
    <div
      key={`tool-reorder-drop-slot-${section.id}-${index}`}
      className="tool-reorder-drop-slot"
      aria-hidden="true"
      onDragOver={handleDropSlotDragOver}
      onDrop={(event) => void handleSlotDrop(event, section, index)}
    />
  );

  const renderToolGrid = ({
    id,
    section,
    shownTools,
    bodyHandlers,
  }: {
    id: string;
    section: ToolSection;
    shownTools: ToolConfig[];
    bodyHandlers?: SectionDropHandlers;
  }) => {
    const items: JSX.Element[] = [];
    shownTools.forEach((tool, index) => {
      if (previewSlot?.sectionId === section.id && previewSlot.index === index) {
        items.push(renderDropSlot(section, index));
      }
      items.push(renderToolCard(tool));
    });
    if (previewSlot?.sectionId === section.id && previewSlot.index === shownTools.length) {
      items.push(renderDropSlot(section, shownTools.length));
    }
    return (
      <div id={id} className="tools-grid" role="list" {...bodyHandlers}>
        {items}
      </div>
    );
  };

  const renderToolSection = ({ section, shownTools }: VisibleToolSection) => {
    const groupName = section.group?.name ?? UNGROUPED_LABEL;
    const isUngrouped = section.group === null;
    const title = isUngrouped && dragToolId !== null ? UNGROUPED_DROP_TITLE : groupName;
    return (
      <ToolGroupSection
        key={section.id}
        sectionId={section.id}
        title={title}
        description={section.group?.description || undefined}
        toolCount={shownTools.length}
        totalCount={section.tools.length}
        disabledCount={shownTools.filter((tool) => !tool.enabled).length}
        health={summarizeToolGroupHealth(shownTools, heartbeats, testingToolId)}
        collapsed={isSectionCollapsed(section.id)}
        onToggle={toggleSection}
        onOpenActions={section.group ? openGroupMenuFromButton : undefined}
        actionsMenuOpen={groupMenu?.sectionId === section.id}
        onHeaderContextMenu={
          section.group ? (event) => openGroupMenuAtPointer(section.id, event) : undefined
        }
        dropTarget={
          headerDropSectionId === section.id
            ? 'header'
            : bodyDropSectionId === section.id
              ? 'body'
              : null
        }
        dropLabel={`Move to ${groupName}`}
        headerDropHandlers={getSectionHeaderHandlers(section)}
        bodyDropHandlers={getSectionBodyHandlers(section)}
        emptyMessage={section.group ? GROUP_EMPTY_MESSAGE : undefined}
      >
        {renderToolGrid({
          id: `tools-section-${section.id}-grid`,
          section,
          shownTools,
        })}
      </ToolGroupSection>
    );
  };

  return (
    <div ref={panelRef} id="tools-panel" className="tools-panel" tabIndex={-1}>
      <div className="card" id="tools-connections">
        <div className="card-header">
          <h2>Tool Connections</h2>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button type="button" className="btn btn-secondary" onClick={handleImportClick}>
              <Upload size={16} style={{ marginRight: 6, verticalAlign: 'text-bottom' }} />
              Import
            </button>
            <input
              type="file"
              accept=".json,application/json"
              ref={importFileInputRef}
              onChange={handleImportFileChange}
              style={{ display: 'none' }}
            />
            <AnimatedCreateButton
              isExpanded={showWizard}
              onClick={() => (showWizard ? handleWizardClose() : handleAddTool())}
              label="Add Tool"
            />
          </div>
        </div>

        {showWizard ? (
          <ToolWizard
            key={editingTool?.id ?? 'new'}
            existingTool={editingTool}
            onClose={handleWizardClose}
            onSave={handleWizardSave}
            embedded={true}
          />
        ) : (
          <>
            <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
            <ToolAccessModal
              open={toolAccessModalTool != null}
              toolName={toolAccessModalTool?.name ?? ''}
              policy={toolAccessPolicy}
              userOptions={toolAccessUserOptions}
              groupOptions={toolAccessGroupOptions}
              loading={toolAccessLoading}
              saving={toolAccessSaving}
              globalWriteEnabled={toolAccessModalTool?.allow_write ?? true}
              onChange={(policy) => setToolAccessPolicy(policy)}
              onSave={handleSaveToolAccess}
              onContentProtectionModeChange={handleContentProtectionModeChange}
              onClose={closeToolAccessModal}
            />
            {skillsDialogTarget && (
              <InstructionSkillsDialog
                open={skillsDialogOpen}
                owner={skillsDialogTarget.owner}
                ownerName={skillsDialogTarget.ownerName}
                onClose={closeSkillsDialog}
                onSkillsChanged={handleSkillsDialogChanged}
              />
            )}

            {toolMenu &&
              (() => {
                const contextTool = tools.find((t) => t.id === toolMenu.toolId);
                if (!contextTool) return null;
                const contextHasSchemaIndexing = hasSchemaIndexingEnabled(contextTool);
                const closeAnd = (fn: () => void) => {
                  closeToolMenu();
                  fn();
                };
                return (
                  <ContextMenu
                    ref={toolMenuRef}
                    autoFocus
                    ariaLabel={`Tool actions for ${contextTool.name}`}
                    onClose={closeToolMenu}
                    items={[
                      {
                        label: contextTool.enabled ? 'Enabled' : 'Disabled',
                        type: 'toggle',
                        checked: contextTool.enabled,
                        onSelect: () => {
                          handleToggleTool(contextTool.id, !contextTool.enabled);
                        },
                      },
                      ...(contextTool.tool_type === 'solidworks'
                        ? contextTool.allow_write
                          ? [
                              {
                                label: 'Allow changes',
                                type: 'toggle' as const,
                                checked: true,
                                onSelect: () => handleToggleWrite(contextTool.id, false),
                              },
                            ]
                          : []
                        : [
                            {
                              label: 'Write Access',
                              type: 'toggle' as const,
                              checked: contextTool.allow_write,
                              onSelect: () => {
                                handleToggleWrite(contextTool.id, !contextTool.allow_write);
                              },
                            },
                          ]),
                      {
                        label: 'Test',
                        icon: 'play',
                        onSelect: () => closeAnd(() => handleTestTool(contextTool.id)),
                        disabled: testingToolId === contextTool.id,
                      },
                      {
                        label: 'Edit',
                        icon: 'pencil',
                        onSelect: () => closeAnd(() => handleEditTool(contextTool)),
                      },
                      {
                        label: 'Edit Users',
                        icon: 'shield-check',
                        onSelect: () => closeAnd(() => void handleOpenToolAccess(contextTool)),
                      },
                      {
                        label: 'Skills',
                        icon: 'book',
                        description: (
                          <span data-instruction-skills-menu-description>
                            {describeInstructionSkillCount(
                              instructionSkillSummary.tools[contextTool.id],
                              contextTool.tool_type,
                            )}
                          </span>
                        ),
                        onSelect: () => closeAnd(() => openSkillsDialog(contextTool)),
                      },
                      ...(contextTool.undecryptable_fields.length > 0
                        ? [
                            {
                              label: 'Clear broken credentials...' as const,
                              icon: 'alert-circle' as const,
                              description: 'Remove fields that cannot be decrypted',
                              onSelect: () => closeAnd(() => setClearConfirmTool(contextTool)),
                            },
                          ]
                        : []),
                      ...(contextTool.tool_type === 'solidworks_pdm'
                        ? [
                            {
                              label: 'Index',
                              icon: 'folder-search' as const,
                              onSelect: () =>
                                closeAnd(() => handlePdmReindex(contextTool.id, false)),
                              disabled: pdmIndexingToolId === contextTool.id,
                            },
                            {
                              label: 'Full Re-index',
                              icon: 'rotate-ccw' as const,
                              onSelect: () =>
                                closeAnd(() => handlePdmReindex(contextTool.id, true)),
                              disabled: pdmIndexingToolId === contextTool.id,
                            },
                          ]
                        : []),
                      ...(contextHasSchemaIndexing
                        ? [
                            {
                              label: 'Re-index Schema',
                              icon: 'refresh' as const,
                              onSelect: () =>
                                closeAnd(() => handleSchemaReindex(contextTool.id, true)),
                              disabled: schemaIndexingToolId === contextTool.id,
                            },
                          ]
                        : []),
                      {
                        label: 'Move to group…',
                        icon: 'folder',
                        disabled: orderPending,
                        onSelect: () => closeAnd(() => setMoveDialogToolId(contextTool.id)),
                      },
                      {
                        label: 'Move earlier',
                        icon: 'chevron-up',
                        disabled: orderPending || !canStepTool(contextTool.id, 'earlier'),
                        onSelect: () => closeAnd(() => moveToolOneStep(contextTool.id, 'earlier')),
                      },
                      {
                        label: 'Move later',
                        icon: 'chevron-down',
                        disabled: orderPending || !canStepTool(contextTool.id, 'later'),
                        onSelect: () => closeAnd(() => moveToolOneStep(contextTool.id, 'later')),
                      },
                      {
                        label: 'Duplicate',
                        icon: 'copy',
                        onSelect: () => closeAnd(() => void handleDuplicateTool(contextTool.id)),
                      },
                      {
                        label: 'Export',
                        icon: 'download' as const,
                        onSelect: () =>
                          closeAnd(() =>
                            setToolPasswordModal({ mode: 'export', toolId: contextTool.id }),
                          ),
                      },
                      {
                        label: 'Delete',
                        icon: 'trash',
                        onSelect: () =>
                          closeAnd(() => {
                            if (isPairedWindowsTool(contextTool)) {
                              setWindowsDeleteTool(contextTool);
                            } else {
                              setDeleteConfirmTool(contextTool);
                            }
                          }),
                      },
                    ]}
                    {...toolMenu.placement}
                  />
                );
              })()}

            {groupMenu &&
              (() => {
                const group = orderedGroups.find(
                  (candidate) => candidate.id === groupMenu.sectionId,
                );
                if (!group) return null;
                const groupIndex = orderedGroups.indexOf(group);
                const closeAnd = (fn: () => void) => {
                  closeGroupMenu();
                  fn();
                };
                return (
                  <ContextMenu
                    ref={groupMenuRef}
                    autoFocus
                    ariaLabel={`Group actions for ${group.name}`}
                    onClose={closeGroupMenu}
                    items={[
                      {
                        label: 'Edit group…',
                        icon: 'pencil',
                        onSelect: () => closeAnd(() => setGroupDialog({ mode: 'edit', group })),
                      },
                      {
                        label: 'Move up',
                        icon: 'chevron-up',
                        disabled: orderPending || groupIndex <= 0,
                        onSelect: () => closeAnd(() => void moveGroup(group.id, -1)),
                      },
                      {
                        label: 'Move down',
                        icon: 'chevron-down',
                        disabled: orderPending || groupIndex === orderedGroups.length - 1,
                        onSelect: () => closeAnd(() => void moveGroup(group.id, 1)),
                      },
                      {
                        label: 'Delete group…',
                        icon: 'trash',
                        onSelect: () => closeAnd(() => setDeleteGroupTarget(group)),
                      },
                    ]}
                    {...groupMenu.placement}
                  />
                );
              })()}

            {groupDialog && (
              <ToolGroupDialog
                key={groupDialog.mode === 'edit' ? `edit-${groupDialog.group.id}` : 'create'}
                mode={groupDialog.mode}
                initialName={groupDialog.mode === 'edit' ? groupDialog.group.name : ''}
                initialDescription={
                  groupDialog.mode === 'edit' ? groupDialog.group.description : ''
                }
                existingNames={groups.map((group) => group.name)}
                onSubmit={handleSaveGroupDialog}
                onClose={() => setGroupDialog(null)}
              />
            )}
            {moveDialogTool && (
              <MoveToGroupDialog
                toolName={moveDialogTool.name}
                currentGroupId={moveDialogTool.group_id ?? null}
                groups={moveTargetGroups}
                onMove={handleMoveToGroup}
                onClose={() => setMoveDialogToolId(null)}
              />
            )}
            {deleteGroupTarget && (
              <ToolGroupDeleteDialog
                groupName={deleteGroupTarget.name}
                toolCount={
                  sections.find((section) => section.id === deleteGroupTarget.id)?.tools.length ?? 0
                }
                onConfirm={handleConfirmDeleteGroup}
                onClose={() => setDeleteGroupTarget(null)}
              />
            )}

            <p className="fieldset-help">
              Configure connections to databases, shells, and other tools that the AI can use during
              conversations. Each tool can have multiple instances (e.g., production and staging
              databases). To group a tool, create a group and drag the tool into it. Deleting a
              group does not delete the tools.
            </p>

            {loading ? (
              <p className="muted">Loading tools...</p>
            ) : tools.length === 0 && groups.length === 0 ? (
              <div className="empty-state">
                <p>No tools configured yet.</p>
                <p className="muted">Click "Add Tool" to set up your first connection.</p>
              </div>
            ) : (
              <>
                <div id="tools-toolbar" className="tools-toolbar">
                  <SearchFilterBar
                    state={toolFilter}
                    inputRef={toolFilterInputRef}
                    placeholder="Filter tools by keyword..."
                    ariaLabel="Filter tools by keyword"
                    className="tools-toolbar-search"
                  />
                  <button
                    id={NEW_GROUP_BUTTON_ID}
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setGroupDialog({ mode: 'create' })}
                  >
                    New group
                  </button>
                  {groups.length > 0 && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={toggleAllSections}
                      disabled={visibleSections.length === 0}
                    >
                      {allSectionsCollapsed ? 'Expand all' : 'Collapse all'}
                    </button>
                  )}
                  <span className="tools-toolbar-summary" aria-live="polite">
                    {summaryText}
                  </span>
                </div>
                <p className="rc-visually-hidden" aria-live="polite">
                  {announcement}
                </p>
                {groups.length === 0 ? (
                  matchedTools.length === 0 && isToolFiltering ? (
                    <p className="muted">No tools match the current filters.</p>
                  ) : (
                    renderToolGrid({
                      id: 'tools-grid-all',
                      section: plainSection,
                      shownTools: matchedTools,
                      bodyHandlers: getSectionBodyHandlers(plainSection),
                    })
                  )
                ) : visibleSections.length === 0 ? (
                  <p className="muted">No tools match the current filters.</p>
                ) : (
                  visibleSections.map(renderToolSection)
                )}
              </>
            )}
          </>
        )}
      </div>

      {/* Mounts section */}
      <div className="card" id="tools-mount-sources">
        <div className="card-header">
          <h2>Mounts</h2>
          <AnimatedCreateButton
            isExpanded={showMountSourceWizard}
            onClick={() => {
              if (showMountSourceWizard) {
                setShowMountSourceWizard(false);
                setEditingMountSource(null);
              } else {
                setEditingMountSource(null);
                setShowMountSourceWizard(true);
              }
            }}
            label="New Mount Source"
          />
        </div>

        {showMountSourceWizard ? (
          <MountSourceWizard
            existingSource={editingMountSource}
            existingNames={mountSources.map((s) => s.name)}
            onClose={() => {
              setShowMountSourceWizard(false);
              setEditingMountSource(null);
            }}
            onSaved={handleMountSourceWizardSaved}
            embedded={true}
          />
        ) : (
          <>
            <p className="fieldset-help">
              Define mount sources backed by SSH, filesystem, OneDrive, or Google Drive. Workspaces
              attach these sources without duplicating connection credentials.
            </p>

            <div
              style={{
                display: 'grid',
                gap: 12,
                gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
              }}
            >
              {mountSources.length > 0 ? (
                mountSources.map((source) => (
                  <div
                    key={source.id}
                    id={`tools-mount-source-${source.id}`}
                    className="card mount-source-card"
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                      padding: '12px 14px',
                      opacity: source.enabled ? 1 : 0.6,
                    }}
                  >
                    {(() => {
                      const sourceUnavailable = source.enabled && source.source_available === false;
                      const sourceUnavailableReason =
                        source.source_unavailable_reason ||
                        'Mount source is currently unavailable.';
                      const sourceNeedsOAuthReconnect =
                        source.source_unavailable_kind === 'cloud_auth' &&
                        isCloudMountSource(source);
                      return (
                        <>
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              gap: 8,
                            }}
                          >
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 8,
                                flex: 1,
                                minWidth: 0,
                              }}
                            >
                              <HardDrive size={14} style={{ flexShrink: 0 }} />
                              <span
                                style={{
                                  fontWeight: 500,
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                {source.name}
                              </span>
                              <span className="muted" style={{ fontSize: '0.8rem', flexShrink: 0 }}>
                                {(() => {
                                  switch (source.source_type) {
                                    case 'ssh':
                                      return 'SSH';
                                    case 'microsoft_drive':
                                      return 'OneDrive';
                                    case 'google_drive':
                                      return 'Google Drive';
                                    default:
                                      return source.mount_backend.replace('_', ' ');
                                  }
                                })()}
                              </span>
                              {sourceUnavailable && (
                                <span
                                  className="userspace-status-pill userspace-status-pill-danger"
                                  style={{ fontSize: 11 }}
                                  title={sourceUnavailableReason}
                                >
                                  Disconnected
                                </span>
                              )}
                              {source.usage_count > 0 && (
                                <span
                                  style={{
                                    fontSize: '0.7rem',
                                    fontWeight: 600,
                                    padding: '1px 6px',
                                    borderRadius: 8,
                                    background: 'var(--color-accent)',
                                    color: 'var(--color-bg)',
                                    flexShrink: 0,
                                  }}
                                  title={`Used by ${source.usage_count} workspace${source.usage_count !== 1 ? 's' : ''}`}
                                >
                                  {source.usage_count}
                                </span>
                              )}
                            </span>
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 6,
                                flexShrink: 0,
                              }}
                            >
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => {
                                  setEditingMountSource(source);
                                  setShowMountSourceWizard(true);
                                }}
                              >
                                <Pencil size={12} />
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => void handleDeleteMountSource(source.id)}
                                disabled={mountSourceDeletingId === source.id}
                              >
                                <Trash2 size={12} />
                              </button>
                              <label className="toggle-switch" style={{ marginLeft: 2 }}>
                                <input
                                  type="checkbox"
                                  checked={source.enabled}
                                  onChange={() => void handleToggleMountSourceEnabled(source)}
                                  disabled={reconnectStartingMountSourceId === source.id}
                                />
                                <span className="toggle-slider"></span>
                              </label>
                            </span>
                          </div>
                          {sourceUnavailable && (
                            <div
                              style={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: 6,
                                marginTop: 2,
                                color: 'var(--color-error, #c0392b)',
                                fontSize: 12,
                              }}
                            >
                              <Icon name="alert-circle" size={12} />
                              <span>
                                {sourceUnavailableReason}
                                {sourceNeedsOAuthReconnect && (
                                  <>
                                    {' '}
                                    <a
                                      href="#"
                                      onClick={(event) => {
                                        event.preventDefault();
                                        void handleStartReconnectMountSource(source);
                                      }}
                                      style={{ color: 'inherit', textDecoration: 'underline' }}
                                    >
                                      Reconnect OAuth
                                    </a>
                                  </>
                                )}
                              </span>
                            </div>
                          )}
                          {source.approved_paths.length > 0 && (
                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                              {source.approved_paths.map((p) => {
                                const displayPath = resolveSourceDisplayPath(p, undefined, {
                                  sourceType: source.source_type,
                                });
                                return (
                                  <code
                                    key={p}
                                    style={{
                                      fontSize: '0.8rem',
                                      padding: '2px 8px',
                                      background: 'var(--color-bg-tertiary)',
                                      borderRadius: 4,
                                      border: '1px solid var(--color-border)',
                                    }}
                                  >
                                    {displayPath === '/'
                                      ? '/'
                                      : displayPath
                                          .split('/')
                                          .filter(Boolean)
                                          .map((seg, i) => (
                                            <span key={i}>
                                              <span style={{ opacity: 0.5 }}>/</span>
                                              {seg}
                                            </span>
                                          ))}
                                  </code>
                                );
                              })}
                            </div>
                          )}
                          {isAutoSyncMountSource(source) &&
                            source.sync_interval_seconds != null && (
                              <span className="muted" style={{ fontSize: '0.75rem' }}>
                                Sync interval:{' '}
                                {formatMountSourceInterval(source.sync_interval_seconds)}
                              </span>
                            )}
                          {(source.source_type === 'microsoft_drive' ||
                            source.source_type === 'google_drive') &&
                            source.account_email && (
                              <span className="muted" style={{ fontSize: '0.75rem' }}>
                                Account: {source.account_email}
                              </span>
                            )}
                        </>
                      );
                    })()}
                  </div>
                ))
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  No mount sources configured yet.
                </p>
              )}
            </div>
          </>
        )}
      </div>

      {/* Mount Source Disable Confirmation Modal */}
      {disableConfirmation && (
        <div className="modal-overlay" onClick={() => setDisableConfirmation(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Disable Mount Source</h3>
              <button className="modal-close" onClick={() => setDisableConfirmation(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              {disableConfirmation.loading ? (
                <p className="muted">Loading affected workspaces...</p>
              ) : (
                <>
                  <p>
                    Disabling <strong>{disableConfirmation.source.name}</strong> will immediately
                    stop
                    {disableConfirmation.source.source_type === 'ssh'
                      ? ' SFTP sync'
                      : ' mount access'}{' '}
                    for{' '}
                    <strong>
                      {disableConfirmation.affected?.total_mounts ??
                        disableConfirmation.source.usage_count}
                    </strong>{' '}
                    mount
                    {(disableConfirmation.affected?.total_mounts ??
                      disableConfirmation.source.usage_count) !== 1
                      ? 's'
                      : ''}{' '}
                    across <strong>{disableConfirmation.affected?.workspaces.length ?? '?'}</strong>{' '}
                    workspace
                    {(disableConfirmation.affected?.workspaces.length ?? 0) !== 1 ? 's' : ''}.
                  </p>
                  {disableConfirmation.affected &&
                    disableConfirmation.affected.workspaces.length > 0 && (
                      <div
                        style={{
                          maxHeight: 200,
                          overflowY: 'auto',
                          margin: '12px 0',
                          border: '1px solid var(--color-border)',
                          borderRadius: 6,
                        }}
                      >
                        <table
                          style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}
                        >
                          <thead>
                            <tr
                              style={{
                                borderBottom: '1px solid var(--color-border)',
                                background: 'var(--color-bg-tertiary)',
                              }}
                            >
                              <th style={{ padding: '6px 10px', textAlign: 'left' }}>Workspace</th>
                              <th style={{ padding: '6px 10px', textAlign: 'right' }}>Mounts</th>
                            </tr>
                          </thead>
                          <tbody>
                            {disableConfirmation.affected.workspaces.map((ws) => (
                              <tr
                                key={ws.workspace_id}
                                style={{ borderBottom: '1px solid var(--color-border)' }}
                              >
                                <td style={{ padding: '6px 10px' }}>{ws.workspace_name}</td>
                                <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                                  {ws.mount_count}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  <p className="field-help" style={{ margin: '8px 0 0' }}>
                    {disableConfirmation.source.source_type === 'ssh'
                      ? 'Auto-sync will be stopped for all affected mounts. Re-enabling the source will allow mounts to resume syncing.'
                      : 'Workspace access to this mount will be interrupted. Re-enabling restores access without reconfiguration.'}
                  </p>
                </>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setDisableConfirmation(null)}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => void handleConfirmDisableMountSource()}
                disabled={disableConfirmation.loading}
              >
                Disable Mount Source
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Write Access Enable Confirmation Modal */}
      {writeConfirmTool && (
        <div className="modal-overlay" onClick={() => setWriteConfirmTool(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Enable Write Access</h3>
              <button className="modal-close" onClick={() => setWriteConfirmTool(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p>
                Are you sure you want to enable write access for{' '}
                <strong>{writeConfirmTool.name}</strong>?
              </p>
              <p className="field-help" style={{ marginTop: 8 }}>
                This allows the AI to modify data, run destructive queries, and make changes to the
                connected system.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setWriteConfirmTool(null)}>
                Cancel
              </button>
              <button className="btn btn-danger" onClick={() => void handleConfirmWriteEnable()}>
                Enable Write Access
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Tool Confirmation Modal */}
      {deleteConfirmTool && (
        <div className="modal-overlay" onClick={() => setDeleteConfirmTool(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Delete Tool</h3>
              <button className="modal-close" onClick={() => setDeleteConfirmTool(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p>
                Are you sure you want to delete <strong>{deleteConfirmTool.name}</strong>?
              </p>
              <p className="field-help" style={{ marginTop: 8 }}>
                This will permanently remove the tool configuration. This action cannot be undone.
              </p>
              {deleteConfirmTool.tool_type === 'solidworks_pdm' &&
                pdmLinkRelations.some(
                  (relation) => relation.pdmToolId === deleteConfirmTool.id,
                ) && (
                  <div id="tool-delete-linked-note" role="note">
                    {pdmLinkRelations
                      .filter((relation) => relation.pdmToolId === deleteConfirmTool.id)
                      .map((relation) => (
                        <p key={relation.solidworksToolId}>
                          “{deleteConfirmTool.name}” is linked to the SolidWorks tool “
                          {relation.solidworksToolName}”. If you delete it, that tool can no longer
                          search the vault or open, add, check out or check in vault files until you
                          link another PDM tool. The SolidWorks tool keeps its other settings and
                          shows a warning.
                          {relation.allowWrite &&
                            ` Changes are currently on in ${relation.approvedFolderCount} approved folders.`}
                        </p>
                      ))}
                  </div>
                )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setDeleteConfirmTool(null)}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => {
                  const toolId = deleteConfirmTool.id;
                  setDeleteConfirmTool(null);
                  void handleDeleteTool(toolId)
                    .then(() => toast.success('Tool deleted successfully'))
                    .catch((err) =>
                      toast.error(err instanceof Error ? err.message : 'Failed to delete tool'),
                    );
                }}
              >
                Delete Tool
              </button>
            </div>
          </div>
        </div>
      )}
      {windowsDeleteTool && (
        <ToolDeleteDialog
          toolName={windowsDeleteTool.name}
          fallbackFocusRef={panelRef}
          onClose={() => setWindowsDeleteTool(null)}
          onDelete={(options) => handleDeleteTool(windowsDeleteTool.id, options)}
        />
      )}
      {/* Tool Export/Import Password Modal */}

      {clearConfirmTool && (
        <div className="modal-overlay" onClick={() => setClearConfirmTool(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Clear Broken Credentials</h3>
              <button className="modal-close" onClick={() => setClearConfirmTool(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p>
                Some credentials for <strong>{clearConfirmTool.name}</strong> cannot be decrypted
                with the current server key.
              </p>
              <p>The following encrypted credential fields will be removed:</p>
              <ul style={{ margin: 'var(--space-sm) 0', paddingLeft: 'var(--space-md)' }}>
                {clearConfirmTool.undecryptable_fields.map((field) => (
                  <li key={field}>
                    <code>{field}</code>
                  </li>
                ))}
              </ul>
              <p>
                Clearing removes only the listed broken fields. You must re-enter them in the next
                step to restore the connection.
              </p>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setClearConfirmTool(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleClearUndecryptableCredentials}
              >
                Clear Credentials
              </button>
            </div>
          </div>
        </div>
      )}

      {toolPasswordModal && (
        <div className="modal-overlay" onClick={() => closeToolPasswordModal()}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {toolPasswordModal.mode === 'export' ? 'Export Tool Config' : 'Import Tool Config'}
              </h3>
              <button className="modal-close" onClick={() => closeToolPasswordModal()}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              {toolPasswordModal.mode === 'export' && (
                <p className="field-help" style={{ marginBottom: 'var(--space-md)' }}>
                  This export is encrypted with your password. Choose a strong one, since anyone who
                  obtains the file can attempt to crack a weak password offline.
                </p>
              )}
              <div className="form-group">
                <label htmlFor="tool-export-import-password">Password</label>
                <input
                  id="tool-export-import-password"
                  type="password"
                  value={toolPassword}
                  onChange={(e) => setToolPassword(e.target.value)}
                  placeholder="Enter password"
                  autoFocus
                />
                {toolPasswordModal.mode === 'export' && (
                  <PasswordRequirementsChecklist
                    password={toolPassword}
                    policy={exportPasswordPolicy}
                  />
                )}
              </div>
              {toolPasswordModal.mode === 'export' && (
                <div className="form-group" style={{ marginTop: 'var(--space-md)' }}>
                  <label htmlFor="tool-export-confirm-password">Confirm Password</label>
                  <input
                    id="tool-export-confirm-password"
                    type="password"
                    value={toolPasswordConfirm}
                    onChange={(e) => setToolPasswordConfirm(e.target.value)}
                    placeholder="Re-enter password"
                  />
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => closeToolPasswordModal()}
                disabled={passwordModalBusy}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={() => void handleToolPasswordModalSubmit()}
                disabled={
                  passwordModalBusy ||
                  (toolPasswordModal.mode === 'export' &&
                    (!passwordMeetsRequirements(toolPassword, exportPasswordPolicy) ||
                      toolPassword !== toolPasswordConfirm))
                }
              >
                {passwordModalBusy
                  ? toolPasswordModal.mode === 'export'
                    ? 'Exporting...'
                    : 'Importing...'
                  : toolPasswordModal.mode === 'export'
                    ? 'Export'
                    : 'Import'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
