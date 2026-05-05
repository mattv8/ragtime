import { useState, useEffect, useCallback, useMemo, useRef, Fragment } from 'react';
import { Pencil, Shield, UserPlus } from 'lucide-react';
import { api, ApiError } from '@/api';
import type {
  AvailableModel,
  Conversation,
  User,
  AuthGroup,
  UserUsageSummary,
  ProviderModelBreakdown,
  DailyUsageTrend,
  ApiDailyTrend,
  UserDailyUsageSeriesPoint,
  McpUserUsage,
  McpDailyTrend,
  McpRouteUsage,
  UserSpaceWorkspace,
  UserSpaceWorkspaceDeleteTask,
  WorkspaceConversationStateSummaryItem,
} from '@/types';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Tooltip,
  Legend,
  type ChartOptions,
} from 'chart.js';
import { Bar, Chart, Line } from 'react-chartjs-2';
import { WorkspaceRowList } from './shared/WorkspaceRowList';
import { UserConversationRowList } from './shared/UserConversationRowList';
import { DataTable, type DataTableColumn, type TableSortConfig } from './shared/DataTable';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { useToast, ToastContainer } from './shared/Toast';
import { CheckboxDropdown } from './shared/CheckboxDropdown';
import { formatProviderDisplayName, formatModelDisplayName } from '@/utils/modelDisplay';
import { calculateConversationContextUsage, parseStoredModelIdentifier } from '@/utils/contextUsage';
import { AuthAdminModalHost } from './shared/AuthAdminModals';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Filler, Tooltip, Legend);

const CHART_PALETTE = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6'];

function useThemeColors() {
  const read = useCallback(() => {
    const s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue('--color-text-primary').trim() || '#f1f5f9',
      textSecondary: s.getPropertyValue('--color-text-secondary').trim() || '#94a3b8',
      grid: s.getPropertyValue('--color-border').trim() || 'rgba(148,163,184,0.12)',
    };
  }, []);

  const [colors, setColors] = useState(read);

  useEffect(() => {
    const observer = new MutationObserver(() => setColors(read()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => setColors(read());
    mq.addEventListener('change', handler);
    return () => { observer.disconnect(); mq.removeEventListener('change', handler); };
  }, [read]);

  return colors;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'n/a';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'n/a';
  return date.toLocaleString();
}

function toEpochMs(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function resolveConversationContextLimit(storedModel: string | null | undefined, availableModels: AvailableModel[], fallbackLimit: number): number {
  const parsed = parseStoredModelIdentifier(storedModel || '');
  const modelId = parsed.modelId.trim();
  const explicitProvider = (parsed.provider || '').trim().toLowerCase();

  if (!modelId) return fallbackLimit;

  let matchedModel: AvailableModel | undefined;
  if (explicitProvider) {
    matchedModel = availableModels.find((model) => model.id === modelId && String(model.provider).toLowerCase() === explicitProvider);
  }

  if (!matchedModel) {
    matchedModel = availableModels.find((model) => model.id === modelId);
  }

  if (!matchedModel && modelId.includes('/')) {
    const slashIndex = modelId.indexOf('/');
    const providerFromModelId = modelId.slice(0, slashIndex).trim().toLowerCase();
    const providerModelId = modelId.slice(slashIndex + 1).trim();
    matchedModel = availableModels.find((model) => (
      model.id === providerModelId
      && (!explicitProvider || String(model.provider).toLowerCase() === explicitProvider || String(model.provider).toLowerCase() === providerFromModelId)
    ));
  }

  return matchedModel?.context_limit || fallbackLimit;
}

function normalizeDateLabel(value: string): string {
  return value.includes('T') ? value.slice(0, 10) : value;
}

function buildDateLabels(days: number): string[] {
  const labels: string[] = [];
  const todayUtc = new Date();
  const base = new Date(Date.UTC(todayUtc.getUTCFullYear(), todayUtc.getUTCMonth(), todayUtc.getUTCDate()));

  for (let offset = days - 1; offset >= 0; offset -= 1) {
    const current = new Date(base);
    current.setUTCDate(base.getUTCDate() - offset);
    labels.push(current.toISOString().slice(0, 10));
  }

  return labels;
}

function paginate<T>(items: T[], page: number, pageSize: number): { pageItems: T[]; totalPages: number; safePage: number } {
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const start = (safePage - 1) * pageSize;
  return { pageItems: items.slice(start, start + pageSize), totalPages, safePage };
}

function isInternalAuthGroup(group: AuthGroup): boolean {
  return group.provider === 'local_managed';
}

function getUserManualGroupIds(user: User): string[] {
  return user.manual_group_ids ?? user.local_group_ids ?? [];
}

function formatGroupIdentifierForDisplay(identifier: string): string {
  const firstPart = identifier.split(',', 1)[0] || identifier;
  if (firstPart.includes('=')) {
    return firstPart.split('=').slice(1).join('=') || identifier;
  }
  return firstPart;
}

type PanelTab = 'management' | 'usage';
type TrendTab = 'reliability' | 'daily-chart' | 'daily-table';
type UsageMetric = 'requests' | 'tokens';
type McpUsageTab = 'chart' | 'table';
type ManagementSortKey = 'user' | 'auth' | 'chats' | 'workspaces' | 'memberships' | 'workspaceChats' | 'liveInterrupted' | 'storage' | 'role' | 'actions';
type UsageSortKey = 'user' | 'requests' | 'input' | 'output' | 'total' | 'completed' | 'failed';
type ProviderSortKey = 'provider' | 'model' | 'source' | 'requests' | 'input' | 'output' | 'total';
type DailySortKey = 'date' | 'requests' | 'input' | 'output' | 'total' | 'completed' | 'failed' | 'mcpRequests' | 'mcpErrors' | 'apiRequests' | 'apiErrors';
type McpUserSortKey = 'user' | 'auth' | 'route' | 'requests' | 'success' | 'errors';

interface DailyCombinedRow extends DailyUsageTrend {
  mcp_requests: number;
  mcp_errors: number;
  api_requests: number;
  api_errors: number;
}

const ALL_DAY_RANGES = [7, 30, 90, 180, 240, 360] as const;

interface UsersPanelProps {
  currentUser: User | null;
  onOpenWorkspace: (workspaceId: string) => void;
  onOpenChat: (conversationId: string) => void;
}

type ExpandedUserDetailMode = 'workspaces' | 'chats';

interface DerivedUserStats {
  user: User;
  usage: UserUsageSummary | null;
  ownedWorkspaceCount: number;
  memberWorkspaceCount: number;
  workspaceConversationCount: number;
  workspaceMemberSlots: number;
  liveWorkspaceCount: number;
  interruptedWorkspaceCount: number;
  ownedWorkspaces: UserSpaceWorkspace[];
  storageBytesKnown: number;
  storageCoverage: number;
}

interface UsageDataSnapshot {
  usageSummary: UserUsageSummary[];
  providerBreakdown: ProviderModelBreakdown[];
  dailyTrend: DailyUsageTrend[];
  apiDaily: ApiDailyTrend[];
  userDailySeries: UserDailyUsageSeriesPoint[];
  mcpUsers: McpUserUsage[];
  mcpDaily: McpDailyTrend[];
  mcpRoutes: McpRouteUsage[];
}

function isWorkspaceDeleteTaskTerminal(task: UserSpaceWorkspaceDeleteTask): boolean {
  return task.phase === 'completed' || task.phase === 'failed';
}

interface PagerProps {
  page: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

function TablePager({ page, totalPages, totalItems, pageSize, onPageChange, onPageSizeChange }: PagerProps) {
  return (
    <div className="users-pager">
      <span className="users-pager-meta">{totalItems} items</span>
      <label className="users-pager-size">
        <span>Rows</span>
        <select value={pageSize} onChange={(e) => onPageSizeChange(Number(e.target.value))}>
          <option value={5}>5</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={50}>50</option>
        </select>
      </label>
      <button type="button" className="users-page-btn" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        Prev
      </button>
      <span className="users-page-indicator">{page}/{totalPages}</span>
      <button type="button" className="users-page-btn" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
        Next
      </button>
    </div>
  );
}

interface UserEditModalProps {
  user: User;
  authGroups: AuthGroup[];
  actionLoading: string | null;
  onRoleChange: (userId: string, role: 'admin' | 'user') => Promise<void>;
  onResetRoleOverride: (userId: string) => Promise<void>;
  onLocalGroupsChange: (userId: string, groupIds: string[]) => Promise<void>;
  onClose: () => void;
}

function UserEditModal({
  user,
  authGroups,
  actionLoading,
  onRoleChange,
  onResetRoleOverride,
  onLocalGroupsChange,
  onClose,
}: UserEditModalProps) {
  const internalAuthGroups = authGroups.filter(isInternalAuthGroup);
  const manualGroupIds = getUserManualGroupIds(user);
  const ldapGroupIds = new Set(user.ldap_group_ids ?? []);
  const ldapAuthGroups = authGroups.filter((group) => group.provider === 'ldap');
  const ldapGroups = ldapAuthGroups.filter((group) => ldapGroupIds.has(group.id));
  const ldapGroupDns = new Set(ldapGroups.map((group) => group.source_dn).filter(Boolean));
  const groupOptions = [
    ...internalAuthGroups.map((group) => {
      const badges: string[] = [];
      if (group.is_logon_group) badges.push('Logon');
      if (group.role === 'admin') badges.push('Admin');
      return {
        id: group.id,
        label: group.display_name,
        badge: badges.length ? badges : undefined,
      };
    }),
    ...ldapAuthGroups.map((group) => {
      const badges: string[] = ['LDAP'];
      if (group.is_logon_group) badges.push('Logon');
      if (group.role === 'admin') badges.push('Admin');
      return {
        id: group.id,
        label: group.display_name,
        badge: badges,
        disabled: true,
        checked: ldapGroupIds.has(group.id),
      };
    }),
    ...(user.cached_groups ?? [])
      .filter((groupDn) => !ldapGroupDns.has(groupDn))
      .map((groupDn) => ({
        id: `cached-ldap:${groupDn}`,
        label: formatGroupIdentifierForDisplay(groupDn),
        badge: 'LDAP',
        disabled: true,
        checked: true,
      })),
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Override - {user.display_name || user.username}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="form-group">
            <label>Role</label>
            <select
              value={user.role}
              disabled={actionLoading === user.id}
              onChange={(e) => void onRoleChange(user.id, e.target.value as 'admin' | 'user')}
            >
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
            {user.role_manually_set && (
              <div className="users-role-override-row" style={{ marginTop: 6 }}>
                <span className="users-role-override-badge">Role overridden.</span>
                <button
                  type="button"
                  className="users-role-reset-btn"
                  disabled={actionLoading === user.id}
                  onClick={() => void onResetRoleOverride(user.id)}
                >
                  Reset to default
                </button>
              </div>
            )}
          </div>
          {groupOptions.length > 0 ? (
            <div className="form-group">
              <label>Group Memberships</label>
              <CheckboxDropdown
                options={groupOptions}
                selectedIds={manualGroupIds}
                onChange={(ids) => void onLocalGroupsChange(user.id, ids)}
                placeholder="No manual groups assigned"
                searchPlaceholder="Search Group Memberships..."
                disabled={actionLoading === user.id}
              />
              <p className="field-help">Internal groups can be assigned manually. LDAP groups are read-only here and stay controlled by directory sync.</p>
            </div>
          ) : user.auth_provider === 'ldap' && (
            <div className="form-group">
              <label>Group Memberships</label>
              <p className="field-help">No LDAP group memberships are currently cached for this user.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function UsersPanel({ currentUser, onOpenWorkspace, onOpenChat }: UsersPanelProps) {
  const [activeTab, setActiveTab] = useState<PanelTab>('management');

  const [users, setUsers] = useState<User[]>([]);
  const [authGroups, setAuthGroups] = useState<AuthGroup[]>([]);
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspaceStateById, setWorkspaceStateById] = useState<Record<string, WorkspaceConversationStateSummaryItem>>({});
  const [deletingWorkspaceTasks, setDeletingWorkspaceTasks] = useState<Record<string, UserSpaceWorkspaceDeleteTask>>({});

  const [loading, setLoading] = useState(true);
  const [toasts, toast] = useToast();

  const [editingUserId, setEditingUserId] = useState<string | null>(null);

  const [usageSummary, setUsageSummary] = useState<UserUsageSummary[]>([]);
  const [providerBreakdown, setProviderBreakdown] = useState<ProviderModelBreakdown[]>([]);
  const [dailyTrend, setDailyTrend] = useState<DailyUsageTrend[]>([]);
  const [apiDaily, setApiDaily] = useState<ApiDailyTrend[]>([]);
  const [userDailySeries, setUserDailySeries] = useState<UserDailyUsageSeriesPoint[]>([]);
  const [mcpUsers, setMcpUsers] = useState<McpUserUsage[]>([]);
  const [mcpDaily, setMcpDaily] = useState<McpDailyTrend[]>([]);
  const [, setMcpRoutes] = useState<McpRouteUsage[]>([]);
  const [days, setDays] = useState(30);
  const [earliestDate, setEarliestDate] = useState<string | null>(null);
  const [usageLoadedDays, setUsageLoadedDays] = useState<number | null>(null);
  const [hasLoadedUsageRange, setHasLoadedUsageRange] = useState(false);
  const [usageLoading, setUsageLoading] = useState(false);

  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showCreateLocalUserModal, setShowCreateLocalUserModal] = useState(false);
  const [showManageAuthGroupsModal, setShowManageAuthGroupsModal] = useState(false);

  const [trendTab, setTrendTab] = useState<TrendTab>('reliability');
  const [mcpUsageTab, setMcpUsageTab] = useState<McpUsageTab>('chart');
  const [providerChartMetric, setProviderChartMetric] = useState<UsageMetric>('tokens');
  const [perUserChartMetric, setPerUserChartMetric] = useState<UsageMetric>('requests');

  const [managementPage, setManagementPage] = useState(1);
  const [usagePage, setUsagePage] = useState(1);
  const [providerPage, setProviderPage] = useState(1);
  const [dailyPage, setDailyPage] = useState(1);
  const [mcpUsersPage, setMcpUsersPage] = useState(1);

  const [managementPageSize, setManagementPageSize] = useState(10);
  const [usagePageSize, setUsagePageSize] = useState(10);
  const [providerPageSize, setProviderPageSize] = useState(10);
  const [dailyPageSize, setDailyPageSize] = useState(10);
  const [mcpUsersPageSize, setMcpUsersPageSize] = useState(10);

  const [managementSort, setManagementSort] = useState<TableSortConfig<ManagementSortKey>>({ key: 'chats', direction: 'desc' });
  const [usageSort, setUsageSort] = useState<TableSortConfig<UsageSortKey>>({ key: 'requests', direction: 'desc' });
  const [providerSort, setProviderSort] = useState<TableSortConfig<ProviderSortKey>>({ key: 'total', direction: 'desc' });
  const [dailySort, setDailySort] = useState<TableSortConfig<DailySortKey>>({ key: 'date', direction: 'desc' });
  const [mcpUserSort, setMcpUserSort] = useState<TableSortConfig<McpUserSortKey>>({ key: 'requests', direction: 'desc' });

  const [expandedUserDetail, setExpandedUserDetail] = useState<{ userId: string; mode: ExpandedUserDetailMode } | null>(null);
  const [standaloneChatsByUserId, setStandaloneChatsByUserId] = useState<Record<string, Conversation[]>>({});
  const [standaloneChatsLoaded, setStandaloneChatsLoaded] = useState(false);
  const [standaloneChatsLoadingUserId, setStandaloneChatsLoadingUserId] = useState<string | null>(null);
  const [workspaceLastMessageAtById, setWorkspaceLastMessageAtById] = useState<Record<string, string | null>>({});
  const [workspaceLastConversationById, setWorkspaceLastConversationById] = useState<Record<string, Conversation | null>>({});
  const [workspaceLastMessageLoadingByUserId, setWorkspaceLastMessageLoadingByUserId] = useState<Record<string, boolean>>({});
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);

  const themeColors = useThemeColors();
  const usageCacheRef = useRef<Record<number, UsageDataSnapshot>>({});
  const usageRequestIdRef = useRef(0);
  const workspaceStateRequestIdRef = useRef(0);

  const [storageByWorkspaceId, setStorageByWorkspaceId] = useState<Record<string, number>>({});
  const [storageLoadingByUserId, setStorageLoadingByUserId] = useState<Record<string, boolean>>({});

  const loadUsers = useCallback(async () => {
    const res = await api.listUsers();
    setUsers(res);
  }, []);

  const loadAuthGroups = useCallback(async () => {
    const groups = await api.listAuthGroups();
    setAuthGroups(groups);
  }, []);

  const fetchUsageDetails = useCallback(async (d: number): Promise<Pick<UsageDataSnapshot, 'userDailySeries'>> => {
    const userDailyRes = await api.getUsageUsersDaily(d);

    return {
      userDailySeries: userDailyRes.series,
    };
  }, []);

  const applyUsageSnapshot = useCallback((snapshot: UsageDataSnapshot) => {
    setUsageSummary(snapshot.usageSummary);
    setProviderBreakdown(snapshot.providerBreakdown);
    setDailyTrend(snapshot.dailyTrend);
    setApiDaily(snapshot.apiDaily);
    setUserDailySeries(snapshot.userDailySeries);
    setMcpUsers(snapshot.mcpUsers);
    setMcpDaily(snapshot.mcpDaily);
    setMcpRoutes(snapshot.mcpRoutes);
  }, []);

  const loadWorkspaceStateSummary = useCallback(async (workspaceIds: string[]) => {
    const requestId = workspaceStateRequestIdRef.current + 1;
    workspaceStateRequestIdRef.current = requestId;

    try {
      const summary = await api.getWorkspacesConversationStateSummaryLite(workspaceIds);
      if (requestId !== workspaceStateRequestIdRef.current) return;

      const byId = summary.reduce<Record<string, WorkspaceConversationStateSummaryItem>>((acc, item) => {
        acc[item.workspace_id] = item;
        return acc;
      }, {});
      setWorkspaceStateById(byId);
    } catch (err) {
      // Keep management table responsive even if state summaries fail.
      console.warn('Failed to load workspace state summary:', err);
    }
  }, []);

  const loadWorkspaces = useCallback(async () => {
    const all: UserSpaceWorkspace[] = [];
    let offset = 0;
    const limit = 50;

    while (true) {
      const page = await api.listUserSpaceWorkspaces(offset, limit, true);
      all.push(...page.items);
      offset += page.items.length;
      if (all.length >= page.total || page.items.length === 0) {
        break;
      }
    }

    setWorkspaces(all);

    if (all.length === 0) {
      setWorkspaceStateById({});
      return;
    }

    void loadWorkspaceStateSummary(all.map((w) => w.id));
  }, [loadWorkspaceStateSummary]);

  const loadManagementData = useCallback(async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadUsers(),
        loadAuthGroups(),
        loadWorkspaces(),
      ]);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to load users data');
    } finally {
      setLoading(false);
    }
  }, [loadAuthGroups, loadUsers, loadWorkspaces]);

  const loadUsageData = useCallback(async (d: number) => {
    const cached = usageCacheRef.current[d];
    if (cached) {
      applyUsageSnapshot(cached);
      setUsageLoadedDays(d);
      return;
    }

    const requestId = usageRequestIdRef.current + 1;
    usageRequestIdRef.current = requestId;

    setUsageLoading(true);
    try {
      const [summaryRes, providersRes, dailyRes, apiRes, mcpRes, rangeRes] = await Promise.all([
        api.getUsageSummary(d),
        api.getUsageProviders(d),
        api.getUsageDaily(d),
        api.getUsageApi(d),
        api.getUsageMcp(d),
        hasLoadedUsageRange ? Promise.resolve(null) : api.getUsageRange(),
      ]);

      if (requestId !== usageRequestIdRef.current) return;

      // Show key usage cards/charts first, then enrich with heavier series calls.
      setUsageSummary(summaryRes.users);
      setProviderBreakdown(providersRes.providers);
      setDailyTrend(dailyRes.daily);
      setApiDaily(apiRes.daily);
      setUserDailySeries([]);
      setMcpUsers(mcpRes.users);
      setMcpDaily(mcpRes.daily);
      setMcpRoutes(mcpRes.routes);

      if (rangeRes) {
        setEarliestDate(rangeRes.earliest_date);
        setHasLoadedUsageRange(true);
      }
      setUsageLoadedDays(d);

      const details = await fetchUsageDetails(d);
      if (requestId !== usageRequestIdRef.current) return;

      const snapshot: UsageDataSnapshot = {
        usageSummary: summaryRes.users,
        providerBreakdown: providersRes.providers,
        dailyTrend: dailyRes.daily,
        apiDaily: apiRes.daily,
        userDailySeries: details.userDailySeries,
        mcpUsers: mcpRes.users,
        mcpDaily: mcpRes.daily,
        mcpRoutes: mcpRes.routes,
      };

      applyUsageSnapshot(snapshot);
      usageCacheRef.current[d] = snapshot;
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to load usage data');
    } finally {
      if (requestId === usageRequestIdRef.current) {
        setUsageLoading(false);
      }
    }
  }, [applyUsageSnapshot, fetchUsageDetails, hasLoadedUsageRange]);

  useEffect(() => {
    void loadManagementData();
  }, [loadManagementData]);

  // Poll workspace live/interrupted state while management tab is visible
  useEffect(() => {
    if (activeTab !== 'management') return;
    if (workspaces.length === 0) return;

    const ids = workspaces.map((w) => w.id);
    const handle = setInterval(() => {
      void loadWorkspaceStateSummary(ids);
    }, 10_000);

    return () => clearInterval(handle);
  }, [activeTab, workspaces, loadWorkspaceStateSummary]);

  useEffect(() => {
    if (activeTab !== 'usage') return;
    if (usageLoadedDays === days && hasLoadedUsageRange) return;
    void loadUsageData(days);
  }, [activeTab, days, hasLoadedUsageRange, loadUsageData, usageLoadedDays]);

  const handleRoleChange = async (userId: string, newRole: 'admin' | 'user') => {
    setActionLoading(userId);
    try {
      await api.updateUserRole(userId, newRole);
      setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, role: newRole, role_manually_set: true } : u)));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to update role');
    } finally {
      setActionLoading(null);
    }
  };

  const handleResetRoleOverride = async (userId: string) => {
    setActionLoading(userId);
    try {
      const response = await api.resetUserRoleOverride(userId);
      setUsers((prev) => prev.map((u) => (
        u.id === userId ? { ...u, role: response.role, role_manually_set: response.role_manually_set } : u
      )));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to reset role override');
    } finally {
      setActionLoading(null);
    }
  };

  const handleLocalGroupsChange = async (userId: string, groupIds: string[]) => {
    setActionLoading(userId);
    try {
      const updated = await api.setUserGroups(userId, { group_ids: groupIds });
      setUsers((prev) => prev.map((u) => (u.id === userId ? updated : u)));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to update user groups');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (userId: string) => {
    setActionLoading(userId);
    try {
      await api.deleteUser(userId);
      setUsers((prev) => prev.filter((u) => u.id !== userId));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete user');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteWorkspace = async (workspaceId: string) => {
    try {
      const task = await api.queueUserSpaceWorkspaceDelete(workspaceId);
      setDeletingWorkspaceTasks((prev) => ({
        ...prev,
        [workspaceId]: task,
      }));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to queue workspace delete');
      throw e;
    }
  };

  useEffect(() => {
    const tasks = Object.values(deletingWorkspaceTasks);
    if (tasks.length === 0) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollDeleteTasks = async () => {
      if (pollInFlight) {
        return;
      }
      pollInFlight = true;

      try {
        const results = await Promise.all(tasks.map(async (task) => {
          try {
            const status = await api.getUserSpaceWorkspaceDeleteTask(task.task_id);
            return { task, status, error: null as Error | null };
          } catch (error) {
            return { task, status: null as UserSpaceWorkspaceDeleteTask | null, error: error as Error };
          }
        }));

        if (cancelled) {
          return;
        }

        const completedWorkspaceIds = new Set<string>();
        const terminalWorkspaceIds = new Set<string>();
        const updatedTasks: Record<string, UserSpaceWorkspaceDeleteTask> = {};
        let nextError: string | null = null;

        for (const result of results) {
          if (result.status) {
            if (isWorkspaceDeleteTaskTerminal(result.status)) {
              terminalWorkspaceIds.add(result.status.workspace_id);
              if (result.status.phase === 'completed') {
                completedWorkspaceIds.add(result.status.workspace_id);
              } else if (!nextError) {
                nextError = result.status.error?.trim() || `Failed to delete ${result.status.workspace_name}`;
              }
            } else {
              updatedTasks[result.status.workspace_id] = result.status;
            }
            continue;
          }

          if (result.error instanceof ApiError && result.error.status === 404) {
            terminalWorkspaceIds.add(result.task.workspace_id);
            completedWorkspaceIds.add(result.task.workspace_id);
            continue;
          }

          if (!nextError && result.error instanceof Error) {
            nextError = result.error.message;
          }
        }

        setDeletingWorkspaceTasks((prev) => {
          const next = { ...prev };
          for (const workspaceId of terminalWorkspaceIds) {
            delete next[workspaceId];
          }
          for (const [workspaceId, task] of Object.entries(updatedTasks)) {
            next[workspaceId] = task;
          }
          return next;
        });

        if (completedWorkspaceIds.size > 0) {
          setWorkspaces((prev) => prev.filter((workspace) => !completedWorkspaceIds.has(workspace.id)));
          setWorkspaceStateById((prev) => {
            const next = { ...prev };
            for (const workspaceId of completedWorkspaceIds) {
              delete next[workspaceId];
            }
            return next;
          });
          setStorageByWorkspaceId((prev) => {
            const next = { ...prev };
            for (const workspaceId of completedWorkspaceIds) {
              delete next[workspaceId];
            }
            return next;
          });
        }

        if (nextError) {
          toast.error(nextError);
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollDeleteTasks();
    const intervalId = window.setInterval(() => {
      void pollDeleteTasks();
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [deletingWorkspaceTasks, toast]);

  const handleTransferWorkspace = async (workspaceId: string, newOwnerId: string) => {
    try {
      const updated = await api.updateUserSpaceWorkspace(workspaceId, { owner_user_id: newOwnerId });
      setWorkspaces((prev) => prev.map((ws) => (ws.id === workspaceId ? updated : ws)));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to transfer workspace ownership');
      throw e;
    }
  };

  const handleComputeStorageForUser = async (userId: string) => {
    const ownedIds = workspaces.filter((w) => w.owner_user_id === userId).map((w) => w.id);
    const missing = ownedIds.filter((id) => storageByWorkspaceId[id] === undefined);
    if (missing.length === 0) return;

    setStorageLoadingByUserId((prev) => ({ ...prev, [userId]: true }));
    try {
      const settled = await Promise.allSettled(
        missing.map(async (workspaceId) => {
          const entries = await api.listUserSpaceFiles(workspaceId);
          const total = entries.reduce((sum, entry) => sum + (entry.size_bytes || 0), 0);
          return { workspaceId, total };
        })
      );

      const updates: Record<string, number> = {};
      for (const item of settled) {
        if (item.status === 'fulfilled') {
          updates[item.value.workspaceId] = item.value.total;
        }
      }
      setStorageByWorkspaceId((prev) => ({ ...prev, ...updates }));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to compute workspace storage usage');
    } finally {
      setStorageLoadingByUserId((prev) => ({ ...prev, [userId]: false }));
    }
  };

  const usageByUserId = useMemo(() => {
    return usageSummary.reduce<Record<string, UserUsageSummary>>((acc, row) => {
      acc[row.user_id] = row;
      return acc;
    }, {});
  }, [usageSummary]);

  const deletingWorkspaceIds = useMemo(
    () => new Set(Object.keys(deletingWorkspaceTasks)),
    [deletingWorkspaceTasks],
  );

  const workspaceConversationIdSet = useMemo(() => (
    new Set(workspaces.flatMap((workspace) => workspace.conversation_ids))
  ), [workspaces]);

  useEffect(() => {
    setStandaloneChatsByUserId({});
    setStandaloneChatsLoaded(false);
    setStandaloneChatsLoadingUserId(null);
  }, [workspaceConversationIdSet]);

  const loadStandaloneChatsSnapshot = useCallback(async () => {
    if (standaloneChatsLoaded) {
      return;
    }

    const allConversations = await api.listConversations();
    const grouped = allConversations.reduce<Record<string, Conversation[]>>((acc, conversation) => {
      const linkedWorkspaceId = conversation.workspace_id ?? conversation.workspaceId ?? null;
      if (linkedWorkspaceId) {
        return acc;
      }
      if (workspaceConversationIdSet.has(conversation.id)) {
        return acc;
      }
      if (!conversation.user_id) {
        return acc;
      }

      if (!acc[conversation.user_id]) {
        acc[conversation.user_id] = [];
      }
      acc[conversation.user_id].push(conversation);
      return acc;
    }, {});

    for (const list of Object.values(grouped)) {
      list.sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at));
    }

    setStandaloneChatsByUserId(grouped);
    setStandaloneChatsLoaded(true);
  }, [standaloneChatsLoaded, workspaceConversationIdSet]);

  const loadStandaloneChatsForUser = useCallback(async (userId: string) => {
    if (standaloneChatsByUserId[userId]) {
      return;
    }

    setStandaloneChatsLoadingUserId(userId);
    try {
      await loadStandaloneChatsSnapshot();
      setStandaloneChatsByUserId((prev) => ({
        ...prev,
        [userId]: prev[userId] ?? [],
      }));
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to load user chats');
    } finally {
      setStandaloneChatsLoadingUserId((prev) => (prev === userId ? null : prev));
    }
  }, [loadStandaloneChatsSnapshot, standaloneChatsByUserId, toast]);

  useEffect(() => {
    if (activeTab !== 'management') {
      return;
    }
    if (standaloneChatsLoaded) {
      return;
    }
    void loadStandaloneChatsSnapshot().catch((e: unknown) => {
      toast.error(e instanceof Error ? e.message : 'Failed to load user chats');
    });
  }, [activeTab, loadStandaloneChatsSnapshot, standaloneChatsLoaded, toast]);

  const standaloneChatCountsByUserId = useMemo(() => (
    Object.entries(standaloneChatsByUserId).reduce<Record<string, number>>((acc, [userId, conversations]) => {
      acc[userId] = conversations.length;
      return acc;
    }, {})
  ), [standaloneChatsByUserId]);

  const toggleUserWorkspaceDetails = useCallback((userId: string) => {
    setExpandedUserDetail((prev) => (
      prev?.userId === userId && prev.mode === 'workspaces'
        ? null
        : { userId, mode: 'workspaces' }
    ));
  }, []);

  useEffect(() => {
    setWorkspaceLastMessageAtById((prev) => {
      const knownWorkspaceIds = new Set(workspaces.map((workspace) => workspace.id));
      const next: Record<string, string | null> = {};
      let changed = false;
      for (const [workspaceId, timestamp] of Object.entries(prev)) {
        if (knownWorkspaceIds.has(workspaceId)) {
          next[workspaceId] = timestamp;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [workspaces]);

  useEffect(() => {
    setWorkspaceLastConversationById((prev) => {
      const knownWorkspaceIds = new Set(workspaces.map((workspace) => workspace.id));
      const next: Record<string, Conversation | null> = {};
      let changed = false;
      for (const [workspaceId, conversation] of Object.entries(prev)) {
        if (knownWorkspaceIds.has(workspaceId)) {
          next[workspaceId] = conversation;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [workspaces]);

  useEffect(() => {
    let cancelled = false;

    const loadAvailableModels = async () => {
      try {
        const response = await api.getAvailableModels();
        if (cancelled) return;
        setAvailableModels(response.models || []);
      } catch {
        // Context metadata can fall back without blocking Users panel.
      }
    };

    void loadAvailableModels();

    return () => {
      cancelled = true;
    };
  }, []);

  const defaultContextLimit = useMemo(() => {
    const maxAvailable = availableModels.reduce((max, model) => {
      const limit = Number(model.context_limit || 0);
      return Number.isFinite(limit) ? Math.max(max, limit) : max;
    }, 0);
    return Math.max(8192, maxAvailable);
  }, [availableModels]);

  const getConversationContextMeta = useCallback((conversation: Conversation | null | undefined): string => {
    if (!conversation) return 'n/a';

    const contextLimit = resolveConversationContextLimit(conversation.model, availableModels, defaultContextLimit);
    const usage = calculateConversationContextUsage({
      messages: conversation.messages,
      persistedConversationTokens: conversation.total_tokens,
      contextLimit,
    });

    return `${formatNumber(usage.totalTokens)} / ${formatNumber(contextLimit)}`;
  }, [availableModels, defaultContextLimit]);

  const loadWorkspaceLastMessageTimesForUser = useCallback(async (userId: string) => {
    const owned = workspaces.filter((workspace) => workspace.owner_user_id === userId);
    const missing = owned
      .map((workspace) => workspace.id)
      .filter((workspaceId) => !(workspaceId in workspaceLastMessageAtById));

    if (missing.length === 0) {
      return;
    }

    setWorkspaceLastMessageLoadingByUserId((prev) => ({ ...prev, [userId]: true }));
    try {
      const settled = await Promise.allSettled(missing.map(async (workspaceId) => {
        const conversations = await api.listConversations(workspaceId);
        const latestConversation = conversations.reduce<Conversation | null>((latest, conversation) => {
          if (!latest) return conversation;
          return toEpochMs(conversation.updated_at) > toEpochMs(latest.updated_at)
            ? conversation
            : latest;
        }, null);
        return {
          workspaceId,
          lastUpdatedAt: latestConversation?.updated_at ?? null,
          latestConversation,
        };
      }));

      const updates: Record<string, string | null> = {};
      const lastConversationUpdates: Record<string, Conversation | null> = {};
      for (const item of settled) {
        if (item.status === 'fulfilled') {
          updates[item.value.workspaceId] = item.value.lastUpdatedAt;
          lastConversationUpdates[item.value.workspaceId] = item.value.latestConversation;
        }
      }

      if (Object.keys(updates).length > 0) {
        setWorkspaceLastMessageAtById((prev) => ({ ...prev, ...updates }));
        setWorkspaceLastConversationById((prev) => ({ ...prev, ...lastConversationUpdates }));
      }
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to load workspace message timestamps');
    } finally {
      setWorkspaceLastMessageLoadingByUserId((prev) => ({ ...prev, [userId]: false }));
    }
  }, [toast, workspaceLastMessageAtById, workspaces]);

  const toggleUserChatDetails = useCallback(async (userId: string) => {
    const isAlreadyExpanded = expandedUserDetail?.userId === userId && expandedUserDetail.mode === 'chats';
    if (isAlreadyExpanded) {
      setExpandedUserDetail(null);
      return;
    }

    setExpandedUserDetail({ userId, mode: 'chats' });
    if (!standaloneChatsByUserId[userId]) {
      await loadStandaloneChatsForUser(userId);
    }
  }, [expandedUserDetail, loadStandaloneChatsForUser, standaloneChatsByUserId]);

  useEffect(() => {
    if (!expandedUserDetail || expandedUserDetail.mode !== 'workspaces') {
      return;
    }
    void loadWorkspaceLastMessageTimesForUser(expandedUserDetail.userId);
  }, [expandedUserDetail, loadWorkspaceLastMessageTimesForUser]);

  const handleDeleteConversation = useCallback(async (conversationId: string) => {
    try {
      await api.deleteConversation(conversationId);
      setStandaloneChatsByUserId((prev) => {
        const next: Record<string, Conversation[]> = {};
        for (const [userId, conversations] of Object.entries(prev)) {
          next[userId] = conversations.filter((conversation) => conversation.id !== conversationId);
        }
        return next;
      });
      toast.success('Chat deleted');
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete chat');
      throw e;
    }
  }, [toast]);

  const handleCancelConversationTask = useCallback(async (conversationId: string, taskId: string) => {
    try {
      await api.cancelChatTask(taskId);
      setStandaloneChatsByUserId((prev) => {
        const next: Record<string, Conversation[]> = {};
        for (const [userId, conversations] of Object.entries(prev)) {
          next[userId] = conversations.map((conversation) => (
            conversation.id === conversationId
              ? { ...conversation, active_task_id: null }
              : conversation
          ));
        }
        return next;
      });
      toast.success('Chat task cancellation requested');
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Failed to cancel chat task');
      throw e;
    }
  }, [toast]);

  const userStatsRows = useMemo<DerivedUserStats[]>(() => {
    const byUserOwned = workspaces.reduce<Record<string, UserSpaceWorkspace[]>>((acc, ws) => {
      if (!acc[ws.owner_user_id]) acc[ws.owner_user_id] = [];
      acc[ws.owner_user_id].push(ws);
      return acc;
    }, {});

    const byUserMember = workspaces.reduce<Record<string, number>>((acc, ws) => {
      for (const member of ws.members) {
        acc[member.user_id] = (acc[member.user_id] || 0) + 1;
      }
      return acc;
    }, {});

    return users.map((user) => {
      const owned = byUserOwned[user.id] || [];
      const usage = usageByUserId[user.id] || null;

      let workspaceConversationCount = 0;
      let workspaceMemberSlots = 0;
      let liveWorkspaceCount = 0;
      let interruptedWorkspaceCount = 0;
      let storageBytesKnown = 0;
      let storageKnownCount = 0;

      for (const ws of owned) {
        workspaceConversationCount += ws.conversation_ids.length;
        workspaceMemberSlots += ws.members.length;

        const state = workspaceStateById[ws.id];
        if (state?.has_live_task) liveWorkspaceCount += 1;
        if (state?.has_interrupted_task) interruptedWorkspaceCount += 1;

        const size = storageByWorkspaceId[ws.id];
        if (typeof size === 'number') {
          storageBytesKnown += size;
          storageKnownCount += 1;
        }
      }

      return {
        user,
        usage,
        ownedWorkspaceCount: owned.length,
        memberWorkspaceCount: byUserMember[user.id] || 0,
        workspaceConversationCount,
        workspaceMemberSlots,
        liveWorkspaceCount,
        interruptedWorkspaceCount,
        ownedWorkspaces: owned,
        storageBytesKnown,
        storageCoverage: owned.length === 0 ? 1 : storageKnownCount / owned.length,
      };
    }).sort((a, b) => {
      const aChats = a.usage?.total_requests || 0;
      const bChats = b.usage?.total_requests || 0;
      if (bChats !== aChats) return bChats - aChats;
      return a.user.username.localeCompare(b.user.username);
    });
  }, [users, usageByUserId, workspaces, workspaceStateById, storageByWorkspaceId]);

  const availableRanges = useMemo(() => {
    if (!earliestDate) return [7, 30, 90];
    const earliest = new Date(earliestDate + 'T00:00:00Z');
    const now = new Date();
    const totalDays = Math.ceil((now.getTime() - earliest.getTime()) / (1000 * 60 * 60 * 24));
    return ALL_DAY_RANGES.filter((d) => d <= totalDays + 1);
  }, [earliestDate]);

  const dateLabels = useMemo(() => buildDateLabels(days), [days]);

  const dailyTrendRows = useMemo<DailyUsageTrend[]>(() => {
    const byDate = dailyTrend.reduce<Record<string, DailyUsageTrend>>((acc, row) => {
      const date = normalizeDateLabel(row.date);
      acc[date] = { ...row, date };
      return acc;
    }, {});

    return dateLabels.map((date) => byDate[date] ?? {
      date,
      total_requests: 0,
      total_input_tokens: 0,
      total_output_tokens: 0,
      total_tokens: 0,
      completed_count: 0,
      failed_count: 0,
    });
  }, [dailyTrend, dateLabels]);

  const compareSortValues = (a: string | number | null, b: string | number | null, direction: 'asc' | 'desc') => {
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;

    if (typeof a === 'number' && typeof b === 'number') {
      if (a < b) return direction === 'asc' ? -1 : 1;
      if (a > b) return direction === 'asc' ? 1 : -1;
      return 0;
    }

    const aStr = String(a).toLowerCase();
    const bStr = String(b).toLowerCase();
    if (aStr < bStr) return direction === 'asc' ? -1 : 1;
    if (aStr > bStr) return direction === 'asc' ? 1 : -1;
    return 0;
  };

  const providerChartData = useMemo(() => {
    const grouped = new Map<string, {
      provider: string;
      model: string;
      bySource: Record<string, { requests: number; tokens: number }>;
      total: number;
    }>();

    for (const row of providerBreakdown) {
      const key = `${row.provider}::${row.model}`;
      let current = grouped.get(key);
      if (!current) {
        current = {
          provider: row.provider,
          model: row.model,
          bySource: {},
          total: 0,
        };
        grouped.set(key, current);
      }

      const sourceKey = row.request_source || 'ui';
      if (!current.bySource[sourceKey]) {
        current.bySource[sourceKey] = { requests: 0, tokens: 0 };
      }
      current.bySource[sourceKey].requests += row.total_requests;
      current.bySource[sourceKey].tokens += row.total_tokens;
      current.total += providerChartMetric === 'requests' ? row.total_requests : row.total_tokens;
    }

    const groupedRows = [...grouped.values()].sort((a, b) => b.total - a.total);
    const labels = groupedRows.map((row) => {
      const providerLabel = formatProviderDisplayName(row.provider);
      const modelLabel = formatModelDisplayName(row.model, row.provider);
      const label = `${providerLabel} / ${modelLabel}`;
      return label.length > 36 ? label.slice(0, 35) + '\u2026' : label;
    });

    const sourceOrder = ['ui', 'api'];
    const sources = [...new Set(providerBreakdown.map((row) => row.request_source || 'ui'))].sort((a, b) => {
      const ai = sourceOrder.indexOf(a);
      const bi = sourceOrder.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });

    const labelForSource = (source: string) => {
      if (source === 'ui') return 'Ragtime Chat';
      if (source === 'api') return 'API';
      return source;
    };

    const datasets = sources.map((source, idx) => ({
      label: labelForSource(source),
      data: groupedRows.map((row) => {
        const bucket = row.bySource[source];
        if (!bucket) return 0;
        return providerChartMetric === 'requests' ? bucket.requests : bucket.tokens;
      }),
      backgroundColor: CHART_PALETTE[idx % CHART_PALETTE.length],
      borderRadius: 4,
      borderSkipped: false,
      stack: 'source',
      maxBarThickness: 22,
    }));

    return { labels, datasets };
  }, [providerBreakdown, providerChartMetric]);

  const perUserChartData = useMemo(() => {
    const pointsByUser = userDailySeries.reduce<Record<string, Record<string, UserDailyUsageSeriesPoint>>>((acc, row) => {
      const date = normalizeDateLabel(row.date);
      if (!acc[row.user_id]) acc[row.user_id] = {};
      acc[row.user_id][date] = { ...row, date };
      return acc;
    }, {});

    const topUsers = [...usageSummary]
      .sort((a, b) => {
        const aValue = perUserChartMetric === 'requests' ? a.total_requests : a.total_tokens;
        const bValue = perUserChartMetric === 'requests' ? b.total_requests : b.total_tokens;
        if (bValue !== aValue) return bValue - aValue;
        return a.username.localeCompare(b.username);
      })
      .slice(0, 5);

    return {
      labels: dateLabels,
      datasets: topUsers.map((row, idx) => {
        const color = CHART_PALETTE[idx % CHART_PALETTE.length];
        return {
          label: (row.display_name || row.username).length > 16
            ? (row.display_name || row.username).slice(0, 15) + '\u2026'
            : (row.display_name || row.username),
          data: dateLabels.map((date) => {
            const point = pointsByUser[row.user_id]?.[date];
            return perUserChartMetric === 'requests' ? point?.total_requests || 0 : point?.total_tokens || 0;
          }),
          borderColor: color,
          backgroundColor: color,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.25,
          fill: false,
        };
      }),
    };
  }, [dateLabels, perUserChartMetric, usageSummary, userDailySeries]);

  const mcpDailyRows = useMemo<McpDailyTrend[]>(() => {
    const byDate = mcpDaily.reduce<Record<string, McpDailyTrend>>((acc, row) => {
      const date = normalizeDateLabel(row.date);
      acc[date] = { ...row, date };
      return acc;
    }, {});

    return dateLabels.map((date) => byDate[date] ?? {
      date,
      total_requests: 0,
      success_count: 0,
      error_count: 0,
      unique_users: 0,
    });
  }, [dateLabels, mcpDaily]);

  const apiDailyRows = useMemo<ApiDailyTrend[]>(() => {
    const byDate = apiDaily.reduce<Record<string, ApiDailyTrend>>((acc, row) => {
      const date = normalizeDateLabel(row.date);
      acc[date] = { ...row, date };
      return acc;
    }, {});

    return dateLabels.map((date) => byDate[date] ?? {
      date,
      total_requests: 0,
      success_count: 0,
      error_count: 0,
      unique_users: 0,
    });
  }, [apiDaily, dateLabels]);

  const dailyCombinedRows = useMemo<DailyCombinedRow[]>(() =>
    dailyTrendRows.map((row, i) => ({
      ...row,
      mcp_requests: mcpDailyRows[i]?.total_requests ?? 0,
      mcp_errors: mcpDailyRows[i]?.error_count ?? 0,
      api_requests: apiDailyRows[i]?.total_requests ?? 0,
      api_errors: apiDailyRows[i]?.error_count ?? 0,
    })),
  [apiDailyRows, dailyTrendRows, mcpDailyRows]);

  const dailyChartData = useMemo(() => ({
    labels: dailyTrendRows.map((row) => row.date),
    datasets: [
      {
        label: 'Chat Requests',
        data: dailyTrendRows.map((row) => row.total_requests),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.25)',
        yAxisID: 'yRequests',
        tension: 0.25,
      },
      {
        label: 'MCP / API Requests',
        data: mcpDailyRows.map((row) => row.total_requests),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.25)',
        yAxisID: 'yRequests',
        tension: 0.25,
      },
      {
        label: 'API Requests',
        data: apiDailyRows.map((row) => row.total_requests),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        yAxisID: 'yRequests',
        tension: 0.25,
      },
      {
        label: 'Total Tokens',
        data: dailyTrendRows.map((row) => row.total_tokens),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.25)',
        yAxisID: 'yTokens',
        tension: 0.25,
      },
    ],
  }), [apiDailyRows, dailyTrendRows, mcpDailyRows]);

  const singleAxisLineOptions = useMemo<ChartOptions<'line'>>(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: themeColors.text, boxWidth: 10, boxHeight: 10 },
      },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: {
        ticks: { color: themeColors.textSecondary, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: days >= 90 ? 12 : undefined },
        grid: { color: themeColors.grid },
      },
      y: {
        type: 'linear',
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
    },
  }), [days, themeColors]);

  const dailyLineOptions = useMemo<ChartOptions<'line'>>(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: themeColors.text, boxWidth: 10, boxHeight: 10 },
      },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: {
        ticks: { color: themeColors.textSecondary, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: days >= 90 ? 12 : undefined },
        grid: { color: themeColors.grid },
      },
      yRequests: {
        type: 'linear',
        position: 'left',
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
      yTokens: {
        type: 'linear',
        position: 'right',
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { drawOnChartArea: false },
      },
    },
  }), [days, themeColors]);

  const providerBarOptions = useMemo<ChartOptions<'bar'>>(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    indexAxis: 'y',
    plugins: {
      legend: { display: true, labels: { color: themeColors.text } },
      tooltip: { mode: 'nearest', intersect: false },
    },
    scales: {
      x: {
        beginAtZero: true,
        stacked: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
      y: {
        stacked: true,
        ticks: { color: themeColors.text, autoSkip: false },
        grid: { display: false },
      },
    },
  }), [themeColors]);

  const mcpChartData = useMemo(() => ({
    labels: mcpDailyRows.map((row) => row.date),
    datasets: [
      {
        label: 'MCP Success',
        data: mcpDailyRows.map((row) => row.success_count),
        backgroundColor: 'rgba(34, 197, 94, 0.75)',
        borderColor: '#22c55e',
        borderWidth: 1,
        stack: 'requests',
      },
      {
        label: 'MCP Error',
        data: mcpDailyRows.map((row) => row.error_count),
        backgroundColor: 'rgba(239, 68, 68, 0.75)',
        borderColor: '#ef4444',
        borderWidth: 1,
        stack: 'requests',
      },
      {
        label: 'API Success',
        data: apiDailyRows.map((row) => row.success_count),
        backgroundColor: 'rgba(59, 130, 246, 0.75)',
        borderColor: '#3b82f6',
        borderWidth: 1,
        stack: 'requests',
      },
      {
        label: 'API Error',
        data: apiDailyRows.map((row) => row.error_count),
        backgroundColor: 'rgba(245, 158, 11, 0.75)',
        borderColor: '#f59e0b',
        borderWidth: 1,
        stack: 'requests',
      },
    ],
  }), [apiDailyRows, mcpDailyRows]);

  const mcpChartOptions = useMemo<ChartOptions<'bar'>>(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: themeColors.text, boxWidth: 10, boxHeight: 10 },
      },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: {
        stacked: true,
        ticks: {
          color: themeColors.textSecondary,
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: days >= 90 ? 12 : undefined,
        },
        grid: { color: themeColors.grid },
      },
      y: {
        stacked: true,
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
    },
  }), [days, themeColors]);

  const reliabilityTrendData = useMemo(() => ({
    labels: dailyTrendRows.map((row) => row.date),
    datasets: [
      {
        label: 'Chat Success %',
        data: dailyTrendRows.map((row) => {
          const total = row.total_requests;
          return total > 0 ? (row.completed_count / total) * 100 : null;
        }),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 4,
        fill: true,
      },
      {
        label: 'MCP Success %',
        data: mcpDailyRows.map((row) => {
          const total = row.total_requests;
          return total > 0 ? (row.success_count / total) * 100 : null;
        }),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.08)',
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 4,
        fill: true,
      },
      {
        label: 'API Success %',
        data: apiDailyRows.map((row) => {
          const total = row.total_requests;
          return total > 0 ? (row.success_count / total) * 100 : null;
        }),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.08)',
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 4,
        fill: true,
      },
    ],
  }), [apiDailyRows, dailyTrendRows, mcpDailyRows]);

  const reliabilityTrendOptions = useMemo<ChartOptions<'line'>>(() => {
    const chatRates = dailyTrendRows
      .filter((row) => row.total_requests > 0)
      .map((row) => (row.completed_count / row.total_requests) * 100);
    const mcpRates = mcpDailyRows
      .filter((row) => row.total_requests > 0)
      .map((row) => (row.success_count / row.total_requests) * 100);
    const apiRates = apiDailyRows
      .filter((row) => row.total_requests > 0)
      .map((row) => (row.success_count / row.total_requests) * 100);
    const allRates = [...chatRates, ...mcpRates, ...apiRates];
    const minRate = allRates.length > 0 ? Math.min(...allRates) : 100;
    const dynamicMin = minRate >= 95 ? 90 : minRate >= 80 ? Math.floor(minRate / 10) * 10 : 0;

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: { color: themeColors.text, boxWidth: 10, boxHeight: 10 },
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${typeof ctx.raw === 'number' ? ctx.raw.toFixed(1) : '\u2014'}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: themeColors.textSecondary, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: days >= 90 ? 12 : undefined },
          grid: { color: themeColors.grid },
        },
        y: {
          type: 'linear',
          min: dynamicMin,
          max: 100,
          ticks: { color: themeColors.textSecondary, callback: (value) => `${value}%` },
          grid: { color: themeColors.grid },
        },
      },
    };
  }, [apiDailyRows, days, dailyTrendRows, mcpDailyRows, themeColors]);

  const paretoChartData = useMemo(() => {
    const sorted = [...usageSummary].sort((a, b) => b.total_tokens - a.total_tokens);
    const totalTokensAll = sorted.reduce((s, u) => s + u.total_tokens, 0);

    let cumulative = 0;
    const cumulativePercentages = sorted.map((u) => {
      cumulative += u.total_tokens;
      return totalTokensAll > 0 ? (cumulative / totalTokensAll) * 100 : 0;
    });

    return {
      labels: sorted.map((u) => {
        const name = u.display_name || u.username;
        return name.length > 15 ? name.slice(0, 14) + '\u2026' : name;
      }),
      datasets: [
        {
          type: 'bar' as const,
          label: 'Total Tokens',
          data: sorted.map((u) => u.total_tokens),
          backgroundColor: sorted.map((_, idx) => CHART_PALETTE[idx % CHART_PALETTE.length] + '99'),
          borderRadius: 4,
          borderSkipped: false as const,
          yAxisID: 'yTokens',
          order: 2,
        },
        {
          type: 'line' as const,
          label: 'Cumulative %',
          data: cumulativePercentages,
          borderColor: '#ef4444',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.2,
          yAxisID: 'yPercent',
          order: 1,
        },
      ],
    };
  }, [usageSummary]);

  const paretoChartOptions = useMemo<ChartOptions<'bar'>>(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: themeColors.text, boxWidth: 10, boxHeight: 10 },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (ctx) => {
            if (ctx.dataset.yAxisID === 'yPercent') {
              return `${ctx.dataset.label}: ${typeof ctx.raw === 'number' ? ctx.raw.toFixed(1) : ''}%`;
            }
            return `${ctx.dataset.label}: ${typeof ctx.raw === 'number' ? formatNumber(ctx.raw) : ''}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: themeColors.textSecondary, maxRotation: 45, minRotation: 45 },
        grid: { color: themeColors.grid },
      },
      yTokens: {
        type: 'linear',
        position: 'left',
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
      yPercent: {
        type: 'linear',
        position: 'right',
        min: 0,
        max: 100,
        ticks: { color: '#ef4444', callback: (value) => `${value}%` },
        grid: { drawOnChartArea: false },
      },
    },
  }), [themeColors]);

  const sortedMcpUsers = useMemo(() => {
    const rows = [...mcpUsers];
    return rows.sort((a, b) => {
      const sortValueA: Record<McpUserSortKey, string | number | null> = {
        user: a.display_name || a.username,
        auth: a.auth_method,
        route: a.route_name,
        requests: a.total_requests,
        success: a.success_count,
        errors: a.error_count,
      };
      const sortValueB: Record<McpUserSortKey, string | number | null> = {
        user: b.display_name || b.username,
        auth: b.auth_method,
        route: b.route_name,
        requests: b.total_requests,
        success: b.success_count,
        errors: b.error_count,
      };
      return compareSortValues(sortValueA[mcpUserSort.key], sortValueB[mcpUserSort.key], mcpUserSort.direction);
    });
  }, [mcpUserSort.direction, mcpUserSort.key, mcpUsers]);

  const totalRequests = usageSummary.reduce((s, u) => s + u.total_requests, 0);
  const totalTokens = usageSummary.reduce((s, u) => s + u.total_tokens, 0);
  const totalFailed = usageSummary.reduce((s, u) => s + u.failed_count + u.interrupted_count, 0);
  const totalCompleted = usageSummary.reduce((s, u) => s + u.completed_count, 0);
  const totalFailedOnly = usageSummary.reduce((s, u) => s + u.failed_count, 0);
  const totalInterrupted = usageSummary.reduce((s, u) => s + u.interrupted_count, 0);
  const successRate = totalRequests > 0 ? (totalCompleted / totalRequests) * 100 : 100;

  const totalWorkspaces = workspaces.length;
  const liveWorkspaces = Object.values(workspaceStateById).filter((s) => s.has_live_task).length;
  const interruptedWorkspaces = Object.values(workspaceStateById).filter((s) => s.has_interrupted_task).length;

  const isSelf = (userId: string) => currentUser?.id === userId;

  const sortedManagementRows = useMemo(() => {
    const rows = [...userStatsRows];
    return rows.sort((a, b) => {
      const aFail = (a.usage?.failed_count || 0) + (a.usage?.interrupted_count || 0);
      const bFail = (b.usage?.failed_count || 0) + (b.usage?.interrupted_count || 0);

      const sortValueA: Record<ManagementSortKey, string | number | null> = {
        user: (a.user.display_name || a.user.username),
        auth: a.user.auth_provider,
        chats: standaloneChatCountsByUserId[a.user.id] ?? 0,
        workspaces: a.ownedWorkspaceCount,
        memberships: a.memberWorkspaceCount,
        workspaceChats: a.workspaceConversationCount,
        liveInterrupted: a.liveWorkspaceCount + a.interruptedWorkspaceCount,
        storage: a.storageBytesKnown,
        role: a.user.role,
        actions: null,
      };
      const sortValueB: Record<ManagementSortKey, string | number | null> = {
        user: (b.user.display_name || b.user.username),
        auth: b.user.auth_provider,
        chats: standaloneChatCountsByUserId[b.user.id] ?? 0,
        workspaces: b.ownedWorkspaceCount,
        memberships: b.memberWorkspaceCount,
        workspaceChats: b.workspaceConversationCount,
        liveInterrupted: b.liveWorkspaceCount + b.interruptedWorkspaceCount,
        storage: b.storageBytesKnown,
        role: b.user.role,
        actions: null,
      };

      const result = compareSortValues(sortValueA[managementSort.key], sortValueB[managementSort.key], managementSort.direction);
      if (result !== 0) return result;
      // Preserve previous relevance as deterministic tiebreaker
      if ((b.usage?.total_requests || 0) !== (a.usage?.total_requests || 0)) {
        return (b.usage?.total_requests || 0) - (a.usage?.total_requests || 0);
      }
      if (bFail !== aFail) return bFail - aFail;
      return a.user.username.localeCompare(b.user.username);
    });
  }, [managementSort.direction, managementSort.key, standaloneChatCountsByUserId, userStatsRows]);

  const sortedUsageRows = useMemo(() => {
    const rows = [...usageSummary];
    return rows.sort((a, b) => {
      const aFail = a.failed_count + a.interrupted_count;
      const bFail = b.failed_count + b.interrupted_count;
      const sortValueA: Record<UsageSortKey, string | number | null> = {
        user: a.display_name || a.username,
        requests: a.total_requests,
        input: a.total_input_tokens,
        output: a.total_output_tokens,
        total: a.total_tokens,
        completed: a.completed_count,
        failed: aFail,
      };
      const sortValueB: Record<UsageSortKey, string | number | null> = {
        user: b.display_name || b.username,
        requests: b.total_requests,
        input: b.total_input_tokens,
        output: b.total_output_tokens,
        total: b.total_tokens,
        completed: b.completed_count,
        failed: bFail,
      };
      return compareSortValues(sortValueA[usageSort.key], sortValueB[usageSort.key], usageSort.direction);
    });
  }, [usageSort.direction, usageSort.key, usageSummary]);

  const sortedProviderRows = useMemo(() => {
    const rows = [...providerBreakdown];
    return rows.sort((a, b) => {
      const sortValueA: Record<ProviderSortKey, string | number | null> = {
        provider: formatProviderDisplayName(a.provider),
        model: formatModelDisplayName(a.model, a.provider),
        source: a.request_source,
        requests: a.total_requests,
        input: a.total_input_tokens,
        output: a.total_output_tokens,
        total: a.total_tokens,
      };
      const sortValueB: Record<ProviderSortKey, string | number | null> = {
        provider: formatProviderDisplayName(b.provider),
        model: formatModelDisplayName(b.model, b.provider),
        source: b.request_source,
        requests: b.total_requests,
        input: b.total_input_tokens,
        output: b.total_output_tokens,
        total: b.total_tokens,
      };
      return compareSortValues(sortValueA[providerSort.key], sortValueB[providerSort.key], providerSort.direction);
    });
  }, [providerBreakdown, providerSort.direction, providerSort.key]);

  const sortedDailyRows = useMemo(() => {
    const rows = [...dailyCombinedRows];
    return rows.sort((a, b) => {
      const sortValueA: Record<DailySortKey, string | number | null> = {
        date: a.date,
        requests: a.total_requests,
        input: a.total_input_tokens,
        output: a.total_output_tokens,
        total: a.total_tokens,
        completed: a.completed_count,
        failed: a.failed_count,
        mcpRequests: a.mcp_requests,
        mcpErrors: a.mcp_errors,
        apiRequests: a.api_requests,
        apiErrors: a.api_errors,
      };
      const sortValueB: Record<DailySortKey, string | number | null> = {
        date: b.date,
        requests: b.total_requests,
        input: b.total_input_tokens,
        output: b.total_output_tokens,
        total: b.total_tokens,
        completed: b.completed_count,
        failed: b.failed_count,
        mcpRequests: b.mcp_requests,
        mcpErrors: b.mcp_errors,
        apiRequests: b.api_requests,
        apiErrors: b.api_errors,
      };
      return compareSortValues(sortValueA[dailySort.key], sortValueB[dailySort.key], dailySort.direction);
    });
  }, [dailyCombinedRows, dailySort.direction, dailySort.key]);

  const managementColumns = useMemo<DataTableColumn<DerivedUserStats, ManagementSortKey>[]>(() => ([
    { key: 'user', label: 'User' },
    { key: 'auth', label: 'Auth' },
    { key: 'chats', label: 'Chats', headerClassName: 'num', cellClassName: 'num' },
    { key: 'workspaces', label: 'Workspaces', headerClassName: 'num', cellClassName: 'num' },
    { key: 'memberships', label: 'Memberships', headerClassName: 'num', cellClassName: 'num' },
    { key: 'workspaceChats', label: 'Workspace Chats', headerClassName: 'num', cellClassName: 'num' },
    { key: 'liveInterrupted', label: 'Live/Interrupted', headerClassName: 'num', cellClassName: 'num' },
    { key: 'storage', label: 'Storage', headerClassName: 'num', cellClassName: 'num' },
    { key: 'role', label: 'Role' },
    { key: 'actions', label: 'Actions', headerClassName: 'num', cellClassName: 'num', sortable: false },
  ]), []);

  const usageColumns = useMemo<DataTableColumn<UserUsageSummary, UsageSortKey>[]>(() => ([
    { key: 'user', label: 'User' },
    { key: 'requests', label: 'Requests', headerClassName: 'num', cellClassName: 'num' },
    { key: 'input', label: 'Input Tokens', headerClassName: 'num', cellClassName: 'num' },
    { key: 'output', label: 'Output Tokens', headerClassName: 'num', cellClassName: 'num' },
    { key: 'total', label: 'Total Tokens', headerClassName: 'num', cellClassName: 'num' },
    { key: 'completed', label: 'Completed', headerClassName: 'num', cellClassName: 'num' },
    { key: 'failed', label: 'Failed', headerClassName: 'num', cellClassName: 'num' },
  ]), []);

  const providerColumns = useMemo<DataTableColumn<ProviderModelBreakdown, ProviderSortKey>[]>(() => ([
    { key: 'provider', label: 'Provider' },
    { key: 'model', label: 'Model' },
    { key: 'source', label: 'Source' },
    { key: 'requests', label: 'Requests', headerClassName: 'num', cellClassName: 'num' },
    { key: 'input', label: 'Input Tokens', headerClassName: 'num', cellClassName: 'num' },
    { key: 'output', label: 'Output Tokens', headerClassName: 'num', cellClassName: 'num' },
    { key: 'total', label: 'Total Tokens', headerClassName: 'num', cellClassName: 'num' },
  ]), []);

  const mcpUsersColumns = useMemo<DataTableColumn<McpUserUsage, McpUserSortKey>[]>(() => ([
    { key: 'user', label: 'User' },
    { key: 'auth', label: 'Auth' },
    { key: 'route', label: 'Route' },
    { key: 'requests', label: 'Requests', headerClassName: 'num', cellClassName: 'num' },
    { key: 'success', label: 'Success', headerClassName: 'num', cellClassName: 'num' },
    { key: 'errors', label: 'Errors', headerClassName: 'num', cellClassName: 'num' },
  ]), []);

  const dailyColumns = useMemo<DataTableColumn<DailyCombinedRow, DailySortKey>[]>(() => ([
    { key: 'date', label: 'Date' },
    { key: 'requests', label: 'Chat Req', headerClassName: 'num', cellClassName: 'num' },
    { key: 'completed', label: 'Chat OK', headerClassName: 'num', cellClassName: 'num' },
    { key: 'failed', label: 'Chat Fail', headerClassName: 'num', cellClassName: 'num' },
    { key: 'mcpRequests', label: 'MCP Req', headerClassName: 'num', cellClassName: 'num' },
    { key: 'mcpErrors', label: 'MCP Fail', headerClassName: 'num', cellClassName: 'num' },
    { key: 'apiRequests', label: 'API Req', headerClassName: 'num', cellClassName: 'num' },
    { key: 'apiErrors', label: 'API Fail', headerClassName: 'num', cellClassName: 'num' },
    { key: 'total', label: 'Tokens', headerClassName: 'num', cellClassName: 'num' },
  ]), []);

  const handleSort = <K extends string>(current: TableSortConfig<K>, key: K, setter: (next: TableSortConfig<K>) => void) => {
    const direction = current.key === key && current.direction === 'asc' ? 'desc' : 'asc';
    setter({ key, direction });
  };

  const managementPaging = paginate(sortedManagementRows, managementPage, managementPageSize);
  const usagePaging = paginate(sortedUsageRows, usagePage, usagePageSize);
  const providerPaging = paginate(sortedProviderRows, providerPage, providerPageSize);
  const dailyPaging = paginate(sortedDailyRows, dailyPage, dailyPageSize);
  const mcpUsersPaging = paginate(sortedMcpUsers, mcpUsersPage, mcpUsersPageSize);

  useEffect(() => {
    if (managementPage !== managementPaging.safePage) setManagementPage(managementPaging.safePage);
  }, [managementPage, managementPaging.safePage]);

  useEffect(() => {
    if (usagePage !== usagePaging.safePage) setUsagePage(usagePaging.safePage);
  }, [usagePage, usagePaging.safePage]);

  useEffect(() => {
    if (providerPage !== providerPaging.safePage) setProviderPage(providerPaging.safePage);
  }, [providerPage, providerPaging.safePage]);

  useEffect(() => {
    if (dailyPage !== dailyPaging.safePage) setDailyPage(dailyPaging.safePage);
  }, [dailyPage, dailyPaging.safePage]);

  useEffect(() => {
    if (mcpUsersPage !== mcpUsersPaging.safePage) setMcpUsersPage(mcpUsersPaging.safePage);
  }, [mcpUsersPage, mcpUsersPaging.safePage]);

  return (
    <div className="users-panel">
      <div className="users-header-bar">
        <div className="tabs" style={{ marginBottom: 0 }}>
          <button className={`tab ${activeTab === 'management' ? 'active' : ''}`} onClick={() => setActiveTab('management')}>
            Users
          </button>
          <button className={`tab ${activeTab === 'usage' ? 'active' : ''}`} onClick={() => setActiveTab('usage')}>
            Usage
          </button>
        </div>
        {activeTab === 'usage' && (
          <div className="users-panel-controls">
            {hasLoadedUsageRange ? (
              availableRanges.map((d) => (
                <button
                  key={d}
                  className={`users-range-btn ${days === d ? 'active' : ''}`}
                  onClick={() => setDays(d)}
                >
                  {d}d
                </button>
              ))
            ) : (
              <span className="users-subnum">loading ranges...</span>
            )}
            {usageLoading && <span className="users-subnum">loading...</span>}
          </div>
        )}
      </div>

      <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />

      {loading ? (
        <div className="card"><div className="card-body"><p>Loading...</p></div></div>
      ) : activeTab === 'management' ? (
        <>
          <div className="users-summary-row">
            <div className="users-summary-card">
              <div className="users-summary-value">{users.length}</div>
              <div className="users-summary-label">Users</div>
            </div>
            <div className="users-summary-card">
              <div className="users-summary-value">{totalWorkspaces}</div>
              <div className="users-summary-label">Workspaces</div>
            </div>
            <div className="users-summary-card">
              <div className="users-summary-value">
                {liveWorkspaces}
                {liveWorkspaces > 0 && <MiniLoadingSpinner variant="icon" size={14} className="users-live-spin" />}
              </div>
              <div className="users-summary-label">Live Task Workspaces</div>
            </div>
            <div className="users-summary-card">
              <div className="users-summary-value">{interruptedWorkspaces}</div>
              <div className="users-summary-label">Interrupted Workspaces</div>
            </div>
          </div>

          <div className="card">
            <div className="card-header"><h3>User Accounts</h3></div>
            <div className="card-body users-compact-card-body">
              {userStatsRows.length === 0 ? (
                <p className="muted-text">No users found.</p>
              ) : (
                <>
                  <div className="users-table-wrap users-table-compact-wrap">
                    <DataTable
                      rows={managementPaging.pageItems}
                      columns={managementColumns}
                      sortConfig={managementSort}
                      onSort={(key) => {
                        if (key === 'actions') return;
                        handleSort(managementSort, key as ManagementSortKey, setManagementSort);
                      }}
                      renderRow={(row) => {
                        const user = row.user;
                        const chatCount = standaloneChatCountsByUserId[user.id] ?? 0;
                        const failedCount = (row.usage?.failed_count || 0) + (row.usage?.interrupted_count || 0);
                        const isRowSelf = isSelf(user.id);
                        const sortedOwnedWorkspaces = [...row.ownedWorkspaces].sort((left, right) => {
                          const leftLast = toEpochMs(workspaceLastMessageAtById[left.id]);
                          const rightLast = toEpochMs(workspaceLastMessageAtById[right.id]);
                          if (rightLast !== leftLast) {
                            return rightLast - leftLast;
                          }
                          return left.name.localeCompare(right.name);
                        });
                        const sortedStandaloneChats = [...(standaloneChatsByUserId[user.id] ?? [])].sort((left, right) => {
                          const updatedDiff = toEpochMs(right.updated_at) - toEpochMs(left.updated_at);
                          if (updatedDiff !== 0) {
                            return updatedDiff;
                          }
                          return (left.title ?? '').localeCompare(right.title ?? '');
                        });

                        return (
                          <Fragment key={user.id}>
                            <tr className={isRowSelf ? 'users-row-self' : ''}>
                              <td>
                                <div className="users-cell-identity">
                                  <span className="users-username">
                                    {user.display_name || user.username}
                                    {isRowSelf && <span className="users-you-badge">you</span>}
                                  </span>
                                  <span className="users-handle">@{user.username}</span>
                                </div>
                              </td>
                              <td>
                                <span className={`users-auth-badge users-auth-${user.auth_provider}`}>
                                  {user.auth_provider}
                                </span>
                                {user.source_provider && user.source_provider !== user.auth_provider && (
                                  <div className="users-subnum">source: {user.source_provider}</div>
                                )}
                                {user.source_synced_at && (
                                  <div className="users-subnum">synced {new Date(user.source_synced_at).toLocaleDateString()}</div>
                                )}
                              </td>
                              <td className="num">
                                <button
                                  type="button"
                                  className="users-link-btn"
                                  onClick={() => { void toggleUserChatDetails(user.id); }}
                                  title="Show non-workspace chats"
                                >
                                  {formatNumber(chatCount)}
                                </button>
                                {failedCount > 0 && <div className="users-subnum">{failedCount} fail/int</div>}
                              </td>
                              <td className="num">
                                <button
                                  type="button"
                                  className="users-link-btn"
                                  onClick={() => toggleUserWorkspaceDetails(user.id)}
                                  title="Show workspace operations"
                                >
                                  {row.ownedWorkspaceCount}
                                </button>
                              </td>
                              <td className="num">{row.memberWorkspaceCount}</td>
                              <td className="num">{row.workspaceConversationCount}</td>
                              <td className="num">
                                {row.liveWorkspaceCount > 0 && <MiniLoadingSpinner variant="icon" size={12} className="users-live-spin" title="Live task running" />}
                                {row.liveWorkspaceCount}/{row.interruptedWorkspaceCount}
                              </td>
                              <td className="num">
                                {row.ownedWorkspaceCount === 0 ? (
                                  <span className="users-action-muted">-</span>
                                ) : row.storageCoverage > 0 ? (
                                  <div>
                                    <div>{formatBytes(row.storageBytesKnown)}</div>
                                    {row.storageCoverage < 1 && (
                                      <div className="users-subnum">{Math.round(row.storageCoverage * 100)}% sampled</div>
                                    )}
                                  </div>
                                ) : (
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-secondary users-btn-inline"
                                    disabled={storageLoadingByUserId[user.id]}
                                    onClick={() => handleComputeStorageForUser(user.id)}
                                  >
                                    {storageLoadingByUserId[user.id] ? '...' : 'Compute'}
                                  </button>
                                )}
                              </td>
                              <td>
                                <span className={`users-role-badge${user.role === 'admin' ? ' users-role-admin' : ''}`}>{user.role}</span>
                                {user.role_manually_set && (
                                  <div className="users-subnum">overridden</div>
                                )}
                                {getUserManualGroupIds(user).length > 0 && (
                                  <div className="users-subnum">{getUserManualGroupIds(user).length} manual group{getUserManualGroupIds(user).length !== 1 ? 's' : ''}</div>
                                )}
                                {(user.ldap_group_ids?.length ?? 0) > 0 && (
                                  <div className="users-subnum">{user.ldap_group_ids!.length} LDAP group{user.ldap_group_ids!.length !== 1 ? 's' : ''}</div>
                                )}
                              </td>
                              <td className="num">
                                {isRowSelf ? (
                                  <span className="users-action-muted">-</span>
                                ) : (
                                  <div className="users-confirm-group">
                                    <button
                                      type="button"
                                      className="btn btn-sm btn-secondary users-btn-inline"
                                      title="Edit role and groups"
                                      onClick={() => setEditingUserId(user.id)}
                                    >
                                      <Pencil size={13} />
                                    </button>
                                    <DeleteConfirmButton
                                      onDelete={() => handleDelete(user.id)}
                                      disabled={actionLoading === user.id}
                                      deleting={actionLoading === user.id}
                                      title="Delete user"
                                    />
                                  </div>
                                )}
                              </td>
                            </tr>
                            {expandedUserDetail?.userId === user.id && (
                              <tr key={`${user.id}-details`} className="users-workspace-detail-row">
                                <td colSpan={10}>
                                  <div className="users-detail-list-shell">
                                    {expandedUserDetail.mode === 'workspaces' ? (
                                      <WorkspaceRowList
                                        workspaces={sortedOwnedWorkspaces}
                                        users={users}
                                        deletingWorkspaceIds={deletingWorkspaceIds}
                                        onTransfer={handleTransferWorkspace}
                                        onDelete={handleDeleteWorkspace}
                                        onSelect={(workspace) => onOpenWorkspace(workspace.id)}
                                        emptyMessage="No owned workspaces."
                                        renderMeta={(ws) => {
                                          const state = workspaceStateById[ws.id];
                                          const wsStorage = storageByWorkspaceId[ws.id];
                                          const lastMessageAt = workspaceLastMessageAtById[ws.id] ?? null;
                                          const latestConversation = workspaceLastConversationById[ws.id] ?? null;
                                          const stateLabel = state?.has_live_task
                                            ? 'live'
                                            : state?.has_interrupted_task
                                              ? 'interrupted'
                                              : 'idle';
                                          return (
                                            <span className="users-detail-meta users-detail-meta-workspace admin-ws-item-date">
                                              <span className="users-detail-col">Chats {ws.conversation_ids.length}</span>
                                              <span className="users-detail-col">
                                                {workspaceLastMessageLoadingByUserId[user.id]
                                                  ? 'Last message loading'
                                                  : formatDateTime(lastMessageAt)}
                                              </span>
                                              <span className="users-detail-col">Context {getConversationContextMeta(latestConversation)}</span>
                                              <span className="users-detail-col">Model {latestConversation ? formatModelDisplayName(latestConversation.model) : 'n/a'}</span>
                                              <span className="users-detail-col">Status {stateLabel}</span>
                                              <span className="users-detail-col">Storage {typeof wsStorage === 'number' ? formatBytes(wsStorage) : 'not sampled'}</span>
                                            </span>
                                          );
                                        }}
                                      />
                                    ) : (
                                      <UserConversationRowList
                                        conversations={sortedStandaloneChats}
                                        loading={standaloneChatsLoadingUserId === user.id}
                                        onSelect={(conversation) => onOpenChat(conversation.id)}
                                        onDelete={handleDeleteConversation}
                                        onCancelTask={handleCancelConversationTask}
                                        renderMeta={(conversation) => {
                                          const messageCount = conversation.messages?.length ?? 0;
                                          const taskState = conversation.active_task_id ? 'running' : 'idle';
                                          return (
                                            <span className="users-detail-meta users-detail-meta-chat admin-ws-item-date">
                                              <span className="users-detail-col">Messages {messageCount}</span>
                                              <span className="users-detail-col">{formatDateTime(conversation.updated_at)}</span>
                                              <span className="users-detail-col">Context {getConversationContextMeta(conversation)}</span>
                                              <span className="users-detail-col">Model {formatModelDisplayName(conversation.model)}</span>
                                              <span className="users-detail-col">Task {taskState}</span>
                                            </span>
                                          );
                                        }}
                                        emptyMessage="No non-workspace chats for this user."
                                      />
                                    )}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        );
                      }}
                    />
                  </div>
                  <TablePager
                    page={managementPaging.safePage}
                    totalPages={managementPaging.totalPages}
                    totalItems={userStatsRows.length}
                    pageSize={managementPageSize}
                    onPageChange={setManagementPage}
                    onPageSizeChange={(size) => { setManagementPageSize(size); setManagementPage(1); }}
                  />
                  <div className="users-admin-table-actions">
                    <button type="button" className="btn btn-secondary" onClick={() => setShowCreateLocalUserModal(true)}>
                      <UserPlus size={16} />
                      Create Internal User
                    </button>
                    <button type="button" className="btn btn-secondary" onClick={() => setShowManageAuthGroupsModal(true)}>
                      <Shield size={16} />
                      Manage Group Memberships
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </>
      ) : usageLoadedDays === null ? (
        <div className="card"><div className="card-body"><p>Loading usage data...</p></div></div>
      ) : (
        <>
          <div className="users-summary-row">
            <div className="users-summary-card">
              <div className="users-summary-value">{users.length}</div>
              <div className="users-summary-label">Users</div>
            </div>
            <div className="users-summary-card">
              <div className="users-summary-value">{formatNumber(totalRequests)}</div>
              <div className="users-summary-label">Requests ({days}d)</div>
            </div>
            <div className="users-summary-card">
              <div className="users-summary-value">{formatNumber(totalTokens)}</div>
              <div className="users-summary-label">Tokens ({days}d)</div>
            </div>
            <div className="users-summary-card">
              <div className={`users-summary-value ${successRate >= 95 ? 'users-slo-green' : successRate >= 90 ? 'users-slo-yellow' : 'users-slo-red'}`}>
                {successRate.toFixed(1)}%
              </div>
              <div className="users-summary-label">
                Success Rate
                {totalFailed > 0 && <span className="users-slo-detail"> ({totalFailedOnly}f · {totalInterrupted}i)</span>}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header users-card-header-inline">
              <h3>Trends</h3>
              <div className="users-inline-switch">
                <button
                  type="button"
                  className={`users-switch-btn ${trendTab === 'reliability' ? 'active' : ''}`}
                  onClick={() => setTrendTab('reliability')}
                >
                  Reliability
                </button>
                <button
                  type="button"
                  className={`users-switch-btn ${trendTab === 'daily-chart' ? 'active' : ''}`}
                  onClick={() => setTrendTab('daily-chart')}
                >
                  Daily Chart
                </button>
                <button
                  type="button"
                  className={`users-switch-btn ${trendTab === 'daily-table' ? 'active' : ''}`}
                  onClick={() => setTrendTab('daily-table')}
                >
                  Daily Table
                </button>
              </div>
            </div>
            <div className="card-body users-compact-card-body">
              {dailyTrend.length === 0 ? (
                <p className="muted-text">No daily data yet.</p>
              ) : trendTab === 'reliability' ? (
                <div className="users-chart-shell">
                  <Line data={reliabilityTrendData} options={reliabilityTrendOptions} />
                </div>
              ) : trendTab === 'daily-chart' ? (
                <div className="users-data-slot">
                  <div className="users-chart-shell">
                    <Line data={dailyChartData} options={dailyLineOptions} />
                  </div>
                </div>
              ) : (
                <div className="users-data-slot">
                  <div className="users-table-wrap">
                    <DataTable
                      rows={dailyPaging.pageItems}
                      columns={dailyColumns}
                      sortConfig={dailySort}
                      onSort={(key) => handleSort(dailySort, key as DailySortKey, setDailySort)}
                      renderRow={(row) => (
                        <tr key={row.date}>
                          <td>{row.date}</td>
                          <td className="num">{formatNumber(row.total_requests)}</td>
                          <td className="num">{row.completed_count}</td>
                          <td className="num">{row.failed_count}</td>
                          <td className="num">{formatNumber(row.mcp_requests)}</td>
                          <td className="num">{row.mcp_errors}</td>
                          <td className="num">{formatNumber(row.api_requests)}</td>
                          <td className="num">{row.api_errors}</td>
                          <td className="num">{formatNumber(row.total_tokens)}</td>
                        </tr>
                      )}
                    />
                  </div>
                  <TablePager
                    page={dailyPaging.safePage}
                    totalPages={dailyPaging.totalPages}
                    totalItems={dailyCombinedRows.length}
                    pageSize={dailyPageSize}
                    onPageChange={setDailyPage}
                    onPageSizeChange={(size) => { setDailyPageSize(size); setDailyPage(1); }}
                  />
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header users-card-header-inline">
              <h3>Per-User Usage</h3>
              <div className="users-inline-switch">
                <button
                  type="button"
                  className={`users-switch-btn ${perUserChartMetric === 'requests' ? 'active' : ''}`}
                  onClick={() => setPerUserChartMetric('requests')}
                >
                  Requests
                </button>
                <button
                  type="button"
                  className={`users-switch-btn ${perUserChartMetric === 'tokens' ? 'active' : ''}`}
                  onClick={() => setPerUserChartMetric('tokens')}
                >
                  Tokens
                </button>
              </div>
            </div>
            <div className="card-body users-compact-card-body">
              {usageSummary.length === 0 ? (
                <p className="muted-text">No usage data yet.</p>
              ) : (
                <div className="users-two-column">
                  <div className="users-two-column-panel users-chart-panel">
                    <div className="users-chart-caption">
                      Top 5 users by {perUserChartMetric === 'requests' ? 'requests' : 'total tokens'} over {days}d
                    </div>
                    <div className="users-chart-shell users-chart-shell-tall">
                      <Line data={perUserChartData} options={singleAxisLineOptions} />
                    </div>
                  </div>
                  <div className="users-two-column-panel">
                    <div className="users-table-wrap">
                      <DataTable
                        rows={usagePaging.pageItems}
                        columns={usageColumns}
                        sortConfig={usageSort}
                        onSort={(key) => handleSort(usageSort, key as UsageSortKey, setUsageSort)}
                        renderRow={(row) => (
                          <tr key={row.user_id}>
                            <td>
                              <span className="users-username">{row.display_name || row.username}</span>
                              {row.display_name && <span className="users-handle">@{row.username}</span>}
                            </td>
                            <td className="num">{formatNumber(row.total_requests)}</td>
                            <td className="num">{formatNumber(row.total_input_tokens)}</td>
                            <td className="num">{formatNumber(row.total_output_tokens)}</td>
                            <td className="num">{formatNumber(row.total_tokens)}</td>
                            <td className="num">{row.completed_count}</td>
                            <td className="num">{row.failed_count + row.interrupted_count}</td>
                          </tr>
                        )}
                      />
                    </div>
                    <TablePager
                      page={usagePaging.safePage}
                      totalPages={usagePaging.totalPages}
                      totalItems={usageSummary.length}
                      pageSize={usagePageSize}
                      onPageChange={setUsagePage}
                      onPageSizeChange={(size) => { setUsagePageSize(size); setUsagePage(1); }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="users-two-column" style={{ marginBottom: 0 }}>
            <div className="card" style={{ minWidth: 0, marginBottom: 0 }}>
              <div className="card-header users-card-header-inline">
                <h3>MCP / API Requests</h3>
                <div className="users-inline-switch">
                  <button
                    type="button"
                    className={`users-switch-btn ${mcpUsageTab === 'chart' ? 'active' : ''}`}
                    onClick={() => setMcpUsageTab('chart')}
                  >
                    Chart
                  </button>
                  <button
                    type="button"
                    className={`users-switch-btn ${mcpUsageTab === 'table' ? 'active' : ''}`}
                    onClick={() => setMcpUsageTab('table')}
                  >
                    Table
                  </button>
                </div>
              </div>
              <div className="card-body users-compact-card-body">
                {mcpUsageTab === 'chart' ? (
                  mcpDailyRows.every((row) => row.total_requests === 0) ? (
                    <p className="muted-text">No MCP requests in the selected window.</p>
                  ) : (
                    <>
                      <div className="users-chart-caption">Daily MCP requests split by success/error</div>
                      <div className="users-chart-shell users-chart-shell-tall">
                        <Bar data={mcpChartData} options={mcpChartOptions} />
                      </div>
                    </>
                  )
                ) : (
                  <div className="users-data-slot">
                    <div className="users-table-wrap">
                      <DataTable
                        rows={mcpUsersPaging.pageItems}
                        columns={mcpUsersColumns}
                        sortConfig={mcpUserSort}
                        onSort={(key) => handleSort(mcpUserSort, key as McpUserSortKey, setMcpUserSort)}
                        renderRow={(row) => (
                          <tr key={`${row.user_id}-${row.auth_method}-${row.route_name}`}>
                            <td>
                              <span className="users-username">{row.display_name || row.username}</span>
                              {row.display_name && row.username !== 'anonymous' && <span className="users-handle">@{row.username}</span>}
                            </td>
                            <td>{row.auth_method}</td>
                            <td>{row.route_name}</td>
                            <td className="num">{formatNumber(row.total_requests)}</td>
                            <td className="num">{formatNumber(row.success_count)}</td>
                            <td className="num">{formatNumber(row.error_count)}</td>
                          </tr>
                        )}
                      />
                    </div>
                    <TablePager
                      page={mcpUsersPaging.safePage}
                      totalPages={mcpUsersPaging.totalPages}
                      totalItems={sortedMcpUsers.length}
                      pageSize={mcpUsersPageSize}
                      onPageChange={setMcpUsersPage}
                      onPageSizeChange={(size) => { setMcpUsersPageSize(size); setMcpUsersPage(1); }}
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="card" style={{ minWidth: 0, marginBottom: 0 }}>
              <div className="card-header"><h3>Token Concentration</h3></div>
              <div className="card-body users-compact-card-body">
                {usageSummary.length < 2 ? (
                  <p className="muted-text">Not enough users for distribution analysis.</p>
                ) : (
                  <>
                    <div className="users-chart-caption">
                      Users sorted by token usage — bars show total tokens, line shows cumulative share
                    </div>
                    <div className="users-chart-shell users-chart-shell-tall">
                      <Chart type="bar" data={paretoChartData as never} options={paretoChartOptions} />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header users-card-header-inline">
              <h3>Provider / Model Breakdown</h3>
              <div className="users-inline-switch">
                <button
                  type="button"
                  className={`users-switch-btn ${providerChartMetric === 'requests' ? 'active' : ''}`}
                  onClick={() => setProviderChartMetric('requests')}
                >
                  Requests
                </button>
                <button
                  type="button"
                  className={`users-switch-btn ${providerChartMetric === 'tokens' ? 'active' : ''}`}
                  onClick={() => setProviderChartMetric('tokens')}
                >
                  Tokens
                </button>
              </div>
            </div>
            <div className="card-body users-compact-card-body">
              {providerBreakdown.length === 0 ? (
                <p className="muted-text">No provider data yet.</p>
              ) : (
                <div className="users-two-column">
                  <div className="users-two-column-panel users-chart-panel">
                    <div className="users-chart-caption">Bar chart for the selected {days}d range</div>
                    <div className="users-chart-shell users-chart-shell-tall">
                      <Bar data={providerChartData} options={providerBarOptions} />
                    </div>
                  </div>
                  <div className="users-two-column-panel">
                    <div className="users-table-wrap">
                      <DataTable
                        rows={providerPaging.pageItems}
                        columns={providerColumns}
                        sortConfig={providerSort}
                        onSort={(key) => handleSort(providerSort, key as ProviderSortKey, setProviderSort)}
                        renderRow={(row) => (
                          <tr key={`${row.provider}-${row.model}-${row.request_source}`}>
                            <td>{formatProviderDisplayName(row.provider)}</td>
                            <td>{formatModelDisplayName(row.model, row.provider)}</td>
                            <td>{row.request_source === 'ui' ? 'Ragtime Chat' : row.request_source === 'api' ? 'API' : row.request_source}</td>
                            <td className="num">{formatNumber(row.total_requests)}</td>
                            <td className="num">{formatNumber(row.total_input_tokens)}</td>
                            <td className="num">{formatNumber(row.total_output_tokens)}</td>
                            <td className="num">{formatNumber(row.total_tokens)}</td>
                          </tr>
                        )}
                      />
                    </div>
                    <TablePager
                      page={providerPaging.safePage}
                      totalPages={providerPaging.totalPages}
                      totalItems={providerBreakdown.length}
                      pageSize={providerPageSize}
                      onPageChange={setProviderPage}
                      onPageSizeChange={(size) => { setProviderPageSize(size); setProviderPage(1); }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

        </>
      )}

      {editingUserId !== null && (() => {
        const editUser = users.find((u) => u.id === editingUserId);
        if (!editUser) return null;
        return (
          <UserEditModal
            user={editUser}
            authGroups={authGroups}
            actionLoading={actionLoading}
            onRoleChange={handleRoleChange}
            onResetRoleOverride={handleResetRoleOverride}
            onLocalGroupsChange={handleLocalGroupsChange}
            onClose={() => setEditingUserId(null)}
          />
        );
      })()}

      <AuthAdminModalHost
        createUserOpen={showCreateLocalUserModal}
        manageGroupsOpen={showManageAuthGroupsModal}
        authGroups={authGroups}
        onAuthGroupsChange={setAuthGroups}
        onUsersChanged={loadUsers}
        onCloseCreateUser={() => setShowCreateLocalUserModal(false)}
        onCloseManageGroups={() => setShowManageAuthGroupsModal(false)}
        toast={toast}
      />
    </div>
  );
}
