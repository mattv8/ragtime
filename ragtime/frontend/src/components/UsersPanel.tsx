import { useState, useEffect, useCallback, useMemo, Fragment } from 'react';
import { api } from '@/api';
import type {
  User,
  UserUsageSummary,
  ProviderModelBreakdown,
  DailyUsageTrend,
  UserDailyUsageSeriesPoint,
  ProviderDailyFailureCell,
  UserSpaceWorkspace,
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
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { formatProviderDisplayName, formatModelDisplayName } from '@/utils/modelDisplay';

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

type PanelTab = 'management' | 'usage';
type TrendTab = 'reliability' | 'daily-chart' | 'daily-table';
type UsageMetric = 'requests' | 'tokens';

const ALL_DAY_RANGES = [7, 30, 90, 180, 240, 360] as const;

interface UsersPanelProps {
  currentUser: User | null;
}

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

export function UsersPanel({ currentUser }: UsersPanelProps) {
  const [activeTab, setActiveTab] = useState<PanelTab>('management');

  const [users, setUsers] = useState<User[]>([]);
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspaceStateById, setWorkspaceStateById] = useState<Record<string, WorkspaceConversationStateSummaryItem>>({});

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [usageSummary, setUsageSummary] = useState<UserUsageSummary[]>([]);
  const [providerBreakdown, setProviderBreakdown] = useState<ProviderModelBreakdown[]>([]);
  const [dailyTrend, setDailyTrend] = useState<DailyUsageTrend[]>([]);
  const [userDailySeries, setUserDailySeries] = useState<UserDailyUsageSeriesPoint[]>([]);
  const [failureCells, setFailureCells] = useState<ProviderDailyFailureCell[]>([]);
  const [days, setDays] = useState(30);
  const [earliestDate, setEarliestDate] = useState<string | null>(null);

  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const [trendTab, setTrendTab] = useState<TrendTab>('reliability');
  const [providerChartMetric, setProviderChartMetric] = useState<UsageMetric>('tokens');
  const [perUserChartMetric, setPerUserChartMetric] = useState<UsageMetric>('requests');

  const [managementPage, setManagementPage] = useState(1);
  const [usagePage, setUsagePage] = useState(1);
  const [providerPage, setProviderPage] = useState(1);
  const [dailyPage, setDailyPage] = useState(1);

  const [managementPageSize, setManagementPageSize] = useState(10);
  const [usagePageSize, setUsagePageSize] = useState(10);
  const [providerPageSize, setProviderPageSize] = useState(10);
  const [dailyPageSize, setDailyPageSize] = useState(10);

  const [expandedUserId, setExpandedUserId] = useState<string | null>(null);

  const themeColors = useThemeColors();

  const [storageByWorkspaceId, setStorageByWorkspaceId] = useState<Record<string, number>>({});
  const [storageLoadingByUserId, setStorageLoadingByUserId] = useState<Record<string, boolean>>({});

  const loadUsers = useCallback(async () => {
    const res = await api.listUsers();
    setUsers(res);
  }, []);

  const loadUsage = useCallback(async (d: number) => {
    const [summaryRes, providersRes, dailyRes, userDailyRes, failureRes] = await Promise.all([
      api.getUsageSummary(d),
      api.getUsageProviders(d),
      api.getUsageDaily(d),
      api.getUsageUsersDaily(d),
      api.getUsageProviderDailyFailures(d),
    ]);
    setUsageSummary(summaryRes.users);
    setProviderBreakdown(providersRes.providers);
    setDailyTrend(dailyRes.daily);
    setUserDailySeries(userDailyRes.series);
    setFailureCells(failureRes.cells);
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

    const summary = await api.getWorkspacesConversationStateSummary(all.map((w) => w.id));
    const byId = summary.reduce<Record<string, WorkspaceConversationStateSummaryItem>>((acc, item) => {
      acc[item.workspace_id] = item;
      return acc;
    }, {});
    setWorkspaceStateById(byId);
  }, []);

  const loadAll = useCallback(async (d: number) => {
    setLoading(true);
    setError(null);
    try {
      const [, , , rangeRes] = await Promise.all([
        loadUsers(),
        loadUsage(d),
        loadWorkspaces(),
        api.getUsageRange(),
      ]);
      setEarliestDate(rangeRes.earliest_date);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load users data');
    } finally {
      setLoading(false);
    }
  }, [loadUsers, loadUsage, loadWorkspaces]);

  useEffect(() => {
    void loadAll(days);
  }, [days, loadAll]);

  const handleRoleChange = async (userId: string, newRole: 'admin' | 'user') => {
    setActionLoading(userId);
    try {
      await api.updateUserRole(userId, newRole);
      setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, role: newRole } : u)));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to update role');
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
      setError(e instanceof Error ? e.message : 'Failed to delete user');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteWorkspace = async (workspaceId: string) => {
    try {
      await api.deleteUserSpaceWorkspace(workspaceId);
      setWorkspaces((prev) => prev.filter((w) => w.id !== workspaceId));
      setWorkspaceStateById((prev) => {
        const next = { ...prev };
        delete next[workspaceId];
        return next;
      });
      setStorageByWorkspaceId((prev) => {
        const next = { ...prev };
        delete next[workspaceId];
        return next;
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to delete workspace');
      throw e;
    }
  };

  const handleTransferWorkspace = async (workspaceId: string, newOwnerId: string) => {
    try {
      const updated = await api.updateUserSpaceWorkspace(workspaceId, { owner_user_id: newOwnerId });
      setWorkspaces((prev) => prev.map((ws) => (ws.id === workspaceId ? updated : ws)));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to transfer workspace ownership');
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
      setError(e instanceof Error ? e.message : 'Failed to compute workspace storage usage');
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

  const providerChartData = useMemo(() => ({
    labels: providerBreakdown.map((row) => {
      const providerLabel = formatProviderDisplayName(row.provider);
      const modelLabel = formatModelDisplayName(row.model, row.provider);
      const label = `${providerLabel} / ${modelLabel}`;
      return label.length > 36 ? label.slice(0, 35) + '\u2026' : label;
    }),
    datasets: [
      {
        label: providerChartMetric === 'requests' ? 'Requests' : 'Total Tokens',
        data: providerBreakdown.map((row) => (providerChartMetric === 'requests' ? row.total_requests : row.total_tokens)),
        backgroundColor: providerBreakdown.map((_, idx) => CHART_PALETTE[idx % CHART_PALETTE.length]),
        borderRadius: 6,
        borderSkipped: false,
        maxBarThickness: 20,
      },
    ],
  }), [providerBreakdown, providerChartMetric]);

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

  const dailyChartData = useMemo(() => ({
    labels: dailyTrendRows.map((row) => row.date),
    datasets: [
      {
        label: 'Requests',
        data: dailyTrendRows.map((row) => row.total_requests),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.25)',
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
  }), [dailyTrendRows]);

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
      legend: { display: false },
      tooltip: { mode: 'nearest', intersect: false },
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: { color: themeColors.textSecondary },
        grid: { color: themeColors.grid },
      },
      y: {
        ticks: { color: themeColors.text, autoSkip: false },
        grid: { display: false },
      },
    },
  }), [themeColors]);

  const reliabilityTrendData = useMemo(() => ({
    labels: dailyTrendRows.map((row) => row.date),
    datasets: [
      {
        label: 'Success Rate %',
        data: dailyTrendRows.map((row) => {
          const total = row.total_requests;
          return total > 0 ? (row.completed_count / total) * 100 : null;
        }),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.12)',
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 4,
        fill: true,
      },
    ],
  }), [dailyTrendRows]);

  const reliabilityTrendOptions = useMemo<ChartOptions<'line'>>(() => {
    const rates = dailyTrendRows
      .filter((row) => row.total_requests > 0)
      .map((row) => (row.completed_count / row.total_requests) * 100);
    const minRate = rates.length > 0 ? Math.min(...rates) : 100;
    const dynamicMin = minRate >= 95 ? 90 : minRate >= 80 ? Math.floor(minRate / 10) * 10 : 0;

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
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
  }, [days, dailyTrendRows, themeColors]);

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

  const failureHeatmap = useMemo(() => {
    if (failureCells.length === 0) return null;

    const modelKeys = new Set<string>();
    const lookup = new Map<string, { failed: number; interrupted: number }>();

    for (const c of failureCells) {
      const date = normalizeDateLabel(c.date);
      const key = `${c.provider}::${c.model}`;
      modelKeys.add(key);
      const mapKey = `${date}|${key}`;
      const existing = lookup.get(mapKey);
      if (existing) {
        existing.failed += c.failed_count;
        existing.interrupted += c.interrupted_count;
      } else {
        lookup.set(mapKey, { failed: c.failed_count, interrupted: c.interrupted_count });
      }
    }

    const sortedModels = [...modelKeys].sort();
    const modelLabels = sortedModels.map((key) => {
      const [prov, model] = key.split('::');
      const provLabel = formatProviderDisplayName(prov);
      const modelLabel = formatModelDisplayName(model, prov);
      const label = `${provLabel} / ${modelLabel}`;
      return label.length > 28 ? label.slice(0, 27) + '\u2026' : label;
    });

    let maxCount = 0;
    for (const v of lookup.values()) {
      const total = v.failed + v.interrupted;
      if (total > maxCount) maxCount = total;
    }

    return { dates: dateLabels, models: sortedModels, modelLabels, lookup, maxCount };
  }, [failureCells, dateLabels]);

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

  const managementPaging = paginate(userStatsRows, managementPage, managementPageSize);
  const usagePaging = paginate(usageSummary, usagePage, usagePageSize);
  const providerPaging = paginate(providerBreakdown, providerPage, providerPageSize);
  const dailyPaging = paginate(dailyTrendRows, dailyPage, dailyPageSize);

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
            {availableRanges.map((d) => (
              <button
                key={d}
                className={`users-range-btn ${days === d ? 'active' : ''}`}
                onClick={() => setDays(d)}
              >
                {d}d
              </button>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="card"><div className="card-body"><p>Loading...</p></div></div>
      ) : error ? (
        <div className="card"><div className="card-body"><p className="error-text">{error}</p></div></div>
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
              <div className="users-summary-value">{liveWorkspaces}</div>
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
                    <table className="users-table users-table-compact">
                      <thead>
                        <tr>
                          <th>User</th>
                          <th>Auth</th>
                          <th className="num">Chats</th>
                          <th className="num">Workspaces</th>
                          <th className="num">Memberships</th>
                          <th className="num">Workspace Chats</th>
                          <th className="num">Live/Interrupted</th>
                          <th className="num">Storage</th>
                          <th>Role</th>
                          <th className="num">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {managementPaging.pageItems.map((row) => {
                          const user = row.user;
                          const chatCount = row.usage?.total_requests || 0;
                          const failedCount = (row.usage?.failed_count || 0) + (row.usage?.interrupted_count || 0);
                          const isRowSelf = isSelf(user.id);

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
                                </td>
                                <td className="num">
                                  <div>{formatNumber(chatCount)}</div>
                                  {failedCount > 0 && <div className="users-subnum">{failedCount} fail/int</div>}
                                </td>
                                <td className="num">
                                  <button
                                    type="button"
                                    className="users-link-btn"
                                    onClick={() => setExpandedUserId(expandedUserId === user.id ? null : user.id)}
                                    title="Show workspace operations"
                                  >
                                    {row.ownedWorkspaceCount}
                                  </button>
                                </td>
                                <td className="num">{row.memberWorkspaceCount}</td>
                                <td className="num">{row.workspaceConversationCount}</td>
                                <td className="num">{row.liveWorkspaceCount}/{row.interruptedWorkspaceCount}</td>
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
                                  {isRowSelf ? (
                                    <span className="users-role-badge users-role-admin">{user.role}</span>
                                  ) : (
                                    <select
                                      className="users-role-select"
                                      value={user.role}
                                      disabled={actionLoading === user.id}
                                      onChange={(e) => handleRoleChange(user.id, e.target.value as 'admin' | 'user')}
                                    >
                                      <option value="user">user</option>
                                      <option value="admin">admin</option>
                                    </select>
                                  )}
                                </td>
                                <td className="num">
                                  {isRowSelf ? (
                                    <span className="users-action-muted">-</span>
                                  ) : (
                                    <DeleteConfirmButton
                                      onDelete={() => handleDelete(user.id)}
                                      disabled={actionLoading === user.id}
                                      deleting={actionLoading === user.id}
                                      title="Delete user"
                                    />
                                  )}
                                </td>
                              </tr>
                              {expandedUserId === user.id && (
                                <tr key={`${user.id}-details`} className="users-workspace-detail-row">
                                  <td colSpan={10}>
                                    <WorkspaceRowList
                                      workspaces={row.ownedWorkspaces}
                                      users={users}
                                      onTransfer={handleTransferWorkspace}
                                      onDelete={handleDeleteWorkspace}
                                      emptyMessage="No owned workspaces."
                                      renderMeta={(ws) => {
                                        const state = workspaceStateById[ws.id];
                                        const wsStorage = storageByWorkspaceId[ws.id];
                                        const parts: string[] = [];
                                        parts.push(`chats:${ws.conversation_ids.length}`);
                                        parts.push(`members:${ws.members.length}`);
                                        if (state?.has_live_task) parts.push('live');
                                        if (state?.has_interrupted_task) parts.push('interrupted');
                                        return (
                                          <span className="admin-ws-item-date">
                                            {parts.join(' ')} \u00b7 storage: {typeof wsStorage === 'number' ? formatBytes(wsStorage) : 'not sampled'}
                                          </span>
                                        );
                                      }}
                                    />
                                  </td>
                                </tr>
                              )}
                            </Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <TablePager
                    page={managementPaging.safePage}
                    totalPages={managementPaging.totalPages}
                    totalItems={userStatsRows.length}
                    pageSize={managementPageSize}
                    onPageChange={setManagementPage}
                    onPageSizeChange={(size) => { setManagementPageSize(size); setManagementPage(1); }}
                  />
                </>
              )}
            </div>
          </div>
        </>
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
                    <table className="users-table users-table-compact">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th className="num">Requests</th>
                          <th className="num">Input Tokens</th>
                          <th className="num">Output Tokens</th>
                          <th className="num">Total Tokens</th>
                          <th className="num">Completed</th>
                          <th className="num">Failed</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dailyPaging.pageItems.map((row) => (
                          <tr key={row.date}>
                            <td>{row.date}</td>
                            <td className="num">{formatNumber(row.total_requests)}</td>
                            <td className="num">{formatNumber(row.total_input_tokens)}</td>
                            <td className="num">{formatNumber(row.total_output_tokens)}</td>
                            <td className="num">{formatNumber(row.total_tokens)}</td>
                            <td className="num">{row.completed_count}</td>
                            <td className="num">{row.failed_count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <TablePager
                    page={dailyPaging.safePage}
                    totalPages={dailyPaging.totalPages}
                    totalItems={dailyTrendRows.length}
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
                      <table className="users-table users-table-compact">
                        <thead>
                          <tr>
                            <th>User</th>
                            <th className="num">Requests</th>
                            <th className="num">Input Tokens</th>
                            <th className="num">Output Tokens</th>
                            <th className="num">Total Tokens</th>
                            <th className="num">Completed</th>
                            <th className="num">Failed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {usagePaging.pageItems.map((row) => (
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
                          ))}
                        </tbody>
                      </table>
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
              <div className="card-header"><h3>Failure Heatmap</h3></div>
              <div className="card-body users-compact-card-body">
                {!failureHeatmap ? (
                  <p className="muted-text">No failures in the selected window.</p>
                ) : (
                  <div className="users-heatmap-wrap">
                    <table className="users-heatmap-table">
                      <thead>
                        <tr>
                          <th className="users-heatmap-corner" />
                          {failureHeatmap.modelLabels.map((label, idx) => (
                            <th key={idx} className="users-heatmap-col-header" title={failureHeatmap.models[idx]}>
                              {label}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {failureHeatmap.dates.map((date) => (
                          <tr key={date}>
                            <td className="users-heatmap-row-header">{date.slice(5)}</td>
                            {failureHeatmap.models.map((modelKey) => {
                              const cell = failureHeatmap.lookup.get(`${date}|${modelKey}`);
                              const total = cell ? cell.failed + cell.interrupted : 0;
                              const intensity = failureHeatmap.maxCount > 0 ? total / failureHeatmap.maxCount : 0;
                              return (
                                <td
                                  key={modelKey}
                                  className="users-heatmap-cell"
                                  style={{
                                    backgroundColor: total === 0 ? 'transparent' : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`,
                                  }}
                                  title={total > 0 ? `${date} · ${modelKey}\n${cell!.failed}f · ${cell!.interrupted}i` : ''}
                                >
                                  {total > 0 ? total : ''}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
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
                      <table className="users-table users-table-compact">
                        <thead>
                          <tr>
                            <th>Provider</th>
                            <th>Model</th>
                            <th className="num">Requests</th>
                            <th className="num">Input Tokens</th>
                            <th className="num">Output Tokens</th>
                            <th className="num">Total Tokens</th>
                          </tr>
                        </thead>
                        <tbody>
                          {providerPaging.pageItems.map((row, i) => (
                            <tr key={`${row.provider}-${row.model}-${i}`}>
                              <td>{formatProviderDisplayName(row.provider)}</td>
                              <td>{formatModelDisplayName(row.model, row.provider)}</td>
                              <td className="num">{formatNumber(row.total_requests)}</td>
                              <td className="num">{formatNumber(row.total_input_tokens)}</td>
                              <td className="num">{formatNumber(row.total_output_tokens)}</td>
                              <td className="num">{formatNumber(row.total_tokens)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
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
    </div>
  );
}
