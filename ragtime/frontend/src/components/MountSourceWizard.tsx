import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { Icon } from './Icon';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';
import { ToolWizard } from './ToolWizard';
import { Popover } from './Popover';
import { InlineCopyButton } from './shared/InlineCopyButton';
import { ExternalLink, Info, Trash2, X } from 'lucide-react';
import { api } from '@/api';
import {
  browserPathToSourcePath,
  createBrowserPathDisplayMap,
  mergeBrowserPathDisplayMapFromBrowseResponse,
  normalizeMountBrowserPath,
  resolveSourceDisplayPath,
  sourcePathToBrowserPath,
  type BrowserPathDisplayMap,
} from '@/utils/mountPaths';
import type {
  AuthGroup,
  UserSpaceObjectStorageConfig,
  UserSpaceObjectStorageBucket,
  CloudOAuthProviderStatus,
  UserspaceMountSource,
  ToolConfig,
  ToolType,
  FilesystemConnectionConfig,
  SSHShellConnectionConfig,
  UserCloudOAuthAccount,
  UserCloudOAuthProvider,
  UserDirectoryEntry,
  UserspaceMountSourceType,
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
  source_type: UserspaceMountSourceType | null;
  cloud_oauth_account_id: string | null;
  cloud_access_token: string;
  cloud_refresh_token: string;
  cloud_account_email: string;
  approved_paths: string[];
  access_user_ids: string[];
  access_group_identifiers: string[];
  sync_interval_seconds: number;
};

export function createEmptyMountSourceDraft(): MountSourceDraft {
  return {
    id: null,
    name: '',
    description: '',
    enabled: true,
    tool_config_id: null,
    source_type: null,
    cloud_oauth_account_id: null,
    cloud_access_token: '',
    cloud_refresh_token: '',
    cloud_account_email: '',
    approved_paths: ['.'],
    access_user_ids: [],
    access_group_identifiers: [],
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
    source_type: source.source_type,
    cloud_oauth_account_id: source.oauth_account_id || (source.connection_config && 'oauth_account_id' in source.connection_config ? String(source.connection_config.oauth_account_id || '') : '') || null,
    cloud_access_token: source.connection_config && 'access_token' in source.connection_config ? String(source.connection_config.access_token || '') : '',
    cloud_refresh_token: source.connection_config && 'refresh_token' in source.connection_config ? String(source.connection_config.refresh_token || '') : '',
    cloud_account_email: source.account_email || (source.connection_config && 'account_email' in source.connection_config ? String(source.connection_config.account_email || '') : ''),
    approved_paths: source.approved_paths.length > 0 ? [...source.approved_paths] : ['.'],
    access_user_ids: [...(source.access_user_ids || [])],
    access_group_identifiers: [...(source.access_group_identifiers || [])],
    sync_interval_seconds: source.sync_interval_seconds ?? 30,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MOUNT_TOOL_TYPES: ToolType[] = ['ssh_shell', 'filesystem_indexer'];
const CLOUD_MOUNT_SOURCE_TYPES = ['microsoft_drive', 'google_drive'] as const;

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

function isCloudSourceType(sourceType: UserspaceMountSourceType | null | undefined): sourceType is 'microsoft_drive' | 'google_drive' {
  return sourceType === 'microsoft_drive' || sourceType === 'google_drive';
}

function cloudSourceLabel(sourceType: UserspaceMountSourceType | null | undefined): string {
  if (sourceType === 'microsoft_drive') return 'OneDrive';
  if (sourceType === 'google_drive') return 'Google Drive';
  return 'Cloud Drive';
}

function getCloudOAuthCallbackUrl(): string {
  return new URL('/indexes/userspace/cloud-oauth/callback', window.location.origin).toString();
}

function getCloudSetupInstructions(sourceType: 'microsoft_drive' | 'google_drive', callbackUrl: string): JSX.Element {
  const isMicrosoft = sourceType === 'microsoft_drive';
  const consoleUrl = isMicrosoft
    ? 'https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade'
    : 'https://console.cloud.google.com/apis/credentials';

  return (
    <div style={{ display: 'grid', gap: 8, maxWidth: 300 }}>
      <strong style={{ fontSize: '0.85rem' }}>{isMicrosoft ? 'Set up OneDrive/SharePoint OAuth' : 'Set up Google Drive OAuth'}</strong>
      <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
        Register an OAuth app and configure this redirect URI:
      </span>
      <div className="cloud-oauth-callback-row">
        <code className="cloud-oauth-callback-code">{callbackUrl}</code>
        <InlineCopyButton
          copyText={callbackUrl}
          className="cloud-oauth-callback-copy"
          title="Copy redirect URI"
          ariaLabel="Copy redirect URI"
          copiedTitle="Redirect URI copied"
          copiedAriaLabel="Redirect URI copied"
          iconSize={12}
        />
      </div>
      <a href={consoleUrl} target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>
        Open {isMicrosoft ? 'Microsoft App Registrations' : 'Google API Credentials'}
      </a>
      {isMicrosoft ? (
        <>
          <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
            Set <code>CLOUD_MOUNT_MICROSOFT_TENANT_ID</code> to your Azure Directory tenant ID or primary tenant domain for single-tenant app registrations. Use <code>common</code> or <code>organizations</code> only for multi-tenant apps.
          </span>
          <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
            Add Microsoft Graph delegated permissions: <code>offline_access</code>, <code>User.Read</code>, <code>Files.ReadWrite.All</code>, and <code>Sites.ReadWrite.All</code>. Some tenants require admin consent.
          </span>
        </>
      ) : (
        <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
          Enable the Google Drive API (<code>drive.googleapis.com</code>) in the same Google Cloud project, then add OAuth consent scopes <code>https://www.googleapis.com/auth/drive</code> and <code>https://www.googleapis.com/auth/userinfo.email</code>.
        </span>
      )}
      <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
        Then set the corresponding `CLOUD_MOUNT_*` client id and client secret env vars.
      </span>
    </div>
  );
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

function formatUserDirectoryLabel(user: UserDirectoryEntry | undefined, fallback: string): string {
  if (!user) return fallback;
  return user.display_name ? `${user.display_name} (${user.username})` : user.username;
}

function groupAccessIdentifier(group: AuthGroup): string {
  return group.source_dn || group.key || group.id;
}

function formatAuthGroupLabel(group: AuthGroup | undefined, fallback: string): string {
  if (!group) return fallback;
  const provider = group.provider === 'ldap' ? 'LDAP' : 'Local';
  return `${group.display_name || group.key} (${provider})`;
}

// ---------------------------------------------------------------------------
// Wizard steps
// ---------------------------------------------------------------------------

type MountSourceWizardStep = 'select_tool' | 'mount_details' | 'access_control' | 'review';

const WIZARD_STEPS: MountSourceWizardStep[] = ['select_tool', 'mount_details', 'access_control', 'review'];
const EDIT_WIZARD_STEPS: MountSourceWizardStep[] = ['mount_details', 'access_control', 'review'];

function getStepTitle(step: MountSourceWizardStep): string {
  switch (step) {
    case 'select_tool': return 'Select Backing Tool';
    case 'mount_details': return 'Mount Configuration';
    case 'access_control': return 'Access Control';
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
  const [browserPathDisplayMap, setBrowserPathDisplayMap] = useState<BrowserPathDisplayMap>(createBrowserPathDisplayMap());
  const [stagedDirectories, setStagedDirectories] = useState<string[]>([]);
  const [editingName, setEditingName] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Tool selection state
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [loadingTools, setLoadingTools] = useState(true);
  const [showToolWizard, setShowToolWizard] = useState(false);
  const [newToolType, setNewToolType] = useState<ToolType | undefined>(undefined);
  const [cloudProviderStatuses, setCloudProviderStatuses] = useState<CloudOAuthProviderStatus[]>([]);
  const [cloudOAuthAccounts, setCloudOAuthAccounts] = useState<UserCloudOAuthAccount[]>([]);
  const [savingCloudProvider, setSavingCloudProvider] = useState<UserCloudOAuthProvider | null>(null);
  const [deletingCloudAccountId, setDeletingCloudAccountId] = useState<string | null>(null);
  const [availableUsers, setAvailableUsers] = useState<UserDirectoryEntry[]>([]);
  const [authGroups, setAuthGroups] = useState<AuthGroup[]>([]);

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

  const loadCloudProviderStatuses = useCallback(async () => {
    try {
      const statuses = await api.listCloudOAuthProviders();
      setCloudProviderStatuses(statuses);
    } catch {
      setCloudProviderStatuses([]);
    }
  }, []);

  useEffect(() => { void loadCloudProviderStatuses(); }, [loadCloudProviderStatuses]);

  const loadCloudOAuthAccounts = useCallback(async () => {
    try {
      const accounts = await api.listUserCloudOAuthAccounts();
      setCloudOAuthAccounts(accounts);
    } catch {
      setCloudOAuthAccounts([]);
    }
  }, []);

  useEffect(() => { void loadCloudOAuthAccounts(); }, [loadCloudOAuthAccounts]);

  const loadAccessOptions = useCallback(async () => {
    try {
      const [users, groups] = await Promise.all([
        api.listUsersDirectory(),
        api.listAuthGroups(),
      ]);
      setAvailableUsers(users);
      setAuthGroups(groups);
    } catch {
      setAvailableUsers([]);
      setAuthGroups([]);
    }
  }, []);

  useEffect(() => { void loadAccessOptions(); }, [loadAccessOptions]);

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;
      if (event.data?.type === 'ragtime-cloud-oauth-complete') {
        void loadCloudOAuthAccounts();
      } else if (event.data?.type === 'ragtime-cloud-oauth-error') {
        setError(typeof event.data.message === 'string' ? event.data.message : 'Cloud OAuth failed');
      }
    };
    window.addEventListener('message', listener);
    return () => window.removeEventListener('message', listener);
  }, [loadCloudOAuthAccounts]);

  // Auto-scroll active step into view
  useEffect(() => {
    const activeStep = progressRef.current?.querySelector('.wizard-step.active');
    if (activeStep) {
      activeStep.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentStep]);

  const selectedTool = tools.find((t) => t.id === draft.tool_config_id) ?? null;
  const selectedCloudSource = isCloudSourceType(draft.source_type) ? draft.source_type : null;
  const cloudProviderConfigured = useMemo(() => {
    const entries = cloudProviderStatuses.map((status) => [status.provider, status.configured] as const);
    return new Map<UserCloudOAuthProvider, boolean>(entries);
  }, [cloudProviderStatuses]);
  const selectedCloudAccount = cloudOAuthAccounts.find((account) => account.id === draft.cloud_oauth_account_id) ?? null;
  const selectedProviderAccounts = selectedCloudSource
    ? cloudOAuthAccounts.filter((account) => account.provider === selectedCloudSource)
    : [];
  const usersById = useMemo(() => new Map(availableUsers.map((user) => [user.id, user])), [availableUsers]);
  const authGroupsByIdentifier = useMemo(() => new Map(authGroups.map((group) => [groupAccessIdentifier(group), group])), [authGroups]);
  const addableUsers = availableUsers.filter((user) => !draft.access_user_ids.includes(user.id));
  const addableGroups = authGroups.filter((group) => !draft.access_group_identifiers.includes(groupAccessIdentifier(group)));
  const isSSHSource = selectedTool?.tool_type === 'ssh_shell' || existingSource?.source_type === 'ssh';
  const isSyncIntervalSource = isSSHSource
    || selectedCloudSource != null
    || existingSource?.source_type === 'microsoft_drive'
    || existingSource?.source_type === 'google_drive';
  const cloudOAuthCallbackUrl = getCloudOAuthCallbackUrl();

  useEffect(() => {
    setBrowserPathDisplayMap(createBrowserPathDisplayMap());
  }, [selectedCloudSource, draft.cloud_oauth_account_id, draft.tool_config_id, draft.id]);

  useEffect(() => {
    if (!selectedCloudSource || draft.cloud_oauth_account_id || selectedProviderAccounts.length !== 1) {
      return;
    }
    const [account] = selectedProviderAccounts;
    setDraft((current) => ({
      ...current,
      cloud_oauth_account_id: account.id,
      cloud_account_email: account.account_email || account.account_name || '',
    }));
  }, [selectedCloudSource, draft.cloud_oauth_account_id, selectedProviderAccounts]);

  const handleConnectCloudProvider = useCallback(async (provider: UserCloudOAuthProvider) => {
    if (!cloudProviderConfigured.get(provider)) {
      setError(`${cloudSourceLabel(provider)} OAuth is not configured.`);
      return;
    }
    setSavingCloudProvider(provider);
    setError(null);
    try {
      const redirectUri = getCloudOAuthCallbackUrl();
      const response = await api.startUserCloudOAuth({ provider, redirect_uri: redirectUri });
      window.open(response.auth_url, 'ragtime-cloud-oauth', 'popup,width=720,height=820');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start cloud OAuth');
    } finally {
      setSavingCloudProvider(null);
    }
  }, [cloudProviderConfigured]);

  const handleDisconnectCloudAccount = useCallback(async (account: UserCloudOAuthAccount) => {
    setDeletingCloudAccountId(account.id);
    setError(null);
    try {
      await api.disconnectUserCloudOAuth(account.id);
      setCloudOAuthAccounts((current) => current.filter((item) => item.id !== account.id));
      setDraft((current) => current.cloud_oauth_account_id === account.id
        ? { ...current, cloud_oauth_account_id: null, cloud_account_email: '', cloud_access_token: '', cloud_refresh_token: '' }
        : current
      );
      await loadCloudOAuthAccounts();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove cloud OAuth account');
    } finally {
      setDeletingCloudAccountId(null);
    }
  }, [loadCloudOAuthAccounts]);

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
        return draft.tool_config_id !== null || selectedCloudSource !== null;
      case 'mount_details':
        return draft.name.trim().length > 0 && draft.approved_paths.length > 0 && (
          !selectedCloudSource || Boolean(draft.cloud_oauth_account_id || draft.cloud_access_token.trim())
        );
      case 'access_control':
        return true;
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

  const buildCloudMountSourceRequest = useCallback((path: string) => {
    if (!selectedCloudSource) {
      return null;
    }
    return {
      source_type: selectedCloudSource,
      oauth_account_id: draft.cloud_oauth_account_id,
      connection_config: {
        provider: selectedCloudSource,
        auth_mode: 'oauth' as const,
        oauth_account_id: draft.cloud_oauth_account_id || undefined,
        access_token: draft.cloud_access_token.trim() || undefined,
        refresh_token: draft.cloud_refresh_token.trim() || undefined,
        account_email: draft.cloud_account_email.trim() || undefined,
      },
      path,
    };
  }, [draft.cloud_access_token, draft.cloud_account_email, draft.cloud_oauth_account_id, draft.cloud_refresh_token, selectedCloudSource]);

  const updateCloudPathDisplayMap = useCallback((result: { path: string; entries: Array<{ name: string; path: string; is_dir: boolean }> }) => {
    if (!selectedCloudSource) {
      return;
    }
    setBrowserPathDisplayMap((current) => {
      return mergeBrowserPathDisplayMapFromBrowseResponse(current, result, {
        sourceType: selectedCloudSource,
        fallbackDriveName: cloudSourceLabel(selectedCloudSource),
      });
    });
  }, [selectedCloudSource]);

  const displayApprovedPath = useCallback((sourcePath: string): string => {
    return resolveSourceDisplayPath(sourcePath, browserPathDisplayMap, {
      sourceType: selectedCloudSource,
      fallbackDriveName: selectedCloudSource ? cloudSourceLabel(selectedCloudSource) : null,
    });
  }, [browserPathDisplayMap, selectedCloudSource]);

  const addApprovedPathFromBrowserPath = useCallback((selectedPath: string) => {
    const normalizedBrowserPath = normalizeMountBrowserPath(selectedPath);
    if (selectedCloudSource && normalizedBrowserPath === '/') {
      return;
    }
    const nextPath = browserPathToSourcePath(normalizedBrowserPath);
    setDraft((current) => {
      const existing = current.approved_paths.filter((item) => !(item === '.' && nextPath !== '.'));
      if (existing.includes(nextPath)) {
        return existing.length === current.approved_paths.length ? current : { ...current, approved_paths: existing };
      }
      return { ...current, approved_paths: [...existing, nextPath].sort((a, b) => a.localeCompare(b)) };
    });
  }, [selectedCloudSource]);

  const handleRemoveApprovedPath = useCallback((path: string) => {
    setDraft((current) => {
      const remaining = current.approved_paths.filter((item) => item !== path);
      return { ...current, approved_paths: remaining.length > 0 ? remaining : selectedCloudSource ? [] : ['.'] };
    });
  }, [selectedCloudSource]);

  const browseMountSourcePath = useCallback(async (path: string) => {
    if (draft.id) {
      const result = await api.browseUserspaceMountSource(draft.id, { path });
      updateCloudPathDisplayMap(result);
      return result;
    }
    if (selectedCloudSource) {
      if (!draft.cloud_oauth_account_id && !draft.cloud_access_token.trim()) {
        return { path, entries: [], error: 'Connect or select an OAuth account before browsing remote folders.' };
      }
      const request = buildCloudMountSourceRequest(path);
      if (!request) {
        return { path, entries: [], error: 'Select a cloud source first.' };
      }
      const result = await api.browseCloudMountSource(request);
      updateCloudPathDisplayMap(result);
      return result;
    }
    if (draft.tool_config_id) {
      return api.browseToolConfig(draft.tool_config_id, { path });
    }
    return { path, entries: [], error: 'Select a backing tool first.' };
  }, [buildCloudMountSourceRequest, draft.cloud_access_token, draft.cloud_oauth_account_id, draft.id, draft.tool_config_id, selectedCloudSource, updateCloudPathDisplayMap]);

  const handleStageDirectory = useCallback((path: string) => {
    const normalizedPath = normalizeMountBrowserPath(path);
    setStagedDirectories((prev) => prev.includes(normalizedPath) ? prev : [...prev, normalizedPath]);
    if (!selectedCloudSource) {
      return;
    }
    const request = buildCloudMountSourceRequest(normalizedPath);
    if (!request) {
      setError('Select a cloud source before creating a remote folder.');
      return;
    }
    void (async () => {
      try {
        if (draft.id) {
          await api.createUserspaceMountSourceDirectory(draft.id, { path: normalizedPath });
        } else {
          await api.createCloudMountSourceDirectory(request);
        }
      } catch (err) {
        const sourcePath = browserPathToSourcePath(normalizedPath);
        setError(err instanceof Error ? err.message : 'Failed to create remote folder');
        setStagedDirectories((current) => current.filter((item) => item !== normalizedPath));
        setDraft((current) => ({
          ...current,
          approved_paths: current.approved_paths.filter((item) => item !== sourcePath),
        }));
      }
    })();
  }, [buildCloudMountSourceRequest, draft.id, selectedCloudSource]);

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
      const accessUserIds = Array.from(new Set(draft.access_user_ids.map((value) => value.trim()).filter(Boolean)));
      const accessGroupIdentifiers = Array.from(new Set(draft.access_group_identifiers.map((value) => value.trim()).filter(Boolean)));
      if (selectedCloudSource && approvedPaths.length === 0) {
        throw new Error('Select at least one drive or folder for this cloud mount source.');
      }

      if (draft.id) {
        const accountEmail = selectedCloudAccount?.account_email || selectedCloudAccount?.account_name || draft.cloud_account_email.trim() || undefined;
        // Update existing
        const saved = await api.updateUserspaceMountSource(draft.id, {
          name: draft.name.trim(),
          description: null,
          enabled: true,
          connection_config: selectedCloudSource ? {
            provider: selectedCloudSource,
            auth_mode: 'oauth',
            oauth_account_id: draft.cloud_oauth_account_id || undefined,
            access_token: draft.cloud_access_token.trim() || undefined,
            refresh_token: draft.cloud_refresh_token.trim() || undefined,
            account_email: accountEmail,
          } : undefined,
          approved_paths: selectedCloudSource ? approvedPaths : approvedPaths.length > 0 ? approvedPaths : ['.'],
          access_user_ids: accessUserIds,
          access_group_identifiers: accessGroupIdentifiers,
          sync_interval_seconds: draft.sync_interval_seconds,
        });
        onSaved(saved);
      } else if (selectedCloudSource) {
        const accountEmail = selectedCloudAccount?.account_email || selectedCloudAccount?.account_name || draft.cloud_account_email.trim() || undefined;
        const saved = await api.createUserspaceMountSource({
          name: draft.name.trim(),
          description: null,
          enabled: true,
          source_type: selectedCloudSource,
          connection_config: {
            provider: selectedCloudSource,
            auth_mode: 'oauth',
            oauth_account_id: draft.cloud_oauth_account_id || undefined,
            access_token: draft.cloud_access_token.trim() || undefined,
            refresh_token: draft.cloud_refresh_token.trim() || undefined,
            account_email: accountEmail,
          },
          approved_paths: approvedPaths,
          access_user_ids: accessUserIds,
          access_group_identifiers: accessGroupIdentifiers,
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
          access_user_ids: accessUserIds,
          access_group_identifiers: accessGroupIdentifiers,
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
          Choose an existing SSH/filesystem tool or create a cloud drive source.
        </p>

        <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
          {CLOUD_MOUNT_SOURCE_TYPES.map((sourceType) => {
            const isConfigured = cloudProviderConfigured.get(sourceType) === true;
            const handleCloudSourceSelect = () => {
              if (!isConfigured) {
                return;
              }
              setDraft((d) => ({
                ...d,
                source_type: sourceType,
                tool_config_id: null,
                cloud_oauth_account_id: d.source_type === sourceType ? d.cloud_oauth_account_id : null,
                cloud_account_email: d.source_type === sourceType ? d.cloud_account_email : '',
                approved_paths: d.source_type === sourceType ? d.approved_paths : [],
                name: d.name || deduplicateName(cloudSourceLabel(sourceType), namesForDedup),
              }));
            };

            return (
              <div
                key={sourceType}
                className={`tool-type-option ${draft.source_type === sourceType ? 'selected' : ''}${!isConfigured ? ' disabled' : ''}`}
                role="button"
                tabIndex={isConfigured ? 0 : -1}
                aria-disabled={!isConfigured}
                onClick={handleCloudSourceSelect}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleCloudSourceSelect();
                  }
                }}
                title={isConfigured ? undefined : `Configure ${cloudSourceLabel(sourceType)} OAuth client ID and secret to enable this source`}
              >
                <div className="tool-type-option-icon">
                  <Icon name={sourceType === 'microsoft_drive' ? 'folder' : 'harddrive'} size={24} />
                </div>
                <div>
                  <span className="tool-type-option-name">{cloudSourceLabel(sourceType)}</span>
                  {isConfigured ? (
                    <span className="tool-type-option-desc">Global cloud source</span>
                  ) : (
                    <span className="tool-type-option-desc" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      OAuth app not configured
                      <Popover
                        trigger="click"
                        position="bottom"
                        content={getCloudSetupInstructions(sourceType, cloudOAuthCallbackUrl)}
                      >
                        <span
                          role="button"
                          tabIndex={0}
                          aria-label={`How to configure ${cloudSourceLabel(sourceType)} OAuth`}
                          style={{ display: 'inline-flex', alignItems: 'center', cursor: 'pointer' }}
                        >
                          <Info size={12} />
                        </span>
                      </Popover>
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

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
                onClick={() => setDraft((d) => ({ ...d, tool_config_id: tool.id, source_type: null, approved_paths: d.approved_paths.length > 0 ? d.approved_paths : ['.'] }))}
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

      {selectedCloudSource && (
        <div style={{ display: 'grid', gap: 12, padding: '12px 14px', background: 'var(--color-bg-tertiary)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Icon name={selectedCloudSource === 'microsoft_drive' ? 'folder' : 'harddrive'} size={16} />
            <span className="muted" style={{ fontSize: '0.8rem' }}>{cloudSourceLabel(selectedCloudSource)}</span>
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
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <strong>OAuth Account</strong>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => void handleConnectCloudProvider(selectedCloudSource)}
                disabled={savingCloudProvider === selectedCloudSource || !cloudProviderConfigured.get(selectedCloudSource)}
                title={cloudProviderConfigured.get(selectedCloudSource) ? undefined : `Configure ${cloudSourceLabel(selectedCloudSource)} OAuth client ID and secret to enable account connection`}
              >
                <ExternalLink size={12} />
                {savingCloudProvider === selectedCloudSource
                  ? 'Connecting...'
                  : selectedCloudSource === 'microsoft_drive' ? 'Connect OneDrive' : 'Connect Google'}
              </button>
            </div>

            {selectedProviderAccounts.length > 0 ? (
              <div style={{ display: 'grid', gap: 8 }}>
                {selectedProviderAccounts.map((account) => (
                  <div
                    key={account.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '8px 10px',
                      border: '1px solid var(--color-border)',
                      borderRadius: 6,
                      background: draft.cloud_oauth_account_id === account.id ? 'var(--color-bg-secondary)' : 'var(--color-bg-primary)',
                    }}
                  >
                    <input
                      type="radio"
                      checked={draft.cloud_oauth_account_id === account.id}
                      onChange={() => setDraft((d) => ({
                        ...d,
                        cloud_oauth_account_id: account.id,
                        cloud_account_email: account.account_email || account.account_name || '',
                        cloud_access_token: '',
                        cloud_refresh_token: '',
                      }))}
                    />
                    <span>{account.account_email || account.account_name || 'Connected account'}</span>
                    <span style={{ marginLeft: 'auto' }} />
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => void handleDisconnectCloudAccount(account)}
                      disabled={deletingCloudAccountId === account.id}
                      title="Remove OAuth account"
                      aria-label={`Remove ${account.account_email || account.account_name || 'OAuth account'}`}
                      style={{ padding: '5px 7px' }}
                    >
                      {deletingCloudAccountId === account.id ? 'Removing...' : <Trash2 size={12} />}
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="field-help" style={{ margin: 0 }}>
                Connect an account to create this global mount source.
              </p>
            )}

            {!draft.cloud_oauth_account_id && draft.cloud_account_email && (
              <p className="field-help" style={{ margin: 0 }}>
                Current saved account: {draft.cloud_account_email}
              </p>
            )}
          </div>
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
            cacheKey={`mount-source-wizard:${draft.id ?? draft.tool_config_id ?? selectedCloudSource ?? 'draft'}:${draft.cloud_oauth_account_id ?? 'no-account'}`}
            stagedDirectories={stagedDirectories}
            onStageDirectory={handleStageDirectory}
            emptyMessage="No directories found"
            onSelectPath={(selectedPath) => {
              const normalizedPath = normalizeMountBrowserPath(selectedPath);
              setBrowserPath(normalizedPath);
              addApprovedPathFromBrowserPath(normalizedPath);
            }}
            onBrowsePath={browseMountSourcePath}
            pathDisplayMap={selectedCloudSource ? browserPathDisplayMap : undefined}
            pathDisplayOptions={selectedCloudSource ? { sourceType: selectedCloudSource, fallbackDriveName: cloudSourceLabel(selectedCloudSource) } : undefined}
            canSelectPath={(path) => !selectedCloudSource || normalizeMountBrowserPath(path) !== '/'}
            cannotSelectPathMessage={selectedCloudSource ? 'Select a drive or folder first' : undefined}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <span className="field-help" style={{ margin: 0 }}>
              Selecting a folder adds it to the allowed roots.
            </span>
            {draft.approved_paths.map((path) => (
              <div key={path} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: 6, fontSize: '0.85rem' }}>
                <code>{selectedCloudSource ? displayApprovedPath(path) : sourcePathToBrowserPath(path)}</code>
                <button type="button" className="btn btn-secondary" onClick={() => handleRemoveApprovedPath(path)} style={{ padding: '4px' }}>
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sync interval slider for mount sources with background sync */}
      {isSyncIntervalSource && (
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
            Lower values increase responsiveness but use more resources.
          </p>
        </div>
      )}
    </div>
  );

  const renderAccessControl = () => (
    <div className="wizard-step-content" style={{ display: 'grid', gap: 16 }}>
      <p className="field-help" style={{ margin: 0 }}>
        Choose which users and auth groups can attach this source to workspaces. Admins can always manage and mount sources. With no ACL entries, only admins can mount this source.
      </p>

      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(2, 1fr)' }}>
        <div style={{ display: 'grid', gap: 10, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <strong>Users</strong>
            <span style={{ marginLeft: 'auto' }} />
            <select
              className="form-input"
              value=""
              onChange={(event) => {
                const userId = event.target.value;
                if (!userId) return;
                setDraft((current) => ({
                  ...current,
                  access_user_ids: current.access_user_ids.includes(userId)
                    ? current.access_user_ids
                    : [...current.access_user_ids, userId].sort((left, right) => (
                      formatUserDirectoryLabel(usersById.get(left), left).localeCompare(formatUserDirectoryLabel(usersById.get(right), right))
                    )),
                }));
              }}
              style={{ width: 'auto', minWidth: 240 }}
            >
              <option value="">Add user...</option>
              {addableUsers.map((user) => (
                <option key={user.id} value={user.id}>{formatUserDirectoryLabel(user, user.id)}</option>
              ))}
            </select>
          </div>

          {draft.access_user_ids.length > 0 ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {draft.access_user_ids.map((userId) => (
                <div key={userId} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', border: '1px solid var(--color-border)', borderRadius: 6 }}>
                  <span>{formatUserDirectoryLabel(usersById.get(userId), userId)}</span>
                  <span className="muted" style={{ fontSize: '0.8rem' }}>Can mount</span>
                  <span style={{ marginLeft: 'auto' }} />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setDraft((current) => ({
                      ...current,
                      access_user_ids: current.access_user_ids.filter((id) => id !== userId),
                    }))}
                    title="Remove user access"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="field-help" style={{ margin: 0 }}>No explicit users selected.</p>
          )}
        </div>

        <div style={{ display: 'grid', gap: 10, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <strong>Auth Groups</strong>
            <span style={{ marginLeft: 'auto' }} />
            <select
              className="form-input"
              value=""
              onChange={(event) => {
                const groupIdentifier = event.target.value;
                if (!groupIdentifier) return;
                setDraft((current) => ({
                  ...current,
                  access_group_identifiers: current.access_group_identifiers.includes(groupIdentifier)
                    ? current.access_group_identifiers
                    : [...current.access_group_identifiers, groupIdentifier].sort((left, right) => (
                      formatAuthGroupLabel(authGroupsByIdentifier.get(left), left).localeCompare(formatAuthGroupLabel(authGroupsByIdentifier.get(right), right))
                    )),
                }));
              }}
              style={{ width: 'auto', minWidth: 240 }}
            >
              <option value="">Add auth group...</option>
              {addableGroups.map((group) => {
                const identifier = groupAccessIdentifier(group);
                return <option key={identifier} value={identifier}>{formatAuthGroupLabel(group, identifier)}</option>;
              })}
            </select>
          </div>

          {draft.access_group_identifiers.length > 0 ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {draft.access_group_identifiers.map((groupIdentifier) => (
                <div key={groupIdentifier} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', border: '1px solid var(--color-border)', borderRadius: 6 }}>
                  <span>{formatAuthGroupLabel(authGroupsByIdentifier.get(groupIdentifier), groupIdentifier)}</span>
                  <span className="muted" style={{ fontSize: '0.8rem' }}>Can mount</span>
                  <span style={{ marginLeft: 'auto' }} />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setDraft((current) => ({
                      ...current,
                      access_group_identifiers: current.access_group_identifiers.filter((id) => id !== groupIdentifier),
                    }))}
                    title="Remove group access"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="field-help" style={{ margin: 0 }}>No auth groups selected.</p>
          )}
        </div>
      </div>
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
              {selectedCloudSource ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Icon name={selectedCloudSource === 'microsoft_drive' ? 'folder' : 'harddrive'} size={14} />
                  {cloudSourceLabel(selectedCloudSource)}
                  <span className="muted" style={{ fontSize: '0.8rem' }}>
                    ({selectedCloudAccount?.account_email || selectedCloudAccount?.account_name || draft.cloud_account_email || 'connected account'})
                  </span>
                </span>
              ) : selectedTool ? (
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
                  {selectedCloudSource ? displayApprovedPath(path) : sourcePathToBrowserPath(path)}
                </code>
              ))}
            </td>
          </tr>
          <tr>
            <td className="review-label">Access</td>
            <td>
              {draft.access_user_ids.length === 0 && draft.access_group_identifiers.length === 0 ? (
                <span className="muted">Admins only</span>
              ) : (
                <div style={{ display: 'grid', gap: 4 }}>
                  {draft.access_user_ids.length > 0 && (
                    <span>
                      Users:{' '}
                      {draft.access_user_ids.map((userId) => formatUserDirectoryLabel(usersById.get(userId), userId)).join(', ')}
                    </span>
                  )}
                  {draft.access_group_identifiers.length > 0 && (
                    <span>
                      Groups:{' '}
                      {draft.access_group_identifiers.map((groupIdentifier) => formatAuthGroupLabel(authGroupsByIdentifier.get(groupIdentifier), groupIdentifier)).join(', ')}
                    </span>
                  )}
                </div>
              )}
            </td>
          </tr>
          {isSyncIntervalSource && (
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
      case 'access_control': return renderAccessControl();
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
