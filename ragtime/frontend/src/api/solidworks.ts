import type { SSHShellConnectionConfig } from '../types/api';

export type SolidworksInstanceMode = 'background' | 'attach';
export interface SolidworksOptions {
  solidworks_version?: number | null;
  instance_mode?: SolidworksInstanceMode;
  max_instances?: number;
  idle_shutdown_seconds?: number;
  allow_scripts?: boolean;
}
export type SolidworksConnectionConfig = SSHShellConnectionConfig &
  SolidworksOptions & {
    pdm_tool_id?: string | null;
    pdm_approved_folders?: string[];
  };
export interface SolidworksHostArtifact {
  available: boolean;
  host_version: string | null;
  protocol: number | null;
  sha256: string | null;
  size_bytes: number | null;
}
export interface SolidworksHostStatus {
  reachable: boolean;
  installed: boolean;
  running: boolean;
  host_version: string | null;
  expected_host_version: string | null;
  update_available: boolean;
  account: { name: string; is_administrator: boolean } | null;
  session: { interactive: boolean | null } | null;
  solidworks: { installed_versions: number[]; instances: object[] } | null;
  pdm?: {
    client_installed: boolean;
    client_version: string | null;
    views: { vault: string; root: string; logged_in: boolean; user: string | null }[];
  } | null;
  error: { code: string; message: string } | null;
  /** Non-blocking note, e.g. an uninstall that could not delete locked files. */
  warning?: string | null;
}
export const SOLIDWORKS_DEFAULT_OPTIONS: Required<SolidworksOptions> = {
  solidworks_version: null,
  instance_mode: 'background',
  max_instances: 1,
  idle_shutdown_seconds: 900,
  allow_scripts: false,
};
export const SOLIDWORKS_DEFAULT_TIMEOUT_SECONDS = 100;

export interface PdmRights {
  read: boolean;
  add_files: boolean;
  check_out: boolean;
  add_folders: boolean;
}

export interface PdmRightsItem {
  path: string;
  checked: boolean;
  exists: boolean | null;
  rights: PdmRights | null;
  write_ready: boolean;
}

export interface PdmRightsResponse {
  checked: boolean;
  user: string | null;
  root: PdmRightsItem | null;
  folders: PdmRightsItem[];
  error: { code: string; message: string } | null;
}

export interface PdmFolderChild {
  name: string;
  vault_path: string;
  rights: PdmRights | null;
  write_ready: boolean;
}

export interface PdmFoldersResponse {
  items: PdmFolderChild[];
  offset: number;
  next_offset: number | null;
  total: number;
  snapshot: string | null;
  error: { code: string; message: string } | null;
}

/** An HTTP error from the SolidWorks API. `status` lets callers tell e.g. the artifact 503 apart. */
export class SolidworksApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'SolidworksApiError';
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body) headers.set('Content-Type', 'application/json');
  if (typeof window !== 'undefined')
    headers.set('X-Ragtime-Browser-Origin', window.location.origin);
  const response = await fetch(`/indexes/solidworks${path}`, {
    ...options,
    headers,
    credentials: 'include',
  });
  if (!response.ok) {
    const body: { detail?: unknown } = await response.json().catch(() => ({}));
    const detail = body.detail;
    if (detail && typeof detail === 'object') {
      const { message, code } = detail as { message?: unknown; code?: unknown };
      throw new SolidworksApiError(
        typeof message === 'string' ? message : `Request failed with status ${response.status}`,
        response.status,
        typeof code === 'string' ? code : undefined,
      );
    }
    throw new SolidworksApiError(
      typeof detail === 'string' ? detail : `Request failed with status ${response.status}`,
      response.status,
    );
  }
  return response.json() as Promise<T>;
}

const hostPath = (toolId: string) => `/tools/${encodeURIComponent(toolId)}/host`;
export const solidworksApi = {
  checkPdmRights: (
    toolId: string,
    body: { pdm_tool_id: string; folders: string[] },
    signal?: AbortSignal,
  ) =>
    request<PdmRightsResponse>(`/tools/${encodeURIComponent(toolId)}/pdm/rights`, {
      method: 'POST',
      body: JSON.stringify(body),
      signal,
    }),
  listPdmFolders: (
    toolId: string,
    body: { pdm_tool_id: string; path: string; offset: number; snapshot?: string | null },
    signal?: AbortSignal,
  ) =>
    request<PdmFoldersResponse>(`/tools/${encodeURIComponent(toolId)}/pdm/folders`, {
      method: 'POST',
      body: JSON.stringify(body),
      signal,
    }),
  getHostArtifact: (signal?: AbortSignal) =>
    request<SolidworksHostArtifact>('/host/artifact', { signal }),
  getHostStatus: (toolId: string, signal?: AbortSignal) =>
    request<SolidworksHostStatus>(hostPath(toolId), { signal }),
  installHost: (toolId: string) =>
    request<SolidworksHostStatus>(`${hostPath(toolId)}/install`, { method: 'POST' }),
  uninstallHost: (toolId: string) =>
    request<SolidworksHostStatus>(`${hostPath(toolId)}/uninstall`, { method: 'POST' }),
  hostDownloadUrl: () => '/indexes/solidworks/host/download',
};

const guidance: Record<string, { title: string; hint: string }> = {
  ssh_unreachable: {
    title: 'Ragtime could not sign in over SSH.',
    hint: 'Check that the machine is on, OpenSSH Server is running, and the credentials from the connection step are still valid.',
  },
  host_not_running: {
    title: 'The host did not answer.',
    hint: 'Install the host in the panel above and make sure this account is logged in to the machine.',
  },
  protocol_mismatch: {
    title: 'The host and Ragtime do not match.',
    hint: 'Update the host in the panel above.',
  },
  no_interactive_session: {
    title: 'No desktop session.',
    hint: 'Log in to the machine as this account; a disconnected remote desktop session is enough.',
  },
  version_not_installed: {
    title: 'That SolidWorks version is not installed.',
    hint: 'Choose another version above or install the pinned one.',
  },
  version_mismatch: {
    title: 'The open SolidWorks is a different version.',
    hint: 'Open the pinned version, or change the version above.',
  },
  solidworks_not_running: {
    title: 'SolidWorks is not open.',
    hint: "Attach mode needs SolidWorks already open in this account's session. Open it, or switch to Background instance.",
  },
  solidworks_start_failed: {
    title: 'SolidWorks did not start.',
    hint: 'Check that SolidWorks and its licence server work for this account. The host log is %LOCALAPPDATA%\\RagtimeSolidWorks\\logs\\host.log.',
  },
  instance_timeout: {
    title: 'The call timed out.',
    hint: 'Starting SolidWorks takes about 10 seconds and can take longer. Raise Max Timeout or try again.',
  },
  instance_limit: {
    title: 'All instances are busy.',
    hint: 'Raise Max background instances or try again in a moment.',
  },
};
const guidanceAliases: Record<string, string> = {
  ssh_auth_failed: 'ssh_unreachable',
  operation_timeout: 'instance_timeout',
};
export function getSolidworksErrorGuidance(code: string | undefined): {
  title: string;
  hint: string;
} | null {
  if (!code) return null;
  return guidance[guidanceAliases[code] ?? code] ?? null;
}
