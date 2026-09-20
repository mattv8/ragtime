export type ContentProtectionCoverageMode = 'all_supported_traffic' | 'selected_scopes';
export type ContentProtectionRequirementMode = 'require' | 'inherit';
export type ContentProtectionOverrideMode = 'inherit' | 'always_classify' | 'never_classify';

export interface ContentProtectionProfile {
  id: string;
  name: string;
  level: number;
  scope: string;
}

export interface ContentProtectionRequirement {
  scope_kind: 'group' | 'tool' | 'mcp_route' | 'surface';
  scope_key: string;
  mode: ContentProtectionRequirementMode;
}

export interface ContentProtectionUserOverride {
  user_id: string;
  mode: ContentProtectionOverrideMode;
}

export interface ContentProtectionConfig {
  revision: number;
  enabled: boolean;
  classifier_model: string | null;
  coverage_mode: ContentProtectionCoverageMode;
  profiles: ContentProtectionProfile[];
  group_profiles: Array<{ group_id: string; profile_id: string }>;
  requirements: ContentProtectionRequirement[];
  user_overrides: ContentProtectionUserOverride[];
}

export interface ContentProtectionCatalogRow {
  id: string;
  name: string;
}
export interface ContentProtectionCatalog {
  users: ContentProtectionCatalogRow[];
  groups: ContentProtectionCatalogRow[];
  tools: ContentProtectionCatalogRow[];
  mcp_routes: ContentProtectionCatalogRow[];
  surfaces: ContentProtectionCatalogRow[];
}

export interface ContentProtectionPreview {
  required: boolean;
  provenance: string | string[] | Record<string, unknown>;
  /** Each nested array is an independent audience profile set. */
  profiles: ContentProtectionProfile[][];
}

export interface ContentProtectionTestResult {
  verdict?: 'allow' | 'deny';
  code: string;
  reason?: string;
  latency?: number;
}

export interface ContentProtectionDecision {
  id?: string;
  request_id?: string;
  created_at?: string;
  surface?: string;
  direction?: string;
  provenance?: string;
  code?: string;
  verdict?: string;
}

export class ContentProtectionApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'ContentProtectionApiError';
  }
}

function errorMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== 'object') return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (detail && typeof detail === 'object') {
    const { message, code } = detail as { message?: unknown; code?: unknown };
    if (typeof message === 'string' && message.trim()) return message;
    if (typeof code === 'string' && code.trim()) return code;
  }
  return fallback;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body) headers.set('Content-Type', 'application/json');
  if (typeof window !== 'undefined')
    headers.set('X-Ragtime-Browser-Origin', window.location.origin);
  const response = await fetch(`/indexes/content-protection${path}`, {
    ...options,
    headers,
    credentials: 'include',
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ContentProtectionApiError(
      errorMessage(body, `Request failed with status ${response.status}`),
      response.status,
    );
  }
  return response.json() as Promise<T>;
}

export const contentProtectionApi = {
  getConfig: () => request<ContentProtectionConfig>('/config'),
  saveConfig: (expected_revision: number, config: ContentProtectionConfig) =>
    request<ContentProtectionConfig>('/config', {
      method: 'PUT',
      body: JSON.stringify({ expected_revision, config }),
    }),
  getCatalog: () => request<ContentProtectionCatalog>('/catalog'),
  preview: (payload: {
    user_id?: string;
    surface?: string;
    mcp_route?: string;
    tool_id?: string;
    public?: boolean;
  }) =>
    request<ContentProtectionPreview>('/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  test: (config: ContentProtectionConfig, sample: string, profile_ids: string[]) =>
    request<ContentProtectionTestResult>('/test', {
      method: 'POST',
      body: JSON.stringify({ config, sample, profile_ids }),
    }),
  readiness: (config: ContentProtectionConfig) =>
    request<ContentProtectionTestResult>('/readiness', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  decisions: () => request<{ items: ContentProtectionDecision[] }>('/decisions'),
};
