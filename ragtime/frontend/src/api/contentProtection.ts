export type ContentProtectionCoverageMode = 'all_supported_traffic' | 'selected_scopes';
export type ContentProtectionRequirementMode = 'require' | 'inherit';
export type ContentProtectionOverrideMode = 'inherit' | 'always_classify' | 'never_classify';
export type ClassifierBackend = 'jev' | 'llm';
export type JevTransport = 'auto' | 'typesafe' | 'openrouter';
export type Strictness = 'strict' | 'balanced' | 'permissive';

export interface JevConfig {
  transport: JevTransport;
  model: string;
}
export interface ClassifierConfig {
  backend: ClassifierBackend;
  jev: JevConfig;
  llm_model: string | null;
}
export interface ContentCategory {
  id: string;
  name: string;
  description: string;
  includes: string[];
  excludes: string[];
  examples: string[];
  denial_message: string;
  threshold_override: number | null;
  system: boolean;
}
export interface AccessLevel {
  id: string;
  name: string;
  granted_category_ids: string[];
  guidance: string;
}
export interface GroupAccessLevel {
  group_id: string;
  access_level_id: string;
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
  schema_version: 2;
  revision: number;
  enabled: boolean;
  share_with_assistant: boolean;
  classifier: ClassifierConfig;
  strictness: Strictness;
  categories: ContentCategory[];
  access_levels: AccessLevel[];
  group_access_levels: GroupAccessLevel[];
  default_access_level_id: string;
  coverage_mode: ContentProtectionCoverageMode;
  requirements: ContentProtectionRequirement[];
  user_overrides: ContentProtectionUserOverride[];
  legacy_reset?: boolean;
  legacy_was_enabled?: boolean;
}
export const DELETED_ACCESS_LEVEL_MESSAGE = 'This access level was deleted by another admin.';
export interface ContentProtectionCatalogRow {
  id: string;
  name: string;
}
export interface ContentProtectionCatalogClassifierStatus {
  typesafe_key_configured: boolean;
  openrouter_key_configured: boolean;
}
export interface ContentProtectionCatalog {
  users: ContentProtectionCatalogRow[];
  groups: ContentProtectionCatalogRow[];
  tools: ContentProtectionCatalogRow[];
  mcp_routes: ContentProtectionCatalogRow[];
  surfaces: ContentProtectionCatalogRow[];
  classifier_status?: ContentProtectionCatalogClassifierStatus;
}
export interface ContentProtectionPreview {
  required: boolean | null;
  provenance: string | string[] | Record<string, unknown>;
  access_levels: Array<Array<{ id: string; name: string; granted_category_ids: string[] }>>;
  granted_category_ids: string[];
  categories: Array<{ id: string; name: string }>;
  guidance: string[];
  policy_revision: number;
  guidance_revision: string;
  share_with_assistant: boolean;
  prompt_fragment?: string;
}
export interface ContentProtectionPreviewInput {
  user_id?: string;
  baseline?: 'service' | 'public' | 'user' | 'anonymous';
  config?: ContentProtectionConfig;
  surface?: string;
  mcp_route?: string;
  tool_id?: string;
  public?: boolean;
  access_level_ids?: string[] | null;
}
export interface ContentProtectionTestResult {
  verdict?: 'allow' | 'deny';
  code: string;
  reason?: string;
  latency?: number;
  probabilities?: Record<string, number>;
  model?: string;
  usage?: { input_tokens: number; output_tokens: number; cost?: number };
  transport?: string;
}
export interface ContentProtectionReadinessResult {
  code: 'ready' | 'error';
  verdict?: 'allow' | 'deny';
  cases?: Array<{ name: string; verdict: 'allow' | 'deny'; probabilities: Record<string, number> }>;
  model?: string;
  transport?: string;
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
  preview: (payload: ContentProtectionPreviewInput) =>
    request<ContentProtectionPreview>('/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  test: (
    config: ContentProtectionConfig,
    sample: string,
    access_level_ids: string[],
    user_id?: string,
  ) =>
    request<ContentProtectionTestResult>('/test', {
      method: 'POST',
      body: JSON.stringify({ config, sample, access_level_ids, user_id }),
    }),
  readiness: (config: ContentProtectionConfig) =>
    request<ContentProtectionReadinessResult>('/readiness', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  decisions: () => request<{ items: ContentProtectionDecision[] }>('/decisions'),
};
export async function updateContentProtectionConfigSlice(
  mutate: (config: ContentProtectionConfig) => ContentProtectionConfig,
): Promise<ContentProtectionConfig> {
  const saveSlice = async () => {
    const config = await contentProtectionApi.getConfig();
    return contentProtectionApi.saveConfig(config.revision, mutate(config));
  };
  try {
    return await saveSlice();
  } catch (error) {
    if (!(error instanceof ContentProtectionApiError) || error.status !== 409) throw error;
  }
  return saveSlice();
}
export function userOverrideMode(
  config: ContentProtectionConfig,
  userId: string,
): ContentProtectionOverrideMode {
  return config.user_overrides.find((item) => item.user_id === userId)?.mode || 'inherit';
}
export function withUserOverride(
  config: ContentProtectionConfig,
  userId: string,
  mode: ContentProtectionOverrideMode,
): ContentProtectionConfig {
  const user_overrides = config.user_overrides.filter((item) => item.user_id !== userId);
  if (mode !== 'inherit') user_overrides.push({ user_id: userId, mode });
  return { ...config, user_overrides };
}
export function requirementModeFor(
  config: ContentProtectionConfig,
  scopeKind: ContentProtectionRequirement['scope_kind'],
  scopeKey: string,
): ContentProtectionRequirementMode {
  return (
    config.requirements.find((item) => item.scope_kind === scopeKind && item.scope_key === scopeKey)
      ?.mode || 'inherit'
  );
}
export function withRequirement(
  config: ContentProtectionConfig,
  scopeKind: ContentProtectionRequirement['scope_kind'],
  scopeKey: string,
  mode: ContentProtectionRequirementMode,
): ContentProtectionConfig {
  const requirements = config.requirements.filter(
    (item) => item.scope_kind !== scopeKind || item.scope_key !== scopeKey,
  );
  if (mode === 'require') requirements.push({ scope_kind: scopeKind, scope_key: scopeKey, mode });
  return { ...config, requirements };
}
export function withGroupAccessLevel(
  config: ContentProtectionConfig,
  groupId: string,
  accessLevelId: string,
  enabled = true,
): ContentProtectionConfig {
  const group_access_levels = config.group_access_levels.filter(
    (item) => item.group_id !== groupId || item.access_level_id !== accessLevelId,
  );
  if (enabled) group_access_levels.push({ group_id: groupId, access_level_id: accessLevelId });
  return { ...config, group_access_levels };
}
export function withExistingAccessLevel<T>(
  config: ContentProtectionConfig,
  accessLevelId: string,
  mutate: (level: AccessLevel) => T,
): T {
  const level = config.access_levels.find((candidate) => candidate.id === accessLevelId);
  if (!level) throw new Error(DELETED_ACCESS_LEVEL_MESSAGE);
  return mutate(level);
}
export function accessLevelSets(result: Pick<ContentProtectionPreview, 'access_levels'>): string {
  return (
    result.access_levels.map((set) => set.map((level) => level.name).join(', ')).join(' / ') ||
    'None'
  );
}
