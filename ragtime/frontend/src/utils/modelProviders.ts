export type ProviderConnectionDescriptor = {
  protocolField: string;
  hostField: string;
  portField: string;
  baseUrlField: string;
  defaultProtocol: 'http' | 'https';
  defaultHost: string;
  defaultPort: number;
  defaultBaseUrl: string;
};

export const LLM_PROVIDER_KEYS = [
  'openai',
  'anthropic',
  'ollama',
  'llama_cpp',
  'lmstudio',
  'omlx',
  'github_copilot',
] as const;

export const EMBEDDING_PROVIDER_KEYS = [
  'openai',
  'ollama',
  'llama_cpp',
  'lmstudio',
  'omlx',
] as const;

export const KNOWN_PROVIDER_KEYS = new Set<string>([
  ...LLM_PROVIDER_KEYS,
  ...EMBEDDING_PROVIDER_KEYS,
  'github_models',
]);

export const PROVIDER_CONNECTIONS = {
  ollamaEmbedding: {
    protocolField: 'ollama_protocol',
    hostField: 'ollama_host',
    portField: 'ollama_port',
    baseUrlField: 'ollama_base_url',
    defaultProtocol: 'http',
    defaultHost: 'localhost',
    defaultPort: 11434,
    defaultBaseUrl: 'http://localhost:11434',
  },
  ollamaLlm: {
    protocolField: 'llm_ollama_protocol',
    hostField: 'llm_ollama_host',
    portField: 'llm_ollama_port',
    baseUrlField: 'llm_ollama_base_url',
    defaultProtocol: 'http',
    defaultHost: 'localhost',
    defaultPort: 11434,
    defaultBaseUrl: 'http://localhost:11434',
  },
  llamaCppEmbedding: {
    protocolField: 'llama_cpp_protocol',
    hostField: 'llama_cpp_host',
    portField: 'llama_cpp_port',
    baseUrlField: 'llama_cpp_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 8081,
    defaultBaseUrl: 'http://host.docker.internal:8081',
  },
  llamaCppLlm: {
    protocolField: 'llm_llama_cpp_protocol',
    hostField: 'llm_llama_cpp_host',
    portField: 'llm_llama_cpp_port',
    baseUrlField: 'llm_llama_cpp_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 8080,
    defaultBaseUrl: 'http://host.docker.internal:8080',
  },
  lmstudioEmbedding: {
    protocolField: 'lmstudio_protocol',
    hostField: 'lmstudio_host',
    portField: 'lmstudio_port',
    baseUrlField: 'lmstudio_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 1234,
    defaultBaseUrl: 'http://host.docker.internal:1234',
  },
  lmstudioLlm: {
    protocolField: 'llm_lmstudio_protocol',
    hostField: 'llm_lmstudio_host',
    portField: 'llm_lmstudio_port',
    baseUrlField: 'llm_lmstudio_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 1234,
    defaultBaseUrl: 'http://host.docker.internal:1234',
  },
  omlxEmbedding: {
    protocolField: 'omlx_protocol',
    hostField: 'omlx_host',
    portField: 'omlx_port',
    baseUrlField: 'omlx_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 8000,
    defaultBaseUrl: 'http://host.docker.internal:8000',
  },
  omlxLlm: {
    protocolField: 'llm_omlx_protocol',
    hostField: 'llm_omlx_host',
    portField: 'llm_omlx_port',
    baseUrlField: 'llm_omlx_base_url',
    defaultProtocol: 'http',
    defaultHost: 'host.docker.internal',
    defaultPort: 8000,
    defaultBaseUrl: 'http://host.docker.internal:8000',
  },
} satisfies Record<string, ProviderConnectionDescriptor>;

export function normalizeProviderAlias<T extends string | null | undefined>(provider: T): T extends string ? string : T {
  const value = typeof provider === 'string' ? provider.trim().toLowerCase().replace(/-/g, '_') : provider;
  return (value === 'github_models' ? 'github_copilot' : value) as T extends string ? string : T;
}

export function providersEquivalent(selected?: string | null, actual?: string | null): boolean {
  const selectedNorm = normalizeProviderAlias(selected) || '';
  const actualNorm = normalizeProviderAlias(actual) || '';
  if (selectedNorm === actualNorm) {
    return true;
  }
  return ['openai', 'github_copilot'].includes(selectedNorm) && ['openai', 'github_copilot'].includes(actualNorm);
}

export function toProviderScopedModelKey(provider: string | null | undefined, modelId: string): string {
  const normalizedProvider = normalizeProviderAlias(provider);
  return normalizedProvider ? `${normalizedProvider}::${modelId}` : modelId;
}

export interface ParsedModelIdentifier {
  provider: string | null;
  modelId: string;
}

export function parseScopedModelIdentifier(value: string | null | undefined): ParsedModelIdentifier {
  const raw = (value || '').trim();
  if (!raw) {
    return { provider: null, modelId: '' };
  }
  const delimiterIndex = raw.indexOf('::');
  if (delimiterIndex <= 0) {
    return { provider: null, modelId: raw };
  }
  return {
    provider: raw.slice(0, delimiterIndex).trim() || null,
    modelId: raw.slice(delimiterIndex + 2).trim(),
  };
}

function modelIdVariants(modelId: string): string[] {
  const raw = (modelId || '').trim();
  if (!raw) {
    return [];
  }

  const variants = new Set<string>([raw]);
  if (raw.includes('/')) {
    variants.add(raw.split('/', 2)[1]);
  }
  return [...variants].filter(Boolean);
}

function modelIdentifierMatches(candidate: string, configuredIdentifier: string): boolean {
  const parsedCandidate = parseScopedModelIdentifier(candidate);
  const parsedConfigured = parseScopedModelIdentifier(configuredIdentifier);
  const candidateIds = modelIdVariants(parsedCandidate.modelId || candidate);
  const configuredIds = modelIdVariants(parsedConfigured.modelId || configuredIdentifier);

  if (!candidateIds.some((modelId) => configuredIds.includes(modelId))) {
    return false;
  }

  const candidateProvider = normalizeProviderAlias(parsedCandidate.provider);
  const configuredProvider = normalizeProviderAlias(parsedConfigured.provider);
  if (!candidateProvider || !configuredProvider) {
    return true;
  }

  return providersEquivalent(candidateProvider, configuredProvider);
}

export function modelIdentifierInList(identifier: string | null | undefined, identifiers: string[] | null | undefined): boolean {
  if (!identifier || !identifiers?.length) {
    return false;
  }

  return identifiers.some((candidate) => modelIdentifierMatches(candidate, identifier));
}

export function buildProviderBaseUrl(
  connection: ProviderConnectionDescriptor,
  protocol?: string | null,
  host?: string | null,
  port?: number | null,
): string {
  return `${protocol || connection.defaultProtocol}://${host || connection.defaultHost}:${port || connection.defaultPort}`;
}

export interface ModelPrecedenceLike {
  providers?: string[] | null;
  model_overrides?: Record<string, string> | null;
  family_overrides?: Record<string, string> | null;
}

/**
 * Pick the preferred provider for a model from a candidate set, using:
 *   1. Exact model_id override
 *   2. Family override
 *   3. providers[] ordering
 *   4. null (caller decides default)
 */
export function resolveProviderForModel(
  precedence: ModelPrecedenceLike | null | undefined,
  modelId: string,
  family: string | null | undefined,
  candidateProviders: Iterable<string>,
): string | null {
  const candidates = new Set<string>();
  for (const p of candidateProviders) {
    const norm = normalizeProviderAlias(p);
    if (norm) candidates.add(norm);
  }
  if (!candidates.size) return null;
  if (!precedence) return null;

  const id = (modelId || '').trim();
  const fam = (family || '').trim();
  const overrides = precedence.model_overrides || {};
  const familyOverrides = precedence.family_overrides || {};
  const order = precedence.providers || [];

  if (id && overrides[id]) {
    const c = normalizeProviderAlias(overrides[id]);
    if (c && candidates.has(c)) return c;
  }
  if (fam && familyOverrides[fam]) {
    const c = normalizeProviderAlias(familyOverrides[fam]);
    if (c && candidates.has(c)) return c;
  }
  for (const p of order) {
    const c = normalizeProviderAlias(p);
    if (c && candidates.has(c)) return c;
  }
  return null;
}
