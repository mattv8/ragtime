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
  'github_copilot',
] as const;

export const EMBEDDING_PROVIDER_KEYS = [
  'openai',
  'ollama',
  'llama_cpp',
  'lmstudio',
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

export function buildProviderBaseUrl(
  connection: ProviderConnectionDescriptor,
  protocol?: string | null,
  host?: string | null,
  port?: number | null,
): string {
  return `${protocol || connection.defaultProtocol}://${host || connection.defaultHost}:${port || connection.defaultPort}`;
}
