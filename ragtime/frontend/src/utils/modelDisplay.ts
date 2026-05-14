import { normalizeProviderAlias, parseScopedModelIdentifier } from './modelProviders';
export { parseScopedModelIdentifier } from './modelProviders';

export const CHAT_MODEL_PROVIDER_LABELS: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  openrouter: 'OpenRouter',
  ollama: 'Ollama',
  llama_cpp: 'llama.cpp',
  lmstudio: 'LM Studio',
  omlx: 'oMLX',
  github_copilot: 'GitHub Copilot',
  github_models: 'GitHub Copilot',
};

export const CHAT_MODEL_DISPLAY_LABELS: Record<string, string> = {
  'gpt-5.3-codex': 'GPT 5.3 Codex',
};

function toTitleCase(value: string): string {
  return value
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((word) => {
      if (/^gpt$/i.test(word)) return 'GPT';
      if (/^api$/i.test(word)) return 'API';
      if (/^llm$/i.test(word)) return 'LLM';
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');
}

export function formatProviderDisplayName(value: string | null | undefined): string {
  const { provider } = parseScopedModelIdentifier(value);
  const normalized = normalizeProviderAlias(provider || value || '') || '';
  if (CHAT_MODEL_PROVIDER_LABELS[normalized]) {
    return CHAT_MODEL_PROVIDER_LABELS[normalized];
  }
  if (provider || value) {
    return toTitleCase(normalized || provider || value || '');
  }
  if (!normalized) {
    return 'Unknown';
  }
  return toTitleCase(normalized);
}

export function formatModelDisplayName(value: string | null | undefined, providerHint?: string | null): string {
  const { provider, modelId } = parseScopedModelIdentifier(value);
  const fallback = (modelId || value || '').trim();
  if (!fallback) {
    return 'Unknown';
  }

  const providerHintParsed = parseScopedModelIdentifier(providerHint).provider || providerHint;
  const scopedPrefix = normalizeProviderAlias(provider || providerHintParsed || '') || '';
  const unscoped = scopedPrefix && fallback.startsWith(`${scopedPrefix}::`)
    ? fallback.slice(scopedPrefix.length + 2)
    : fallback;

  const normalized = unscoped.toLowerCase();
  if (CHAT_MODEL_DISPLAY_LABELS[normalized]) {
    return CHAT_MODEL_DISPLAY_LABELS[normalized];
  }

  return toTitleCase(unscoped);
}
