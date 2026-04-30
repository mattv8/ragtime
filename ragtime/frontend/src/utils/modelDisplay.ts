export const CHAT_MODEL_PROVIDER_LABELS: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  ollama: 'Ollama',
  llama_cpp: 'llama.cpp',
  github_copilot: 'Github Copilot',
  github_models: 'Github Copilot',
};

export const CHAT_MODEL_DISPLAY_LABELS: Record<string, string> = {
  'gpt-5.3-codex': 'GPT 5.3 Codex',
};

export function parseScopedModelIdentifier(value: string | null | undefined): { provider: string | null; modelId: string } {
  const raw = (value || '').trim();
  if (!raw) {
    return { provider: null, modelId: '' };
  }
  if (!raw.includes('::')) {
    return { provider: null, modelId: raw };
  }
  const [provider, ...rest] = raw.split('::');
  return {
    provider: provider.trim() || null,
    modelId: rest.join('::').trim(),
  };
}

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
  const normalized = (provider || value || '').trim().toLowerCase();
  if (CHAT_MODEL_PROVIDER_LABELS[normalized]) {
    return CHAT_MODEL_PROVIDER_LABELS[normalized];
  }
  if (provider) {
    return toTitleCase(provider);
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
  const scopedPrefix = (provider || providerHintParsed || '').trim();
  const unscoped = scopedPrefix && fallback.startsWith(`${scopedPrefix}::`)
    ? fallback.slice(scopedPrefix.length + 2)
    : fallback;

  const normalized = unscoped.toLowerCase();
  if (CHAT_MODEL_DISPLAY_LABELS[normalized]) {
    return CHAT_MODEL_DISPLAY_LABELS[normalized];
  }

  return toTitleCase(unscoped);
}
