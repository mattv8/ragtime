export class ArtifactRequestError extends Error {
  constructor(
    public status: number,
    message = `Artifact request failed (${status}).`,
  ) {
    super(message);
    this.name = 'ArtifactRequestError';
  }
}

export function artifactStateFromStatus(
  status: number,
): 'signin' | 'forbidden' | 'expired' | 'notfound' | 'error' {
  if (status === 401) return 'signin';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'notfound';
  if (status === 410) return 'expired';
  return 'error';
}

export function formatArtifactSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatArtifactExpiry(
  expiresAt: string,
  now = new Date(),
): { label: string; expired: boolean; soon: boolean } {
  const expiry = new Date(expiresAt);
  const remaining = expiry.getTime() - now.getTime();
  if (remaining < 0) {
    return { label: `Expired ${expiry.toLocaleDateString()}`, expired: true, soon: false };
  }
  const hours = Math.max(1, Math.ceil(remaining / 3_600_000));
  const count = hours < 24 ? hours : Math.ceil(hours / 24);
  const unit = hours < 24 ? 'hour' : 'day';
  return {
    label: `Expires in ${count} ${unit}${count === 1 ? '' : 's'}`,
    expired: false,
    soon: remaining < 86_400_000,
  };
}

export interface ModelArtifact {
  id: string;
  kind: 'preview' | 'model' | 'native';
  format: string;
  file_name: string;
  size_bytes: number;
  expires_at: string;
  content_url: string;
  preview_image_url?: string;
  page_url: string;
}

type ArtifactToolCall = {
  tool: string;
  output?: string;
  connection?: { tool_type?: string };
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validContentUrl(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^\/indexes\/solidworks\/artifacts\/[a-f0-9]{32}\/content$/i.test(value)
  );
}

function validateArtifact(value: unknown): ModelArtifact | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.id !== 'string' ||
    !/^[a-f0-9]{32}$/i.test(value.id) ||
    !['preview', 'model', 'native'].includes(String(value.kind)) ||
    typeof value.format !== 'string' ||
    !value.format ||
    typeof value.file_name !== 'string' ||
    !value.file_name ||
    typeof value.size_bytes !== 'number' ||
    !Number.isFinite(value.size_bytes) ||
    value.size_bytes < 0 ||
    typeof value.expires_at !== 'string' ||
    !Number.isFinite(Date.parse(value.expires_at)) ||
    !validContentUrl(value.content_url) ||
    value.content_url !== `/indexes/solidworks/artifacts/${value.id}/content` ||
    value.page_url !== `/?artifact=${value.id}` ||
    (value.preview_image_url !== undefined && !validContentUrl(value.preview_image_url))
  )
    return null;
  return value as unknown as ModelArtifact;
}

function extractArtifact(output: string): unknown {
  const match = /"artifact"\s*:\s*\{/g.exec(output);
  if (!match) return null;
  const start = output.indexOf('{', match.index);
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let i = start; i < output.length; i++) {
    const char = output[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '{') depth++;
    else if (char === '}' && --depth === 0) {
      try {
        return JSON.parse(output.slice(start, i + 1));
      } catch {
        return null;
      }
    }
  }
  return null;
}

export function parseModelArtifact(toolCall: ArtifactToolCall): ModelArtifact | null {
  if (!toolCall.tool.startsWith('solidworks_') && toolCall.connection?.tool_type !== 'solidworks')
    return null;
  const output = toolCall.output;
  if (!output) return null;
  try {
    const envelope: unknown = JSON.parse(output);
    if (
      !isRecord(envelope) ||
      envelope.ok === false ||
      !['model.export', 'model.preview'].includes(String(envelope.operation))
    )
      return null;
    const result = isRecord(envelope.result) ? envelope.result : {};
    return validateArtifact(result.artifact ?? envelope.artifact);
  } catch {
    if (
      /"ok"\s*:\s*false/.test(output) ||
      !/"operation"\s*:\s*"model\.(?:export|preview)"/.test(output)
    )
      return null;
    return validateArtifact(extractArtifact(output));
  }
}
