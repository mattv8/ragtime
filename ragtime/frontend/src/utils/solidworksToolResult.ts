/**
 * Parsing and humanising of `solidworks_<name>` tool results for the chat UI.
 */

export type SolidWorksResultState = 'running' | 'ok' | 'error' | 'unparsed';

/** `marker`: valid JSON carrying `_ragtime_output_truncated`; `cut`: JSON repaired after a hard cut. */
export type SolidWorksTruncation = 'marker' | 'cut' | null;

export interface SolidWorksArgument {
  key: string;
  label: string;
  value: string;
  kind: 'text' | 'code' | 'boolean' | 'number';
}

export interface SolidWorksErrorDetail {
  label: string;
  value: string;
}

export interface SolidWorksError {
  code: string;
  message: string;
  details: SolidWorksErrorDetail[];
}

export interface ParsedSolidWorksResult {
  state: SolidWorksResultState;
  /** Wire operation name, e.g. `documents.list`; empty when unknown. */
  operation: string;
  /** Humanised title, e.g. "List documents", "Read guide: previews". */
  title: string;
  args: SolidWorksArgument[];
  elapsedMs: number | null;
  solidworksVersion: string | null;
  truncated: SolidWorksTruncation;
  error: SolidWorksError | null;
  /** The `result` object of the envelope (possibly partial after repair); null when absent. */
  result: Record<string, unknown> | null;
}

export interface SolidWorksToolCallLike {
  tool: string;
  connection?: { tool_type?: string | null; tool_config_name?: string | null } | null;
}

export function isSolidWorksToolCall(toolCall: SolidWorksToolCallLike): boolean {
  return toolCall.tool.startsWith('solidworks_') || toolCall.connection?.tool_type === 'solidworks';
}

export function solidWorksToolPillName(toolCall: SolidWorksToolCallLike): string {
  const configuredName = toolCall.connection?.tool_config_name;
  if (configuredName?.trim()) return configuredName;
  return humanizeOperation(toolCall.tool.replace(/^solidworks_/, '')).replace(
    /\bsolidworks\b/i,
    'SolidWorks',
  );
}

export function humanizeOperation(
  operation: string,
  args?: Record<string, unknown> | null,
): string {
  if (operation === 'guide') {
    return typeof args?.topic === 'string' && args.topic
      ? `Read guide: ${args.topic}`
      : 'List guide topics';
  }
  const titles: Record<string, string> = {
    describe: 'Describe operations',
    status: 'Status',
    'documents.list': 'List documents',
    'document.open': 'Open document',
    'document.close': 'Close document',
    'model.export': 'Export model',
    'model.preview': 'Preview model',
    'model.summary': 'Model summary',
    'components.list': 'List components',
    'features.list': 'List features',
    'configurations.list': 'List configurations',
    'dimensions.list': 'List dimensions',
    'mates.list': 'List mates',
    'equations.list': 'List equations',
    'properties.get': 'Read properties',
    'mass_properties.get': 'Mass properties',
    'feature.get': 'Read feature',
    'sketch.get': 'Read sketch',
    'script.run': 'Run script',
    'pdm.search': 'Search PDM',
    'pdm.lookup': 'Look up PDM documents',
    'pdm.file.get_latest': 'PDM get latest',
  };
  return titles[operation] || (operation ? humanizeLabel(operation) : 'SolidWorks');
}

function humanizeLabel(key: string): string {
  if (key.toLowerCase() === 'hresult') return 'HRESULT';
  const words = key.replace(/[._]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function humanizeArguments(
  args: Record<string, unknown> | null | undefined,
): SolidWorksArgument[] {
  return Object.entries(args ?? {}).flatMap(([key, raw]) => {
    if (raw == null) return [];
    let value: string;
    let kind: SolidWorksArgument['kind'] = 'text';
    if (typeof raw === 'boolean') {
      value = raw ? 'Yes' : 'No';
      kind = 'boolean';
    } else if (typeof raw === 'number') {
      value = String(raw);
      kind = 'number';
    } else if (typeof raw === 'object') {
      try {
        value = JSON.stringify(raw);
      } catch {
        value = '[Unserializable]';
      }
      kind = 'code';
    } else {
      value = String(raw);
      if (/^(?:[a-z]:[\\/]|\\\\|\/|\.{1,2}\/|doc-\d+$)/i.test(value)) kind = 'code';
      else if (key === 'format') value = value.toUpperCase();
      else if (key === 'view') value = humanizeLabel(value);
    }
    return [{ key, label: humanizeLabel(key), value, kind }];
  });
}

function repairTruncatedJson(output: string): Record<string, unknown> | null {
  if (!output.trimStart().startsWith('{')) return null;
  const stack: string[] = [];
  const candidates: { end: number; closers: string }[] = [];
  let inString = false;
  let escaped = false;
  for (let i = 0; i < output.length; i++) {
    const char = output[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') {
        inString = false;
        candidates.push({ end: i + 1, closers: [...stack].reverse().join('') });
      }
    } else if (char === '"') inString = true;
    else if (char === '{' || char === '[') stack.push(char === '{' ? '}' : ']');
    else if (char === '}' || char === ']') {
      if (stack.pop() !== char) return null;
      candidates.push({ end: i + 1, closers: [...stack].reverse().join('') });
    } else if (char === ',' || /\s/.test(char)) {
      candidates.push({ end: i, closers: [...stack].reverse().join('') });
    }
  }
  candidates.push({ end: output.length, closers: [...stack].reverse().join('') });
  for (const candidate of candidates.reverse()) {
    try {
      const parsed = asRecord(JSON.parse(output.slice(0, candidate.end) + candidate.closers));
      if (parsed && Object.keys(parsed).length) return parsed;
    } catch {
      // Try the preceding complete value.
    }
  }
  return null;
}

function extractFallback(output: string): Record<string, unknown> | null {
  const stringValue = (text: string, key: string): string | undefined => {
    const match = text.match(new RegExp(`"${key}"\\s*:\\s*("(?:\\\\.|[^"\\\\])*")`));
    if (!match) return undefined;
    try {
      return JSON.parse(match[1]) as string;
    } catch {
      return undefined;
    }
  };
  const operation = stringValue(output, 'operation');
  const ok = output.match(/"ok"\s*:\s*(true|false)\b/)?.[1];
  const errorText = output.match(/"error"\s*:\s*\{([\s\S]*)/)?.[1];
  const code = errorText && stringValue(errorText, 'code');
  const message = errorText && stringValue(errorText, 'message');
  if (!operation && !ok && !code && !message && !errorText) return null;
  return {
    operation,
    ok: ok === undefined ? undefined : ok === 'true',
    ...(errorText ? { error: { code, message } } : {}),
  };
}

function optionalVersion(value: unknown): string | null {
  return (typeof value === 'string' && value.trim()) || typeof value === 'number'
    ? String(value)
    : null;
}

export function parseSolidWorksResult(
  output: string | null | undefined,
  input: unknown,
  status?: string,
): ParsedSolidWorksResult {
  const request = asRecord(input);
  const argumentsObject = asRecord(request?.arguments);
  const args = humanizeArguments(argumentsObject);
  if (request?.timeout != null) {
    args.push({
      key: 'timeout',
      label: 'Timeout',
      value: `${String(request.timeout)} s`,
      kind: 'number',
    });
  }
  let envelope: Record<string, unknown> | null = null;
  let truncated: SolidWorksTruncation = null;
  if (output) {
    try {
      envelope = asRecord(JSON.parse(output));
      if (envelope?._ragtime_output_truncated === true) truncated = 'marker';
    } catch {
      envelope = repairTruncatedJson(output);
      if (envelope) truncated = 'cut';
      else envelope = extractFallback(output);
    }
  }
  const operation =
    typeof envelope?.operation === 'string' && envelope.operation
      ? envelope.operation
      : typeof request?.operation === 'string'
        ? request.operation
        : '';
  const result = asRecord(envelope?.result);
  const host = asRecord(envelope?.host);
  const errorObject = asRecord(envelope?.error);
  const error = errorObject
    ? {
        code: typeof errorObject.code === 'string' ? errorObject.code : '',
        message: typeof errorObject.message === 'string' ? errorObject.message : '',
        details: humanizeArguments(asRecord(errorObject.details)).map(({ label, value }) => ({
          label,
          value,
        })),
      }
    : null;
  return {
    state:
      !output && status === 'running'
        ? 'running'
        : !envelope
          ? 'unparsed'
          : envelope.ok === false || error
            ? 'error'
            : 'ok',
    operation,
    title: humanizeOperation(operation, argumentsObject),
    args,
    elapsedMs: typeof envelope?.elapsed_ms === 'number' ? envelope.elapsed_ms : null,
    solidworksVersion: optionalVersion(host?.solidworks_version),
    truncated,
    error,
    result,
  };
}
