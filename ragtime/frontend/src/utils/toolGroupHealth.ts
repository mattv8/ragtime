import type { HeartbeatStatus, ToolConfig } from '@/types';

export interface ToolGroupHealth {
  tone: 'error' | 'success' | 'neutral';
  label: string;
}

export function summarizeToolGroupHealth(
  tools: ToolConfig[],
  heartbeats: Record<string, HeartbeatStatus | undefined>,
  testingToolId: string | null,
): ToolGroupHealth {
  if (!tools.length) return { tone: 'neutral', label: 'Empty' };
  const enabled = tools.filter((tool) => tool.enabled);
  if (!enabled.length) return { tone: 'neutral', label: 'All disabled' };
  const failing = enabled.filter(
    (tool) => tool.id !== testingToolId && heartbeats[tool.id]?.alive === false,
  ).length;
  if (failing) return { tone: 'error', label: `${failing} failing` };
  if (enabled.some((tool) => tool.id === testingToolId || !heartbeats[tool.id]))
    return { tone: 'neutral', label: 'Checking…' };
  return { tone: 'success', label: 'All healthy' };
}
