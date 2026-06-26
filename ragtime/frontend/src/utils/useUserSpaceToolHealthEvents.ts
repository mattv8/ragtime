import { useEffect, useRef } from 'react';
import { api } from '@/api';
import type { HeartbeatStatus, UserSpaceAvailableTool } from '@/types';

interface ToolHealthEventPayload {
  type?: 'snapshot' | 'delta';
  changed_tool_ids?: string[];
  statuses?: Record<string, HeartbeatStatus>;
}

interface ToastLike {
  error?: (message: string, durationMs?: number) => void;
  info?: (message: string, durationMs?: number) => void;
}

interface UseUserSpaceToolHealthEventsOptions {
  availableTools: UserSpaceAvailableTool[];
  selectedToolIds: Set<string>;
  setAvailableTools: (
    updater: (current: UserSpaceAvailableTool[]) => UserSpaceAvailableTool[],
  ) => void;
  toast?: ToastLike;
  enabled?: boolean;
}

function statusAvailable(status: HeartbeatStatus): boolean {
  return status.available ?? status.alive;
}

function statusReason(status: HeartbeatStatus): string | null {
  return status.reason || status.error || (statusAvailable(status) ? null : 'Heartbeat failed');
}

export function useUserSpaceToolHealthEvents({
  availableTools,
  selectedToolIds,
  setAvailableTools,
  toast,
  enabled = true,
}: UseUserSpaceToolHealthEventsOptions) {
  const toolsRef = useRef(availableTools);
  const selectedToolIdsRef = useRef(selectedToolIds);
  const offlineNoticeRef = useRef<Map<string, string>>(new Map());

  useEffect(() => {
    toolsRef.current = availableTools;
  }, [availableTools]);

  useEffect(() => {
    selectedToolIdsRef.current = selectedToolIds;
  }, [selectedToolIds]);

  useEffect(() => {
    if (!enabled) return;

    const source = api.subscribeToolHealthEvents();

    const handleMessage = (event: MessageEvent) => {
      if (!event.data) return;
      let payload: ToolHealthEventPayload;
      try {
        payload = JSON.parse(event.data) as ToolHealthEventPayload;
      } catch {
        return;
      }
      const statuses = payload.statuses ?? {};
      const statusEntries = Object.entries(statuses);
      if (statusEntries.length === 0) return;

      const currentTools = toolsRef.current;
      const toolsById = new Map(currentTools.map((tool) => [tool.id, tool]));
      const selectedIds = selectedToolIdsRef.current;
      const wentOffline: string[] = [];
      const cameOnline: string[] = [];

      for (const [toolId, status] of statusEntries) {
        const tool = toolsById.get(toolId);
        if (!tool || !selectedIds.has(toolId)) continue;
        const wasAvailable = tool.available !== false;
        const isAvailable = statusAvailable(status);
        const reason = statusReason(status) || 'Heartbeat failed';
        if (wasAvailable && !isAvailable) {
          const previousReason = offlineNoticeRef.current.get(toolId);
          if (previousReason !== reason) {
            offlineNoticeRef.current.set(toolId, reason);
            wentOffline.push(tool.name);
          }
        } else if (!wasAvailable && isAvailable) {
          if (offlineNoticeRef.current.has(toolId)) {
            offlineNoticeRef.current.delete(toolId);
            cameOnline.push(tool.name);
          }
        }
      }

      setAvailableTools((current) =>
        current.map((tool) => {
          const status = statuses[tool.id];
          if (!status) return tool;
          const available = statusAvailable(status);
          const disabledReason = statusReason(status);
          return {
            ...tool,
            available,
            disabled_reason: available ? null : disabledReason,
          };
        }),
      );

      if (wentOffline.length === 1) {
        const toolId = statusEntries.find(
          ([id]) => toolsById.get(id)?.name === wentOffline[0],
        )?.[0];
        const reason = toolId ? offlineNoticeRef.current.get(toolId) : undefined;
        toast?.error?.(
          `Tool disabled: ${wentOffline[0]} failed healthcheck${reason ? ` (${reason})` : ''}.`,
          8000,
        );
      } else if (wentOffline.length > 1) {
        toast?.error?.(
          `${wentOffline.length} selected tools were disabled after failing healthchecks.`,
          9000,
        );
      }
      if (cameOnline.length === 1) {
        toast?.info?.(`Tool available again: ${cameOnline[0]}.`, 4000);
      } else if (cameOnline.length > 1) {
        toast?.info?.(`${cameOnline.length} selected tools are available again.`, 4000);
      }
    };

    source.addEventListener('snapshot', handleMessage as EventListener);
    source.addEventListener('delta', handleMessage as EventListener);
    source.onmessage = handleMessage;

    return () => {
      source.close();
    };
  }, [enabled, setAvailableTools, toast]);
}
