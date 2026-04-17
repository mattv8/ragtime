import { useCallback, useEffect, useMemo, useState, type CSSProperties } from 'react';

import { api } from '@/api';
import type {
  UserSpaceRuntimeRestartBatchTask,
  UserSpaceRuntimeRestartWorkspaceTask,
} from '@/types';

const DEFAULT_RUNTIME_RESTART_POLL_MS = 2000;

function isRuntimeRestartTaskActive(task: UserSpaceRuntimeRestartBatchTask | null | undefined): boolean {
  return task?.phase === 'queued' || task?.phase === 'restarting';
}

function formatRuntimeRestartTaskStatus(task: UserSpaceRuntimeRestartBatchTask | null): string | null {
  if (!task) {
    return null;
  }

  switch (task.phase) {
    case 'queued':
      return `Queued to restart ${task.total_workspaces} active runtime${task.total_workspaces === 1 ? '' : 's'}.`;
    case 'restarting':
      return `Restarting ${task.total_workspaces} active runtime${task.total_workspaces === 1 ? '' : 's'} serially.`;
    case 'completed':
      return task.total_workspaces === 0
        ? 'No active runtime workspaces needed a restart.'
        : null;
    case 'completed_with_failures':
      return task.error || null;
    case 'failed':
      return task.error || 'Runtime restart batch failed.';
    default:
      return task.phase;
  }
}

function formatRuntimeRestartBatchPhaseLabel(phase: UserSpaceRuntimeRestartBatchTask['phase']): string {
  switch (phase) {
    case 'queued':
      return 'Queued';
    case 'restarting':
      return 'Restarting';
    case 'completed':
      return 'Completed';
    case 'completed_with_failures':
      return 'Completed with failures';
    case 'failed':
      return 'Failed';
    default:
      return phase;
  }
}

function formatRuntimeRestartWorkspacePhaseLabel(phase: UserSpaceRuntimeRestartWorkspaceTask['phase']): string {
  switch (phase) {
    case 'queued':
      return 'Queued';
    case 'restarting':
      return 'Restarting';
    case 'completed':
      return 'Completed';
    case 'skipped':
      return 'Skipped';
    case 'failed':
      return 'Failed';
    default:
      return phase;
  }
}

function formatRuntimeRestartWorkspaceStatus(item: UserSpaceRuntimeRestartWorkspaceTask): string {
  if (item.phase === 'restarting') {
    switch (item.runtime_operation_phase) {
      case 'queued':
        return 'Queued on runtime worker';
      case 'provisioning':
        return 'Provisioning sandbox';
      case 'bootstrapping':
        return 'Running bootstrap';
      case 'deps_install':
        return 'Installing dependencies';
      case 'launching':
        return 'Launching devserver';
      case 'probing':
        return 'Checking readiness';
      case 'ready':
        return 'Ready';
      case 'failed':
        return item.error || 'Restart failed';
      case 'stopped':
        return 'Stopped';
      default:
        return 'Restarting runtime';
    }
  }

  switch (item.phase) {
    case 'queued':
      return 'Waiting in queue';
    case 'completed':
      return 'Restarted successfully';
    case 'skipped':
      return item.error || 'Skipped';
    case 'failed':
      return item.error || 'Restart failed';
    default:
      return item.phase;
  }
}

interface UserSpaceRuntimeRestartPanelProps {
  enabled?: boolean;
  isVisible?: boolean;
  title?: string;
  description: string;
  buttonLabel?: string;
  activeButtonLabel?: string;
  queueingButtonLabel?: string;
  queuedSuccessMessage?: string;
  noTargetsSuccessMessage?: string;
  queueErrorMessage?: string;
  pollIntervalMs?: number;
  notifySuccess?: (message: string, durationMs?: number) => void;
  notifyError?: (message: string, durationMs?: number) => void;
  className?: string;
  style?: CSSProperties;
}

export function UserSpaceRuntimeRestartPanel({
  enabled = true,
  isVisible = true,
  title = 'Restart Active Runtime Workspaces',
  description,
  buttonLabel = 'Restart Active Runtimes',
  activeButtonLabel = 'Restart In Progress',
  queueingButtonLabel = 'Queueing...',
  queuedSuccessMessage = 'Queued restart of active runtime workspaces.',
  noTargetsSuccessMessage = 'No active runtime workspaces needed a restart.',
  queueErrorMessage = 'Failed to queue runtime restarts',
  pollIntervalMs = DEFAULT_RUNTIME_RESTART_POLL_MS,
  notifySuccess,
  notifyError,
  className,
  style,
}: UserSpaceRuntimeRestartPanelProps) {
  const [task, setTask] = useState<UserSpaceRuntimeRestartBatchTask | null>(null);
  const [queueing, setQueueing] = useState(false);

  const loadLatestTask = useCallback(async () => {
    if (!enabled) {
      setTask(null);
      return null;
    }
    try {
      const latestTask = await api.getLatestUserSpaceRuntimeRestartTask();
      setTask(latestTask);
      return latestTask;
    } catch {
      setTask(null);
      return null;
    }
  }, [enabled]);

  const loadTaskById = useCallback(async (taskId: string) => {
    try {
      const nextTask = await api.getUserSpaceRuntimeRestartTask(taskId);
      setTask(nextTask);
      return nextTask;
    } catch {
      return null;
    }
  }, []);

  const handleQueueRuntimeRestart = useCallback(async () => {
    if (!enabled) {
      return;
    }
    setQueueing(true);
    try {
      const nextTask = await api.queueUserSpaceRuntimeRestartTask();
      setTask(nextTask);
      if (nextTask.total_workspaces === 0) {
        notifySuccess?.(noTargetsSuccessMessage);
      } else {
        notifySuccess?.(queuedSuccessMessage);
      }
    } catch (err) {
      notifyError?.(err instanceof Error ? err.message : queueErrorMessage);
    } finally {
      setQueueing(false);
    }
  }, [enabled, noTargetsSuccessMessage, notifyError, notifySuccess, queueErrorMessage, queuedSuccessMessage]);

  useEffect(() => {
    if (!enabled) {
      setTask(null);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled || !isVisible) {
      return;
    }
    void loadLatestTask();
  }, [enabled, isVisible, loadLatestTask]);

  const taskActive = isRuntimeRestartTaskActive(task);

  useEffect(() => {
    if (!enabled || !isVisible || !task?.task_id || !taskActive) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadTaskById(task.task_id);
    }, pollIntervalMs);

    return () => window.clearInterval(intervalId);
  }, [enabled, isVisible, loadTaskById, pollIntervalMs, task, taskActive]);

  const taskStatus = formatRuntimeRestartTaskStatus(task);
  const visibleResults = useMemo(() => {
    if (!task) {
      return [] as UserSpaceRuntimeRestartWorkspaceTask[];
    }
    if (taskActive) {
      return task.workspace_results;
    }
    return task.workspace_results.filter((item) => item.phase !== 'completed');
  }, [task, taskActive]);

  if (!enabled) {
    return null;
  }

  return (
    <div
      className={className}
      style={{
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '0.625rem 0.75rem',
        background: 'var(--bg-secondary)',
        ...style,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 320px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.2rem' }}>
            <strong>{title}</strong>
            {taskActive && <span className="live-indicator" title="Auto-refreshing">LIVE</span>}
          </div>
          <p className="field-help" style={{ margin: 0, fontSize: '0.95em' }}>
            {description}
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => { void handleQueueRuntimeRestart(); }}
          disabled={queueing || taskActive}
        >
          {queueing ? queueingButtonLabel : taskActive ? activeButtonLabel : buttonLabel}
        </button>
      </div>

      {task && (
        <div style={{ marginTop: '0.625rem' }}>
          <div style={{ display: 'flex', gap: '0.625rem', flexWrap: 'wrap', alignItems: 'center', marginBottom: taskStatus ? '0.25rem' : 0 }}>
            <span className="badge">{formatRuntimeRestartBatchPhaseLabel(task.phase)}</span>
            <span className="userspace-muted" style={{ fontSize: '0.95em' }}>
              {task.completed_workspaces}/{task.total_workspaces} completed
              {task.failed_workspaces > 0 ? `, ${task.failed_workspaces} failed` : ''}
              {task.skipped_workspaces > 0 ? `, ${task.skipped_workspaces} skipped` : ''}
            </span>
          </div>
          {taskStatus && (
            <p className="field-help" style={{ margin: '0 0 0.375rem', fontSize: '0.95em' }}>
              {taskStatus}
            </p>
          )}
          {task.current_workspace_name && taskActive && (
            <p className="field-help" style={{ margin: '0 0 0.375rem', fontSize: '0.95em' }}>
              Current workspace: <strong>{task.current_workspace_name}</strong>
            </p>
          )}
          {visibleResults.length > 0 && (
            <div style={{ maxHeight: '180px', overflowY: 'auto', borderTop: '1px solid var(--border-color)', paddingTop: '0.375rem' }}>
              {visibleResults.map((item) => (
                <div
                  key={item.workspace_id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '0.625rem',
                    padding: '0.25rem 0',
                    borderBottom: '1px solid var(--border-color)',
                    alignItems: 'flex-start',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, lineHeight: 1.25 }}>{item.workspace_name}</div>
                    <div className="userspace-muted" style={{ fontSize: '0.86em', lineHeight: 1.3, marginTop: '0.1rem' }}>
                      {formatRuntimeRestartWorkspaceStatus(item)}
                    </div>
                  </div>
                  <span className="badge" style={{ whiteSpace: 'nowrap' }}>{formatRuntimeRestartWorkspacePhaseLabel(item.phase)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}