import type {
  UserSpaceWorkspaceDeleteTask,
  UserSpaceWorkspaceDeleteTaskPhase,
} from '@/types';

/**
 * Checks if a workspace delete task has reached a terminal phase.
 * Terminal phases are 'completed' or 'failed'.
 */
export function isWorkspaceDeleteTaskTerminal(phase: UserSpaceWorkspaceDeleteTaskPhase): boolean {
  return phase === 'completed' || phase === 'failed';
}

/**
 * Formats a single workspace delete task into a human-readable status message.
 * Returns null if the task is null or in an unknown phase.
 */
export function formatWorkspaceDeleteTaskStatus(
  task: UserSpaceWorkspaceDeleteTask | null,
): string | null {
  if (!task) {
    return null;
  }

  const label = task.workspace_name?.trim() || 'workspace';
  switch (task.phase) {
    case 'queued':
      return `Preparing to delete ${label}...`;
    case 'stopping_runtime':
      return `Stopping runtime for ${label}...`;
    case 'deleting_conversations':
      return `Deleting conversations for ${label}...`;
    case 'deleting_workspace':
      return `Deleting ${label}...`;
    case 'failed':
      return task.error?.trim() || `Failed to delete ${label}.`;
    default:
      return null;
  }
}

/**
 * Formats multiple workspace delete tasks into a combined status message.
 * Returns null if the array is empty.
 * Shows queued count if there are tasks waiting to start.
 */
export function formatWorkspaceDeleteTasksStatus(
  tasks: UserSpaceWorkspaceDeleteTask[],
): string | null {
  if (tasks.length === 0) {
    return null;
  }
  if (tasks.length === 1) {
    return formatWorkspaceDeleteTaskStatus(tasks[0]);
  }

  const queuedCount = tasks.filter((task) => task.phase === 'queued').length;
  return queuedCount > 0
    ? `Deleting ${tasks.length} workspaces (${queuedCount} queued)...`
    : `Deleting ${tasks.length} workspaces...`;
}
