import type { MountProblem } from '@/types';

function relativeFailureTime(failingSince: string, now: Date): string | null {
  const failedAt = new Date(failingSince).getTime();
  if (Number.isNaN(failedAt)) return null;
  const elapsedMs = Math.max(0, now.getTime() - failedAt);
  const elapsedMinutes = Math.floor(elapsedMs / 60_000);
  if (elapsedMinutes < 1) return '<1 min ago';
  if (elapsedMinutes < 60) return `${elapsedMinutes} min ago`;
  const elapsedHours = Math.floor(elapsedMinutes / 60);
  if (elapsedHours < 24) return `${elapsedHours} h ago`;
  return `${Math.floor(elapsedHours / 24)} d ago`;
}

export function formatMountProblem(problem: MountProblem, now = new Date()): string {
  const container = `${problem.container} container`;
  const state = problem.state === 'failed' ? 'unavailable' : 'not responding';
  const relativeFailure = problem.failing_since
    ? relativeFailureTime(problem.failing_since, now)
    : null;
  const since = relativeFailure ? ` since ${relativeFailure}` : '';
  const error = problem.error ? ` Error: ${problem.error}.` : '';
  const source =
    problem.fstype === 'autofs'
      ? ' The host automount could not mount it.'
      : ` Source: ${problem.source}.`;
  return `${problem.mount_point} (${container}) is ${state}${since}.${source}${error}`;
}

export function mountRemediationHint(problems: MountProblem[]): string {
  const mountPoints = [...new Set(problems.map((problem) => problem.mount_point))];
  const example =
    mountPoints.length > 1
      ? `for example: findmnt ${mountPoints[0]}; repeat for each listed path`
      : `for example: findmnt ${mountPoints[0] ?? ''}`;
  return `Ragtime retries automatically every minute. If this persists, check on the Docker host that the share is reachable and mounted (${example}), restart its mount unit if needed, then select Check again. User Space workspaces started during the outage may need a restart.`;
}

export function buildAdminMountWarnings(problems: MountProblem[], now = new Date()): string[] {
  if (problems.length === 0) return [];
  return [
    ...problems.map((problem) => formatMountProblem(problem, now)),
    mountRemediationHint(problems),
  ];
}

export function mountProblemSignature(problems: MountProblem[]): string {
  return [
    ...new Set(
      problems.map(
        ({ container, mount_point, failing_since }) =>
          `${container}:${mount_point}@${failing_since ?? ''}`,
      ),
    ),
  ]
    .sort()
    .join('|');
}

export const NON_ADMIN_MOUNT_WARNING =
  'Some shared files or workspace data may be temporarily unavailable. Contact an administrator if this persists.';
