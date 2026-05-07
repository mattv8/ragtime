export const MOUNT_SYNC_MIN_SECONDS = 1;
export const MOUNT_SYNC_MAX_SECONDS = 2_592_000;
export const MOUNT_SYNC_DEFAULT_SECONDS = 30;

export function clampMountSyncInterval(seconds: number | null | undefined): number {
  if (!Number.isFinite(seconds ?? Number.NaN)) {
    return MOUNT_SYNC_DEFAULT_SECONDS;
  }
  return Math.max(
    MOUNT_SYNC_MIN_SECONDS,
    Math.min(MOUNT_SYNC_MAX_SECONDS, Math.round(seconds as number)),
  );
}

export function mountSyncIntervalToSlider(seconds: number): number {
  const clamped = clampMountSyncInterval(seconds);
  const minLog = Math.log(MOUNT_SYNC_MIN_SECONDS);
  const maxLog = Math.log(MOUNT_SYNC_MAX_SECONDS);
  return Math.round(((Math.log(clamped) - minLog) / (maxLog - minLog)) * 100);
}

export function sliderToMountSyncInterval(slider: number): number {
  const clampedSlider = Math.max(0, Math.min(100, slider));
  const minLog = Math.log(MOUNT_SYNC_MIN_SECONDS);
  const maxLog = Math.log(MOUNT_SYNC_MAX_SECONDS);
  return clampMountSyncInterval(Math.exp(minLog + (clampedSlider / 100) * (maxLog - minLog)));
}

export function formatMountSyncInterval(seconds: number | null | undefined): string {
  const clamped = clampMountSyncInterval(seconds);
  if (clamped < 60) {
    return `${clamped}s`;
  }
  if (clamped < 3600) {
    const minutes = Math.round(clamped / 60);
    return `${minutes}m`;
  }
  if (clamped < 86400) {
    const hours = Math.round(clamped / 3600);
    return `${hours}h`;
  }
  const days = Math.round(clamped / 86400);
  return `${days}d`;
}
