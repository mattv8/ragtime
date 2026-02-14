/**
 * Shared formatting utilities
 */

/**
 * Formats a size in MB into human-readable string with appropriate unit.
 * Automatically converts to KB, MB, GB, or TB based on magnitude.
 *
 * @param mb - Size in megabytes
 * @param decimals - Number of decimal places (default: 2 for GB/TB, 0 for KB/MB)
 * @returns Formatted string like "1.23 GB", "456 MB", "789 KB"
 */
export function formatSizeMB(mb: number): string {
  if (mb >= 1024 * 1024) {
    // TB
    return `${(mb / (1024 * 1024)).toFixed(2)} TB`;
  }
  if (mb >= 1024) {
    // GB
    return `${(mb / 1024).toFixed(2)} GB`;
  }
  if (mb >= 1) {
    // MB - show with precision for small values
    if (mb < 10) {
      return `${mb.toFixed(2)} MB`;
    }
    return `${mb.toFixed(0)} MB`;
  }
  // KB
  return `${(mb * 1024).toFixed(0)} KB`;
}

/**
 * Formats bytes into human-readable string with appropriate unit.
 *
 * @param bytes - Size in bytes
 * @returns Formatted string like "1.23 GB", "456 MB", "789 KB", "123 B"
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes < 1024 * 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  return `${(bytes / (1024 * 1024 * 1024 * 1024)).toFixed(2)} TB`;
}

/**
 * Formats elapsed time from milliseconds into compact human-readable units.
 *
 * Format rules:
 * - < 60s: "45s"
 * - < 1h: "3m 12s"
 * - >= 1h: "1h 3m 12s"
 *
 * @param milliseconds - Elapsed duration in milliseconds
 * @returns Formatted elapsed time string
 */
export function formatElapsedTime(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }

  return `${minutes}m ${seconds}s`;
}
