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

/**
 * Formats chat timestamps as relative text for recent activity and
 * a human-readable date/time for older messages.
 *
 * Rules:
 * - < 1m: "Just now"
 * - < 12h: "Xm ago" or "Xh ago"
 * - >= 12h: localized date/time string
 *
 * @param dateStr - ISO date string
 * @returns Formatted timestamp for chat UI
 */
export function formatChatTimestamp(dateStr: string): string {
  const normalized = dateStr.trim();
  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized);
  const date = new Date(hasExplicitTimezone ? normalized : `${normalized}Z`);
  if (Number.isNaN(date.getTime())) {
    return dateStr;
  }

  const diffMs = Date.now() - date.getTime();
  if (diffMs < 0) {
    return date.toLocaleString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  }

  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);

  if (diffMs <= 60000) return 'Just now';
  if (diffHours < 12) {
    if (diffMins < 60) return `${diffMins}m ago`;
    return `${diffHours}h ago`;
  }

  return date.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}
