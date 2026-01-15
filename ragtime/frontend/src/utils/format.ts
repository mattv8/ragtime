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
