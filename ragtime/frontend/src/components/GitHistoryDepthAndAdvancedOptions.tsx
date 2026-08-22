import type { CommitHistoryInfo } from '@/types';
import { IndexConfigFields } from './IndexConfigFields';

interface GitHistoryDepthAndAdvancedOptionsProps {
  gitHistoryDepth: number;
  onGitHistoryDepthChange: (depth: number) => void;
  gitCloneTimeoutMinutes: number;
  onGitCloneTimeoutMinutesChange: (timeout: number) => void;
  filePatterns: string;
  onFilePattersChange: (patterns: string) => void;
  excludePatterns: string;
  onExcludePattersChange: (patterns: string) => void;
  chunkSize: number;
  onChunkSizeChange: (size: number) => void;
  chunkOverlap: number;
  onChunkOverlapChange: (overlap: number) => void;
  maxFileSizeKb: number;
  onMaxFileSizeKbChange: (size: number) => void;
  isLoading: boolean;
  onTimeoutManuallySet?: (set: boolean) => void;
  /** Optional commit history for enhanced depth descriptions (review step) */
  commitHistory?: CommitHistoryInfo;
  /** Whether Advanced Options details should be expanded by default */
  patternsExpanded?: boolean;
  /** Callback when Advanced Options section is toggled */
  onPatternsExpandedChange?: (expanded: boolean) => void;
  /** Optional: render a button below Advanced Options (for Re-analyze) */
  advancedOptionsFooterButton?: React.ReactNode;
}

/**
 * Shared component for Git History Depth slider and Advanced Options section.
 * Used in multiple wizard steps to eliminate code duplication.
 */
export function GitHistoryDepthAndAdvancedOptions({
  gitHistoryDepth,
  onGitHistoryDepthChange,
  gitCloneTimeoutMinutes,
  onGitCloneTimeoutMinutesChange,
  filePatterns,
  onFilePattersChange,
  excludePatterns,
  onExcludePattersChange,
  chunkSize,
  onChunkSizeChange,
  chunkOverlap,
  onChunkOverlapChange,
  maxFileSizeKb,
  onMaxFileSizeKbChange,
  isLoading,
  onTimeoutManuallySet,
  commitHistory,
  patternsExpanded = false,
  onPatternsExpandedChange,
  advancedOptionsFooterButton,
}: GitHistoryDepthAndAdvancedOptionsProps) {
  return (
    <>
      {/* Git History Depth - outside Advanced Options for prominence */}
      <div className="form-group" style={{ marginBottom: '16px' }}>
        <label>Git History Depth</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <input
            type="range"
            min={1}
            max={1001}
            value={gitHistoryDepth === 0 ? 1001 : gitHistoryDepth}
            onChange={(e) => {
              const sliderVal = parseInt(e.target.value, 10);
              onGitHistoryDepthChange(sliderVal === 1001 ? 0 : sliderVal);
            }}
            disabled={isLoading}
            style={{ flex: '1 1 300px' }}
          />
          <span style={{ minWidth: '80px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
            {gitHistoryDepth === 0
              ? 'Full'
              : gitHistoryDepth === 1
                ? '1 (shallow)'
                : `${gitHistoryDepth} commits`}
          </span>
        </div>
        <small style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem' }}>
          {gitHistoryDepth === 0
            ? commitHistory?.total_commits
              ? `Full history: Indexes all commits. (${commitHistory.total_commits.toLocaleString()} commits) Large repos may take 30+ min to clone.`
              : 'Full history: Indexes all commits. Large repos may take 30+ min to clone.'
            : gitHistoryDepth === 1
              ? 'Shallow clone: Only latest commit. Fastest, but no git history search.'
              : (() => {
                  // Only compute depth estimate if commitHistory is provided (review step)
                  if (commitHistory) {
                    const dateEstimate = getDepthDateEstimate(gitHistoryDepth, commitHistory);
                    return dateEstimate
                      ? `Indexes last ${gitHistoryDepth} commits (${dateEstimate}). Clone time scales with depth.`
                      : `Indexes last ${gitHistoryDepth} commits. Clone time scales with depth.`;
                  }
                  return `Indexes last ${gitHistoryDepth} commits. Clone time scales with depth.`;
                })()}
        </small>
      </div>

      <details
        style={{ marginBottom: '16px' }}
        open={patternsExpanded}
        onToggle={(e) => onPatternsExpandedChange?.((e.target as HTMLDetailsElement).open)}
      >
        <summary style={{ cursor: 'pointer', color: 'var(--color-accent)', marginBottom: '8px' }}>
          Advanced Options
        </summary>

        <IndexConfigFields
          isLoading={isLoading}
          filePatterns={filePatterns}
          setFilePatterns={onFilePattersChange}
          excludePatterns={excludePatterns}
          setExcludePatterns={onExcludePattersChange}
          chunkSize={chunkSize}
          setChunkSize={onChunkSizeChange}
          chunkOverlap={chunkOverlap}
          setChunkOverlap={onChunkOverlapChange}
          maxFileSizeKb={maxFileSizeKb}
          setMaxFileSizeKb={onMaxFileSizeKbChange}
          gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
          setGitCloneTimeoutMinutes={onGitCloneTimeoutMinutesChange}
          setTimeoutManuallySet={onTimeoutManuallySet}
        />

        {advancedOptionsFooterButton}
      </details>
    </>
  );
}

/**
 * Interpolates a date for a given depth from commit history samples.
 * Returns a human-readable description like "~6 months of history".
 */
function getDepthDateEstimate(
  depth: number,
  commitHistory: CommitHistoryInfo | undefined,
): string | null {
  if (!commitHistory?.samples || commitHistory.samples.length < 2) return null;
  if (depth === 0) return null; // Full history - use oldest_date directly
  if (depth === 1) return null; // Shallow - no history

  const samples = commitHistory.samples;
  const newest = samples.find((s) => s.depth === 0);
  if (!newest) return null;

  const newestDate = new Date(newest.date);
  let estimatedDate: Date | null = null;

  // Find the two samples that bracket the requested depth
  for (let i = 0; i < samples.length - 1; i++) {
    const lower = samples[i];
    const upper = samples[i + 1];
    if (depth >= lower.depth && depth <= upper.depth) {
      // Linear interpolation between the two sample dates
      const ratio = (depth - lower.depth) / (upper.depth - lower.depth);
      const lowerDate = new Date(lower.date);
      const upperDate = new Date(upper.date);
      const interpolatedMs =
        lowerDate.getTime() + ratio * (upperDate.getTime() - lowerDate.getTime());
      estimatedDate = new Date(interpolatedMs);
      break;
    }
  }

  // If depth is beyond our last sample, extrapolate or use oldest_date
  if (!estimatedDate && depth > samples[samples.length - 1].depth) {
    if (commitHistory.oldest_date) {
      estimatedDate = new Date(commitHistory.oldest_date);
    } else {
      // Extrapolate from last two samples
      const last = samples[samples.length - 1];
      const prev = samples[samples.length - 2];
      if (last && prev) {
        const ratio = (depth - prev.depth) / (last.depth - prev.depth);
        const prevDate = new Date(prev.date);
        const lastDate = new Date(last.date);
        const interpolatedMs =
          prevDate.getTime() + ratio * (lastDate.getTime() - prevDate.getTime());
        estimatedDate = new Date(interpolatedMs);
      }
    }
  }

  if (!estimatedDate) return null;

  // Calculate time difference and format nicely
  const diffMs = newestDate.getTime() - estimatedDate.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays < 7) return `~${diffDays} days of history`;
  if (diffDays < 30) return `~${Math.round(diffDays / 7)} weeks of history`;
  if (diffDays < 365) return `~${Math.round(diffDays / 30)} months of history`;
  const years = (diffDays / 365).toFixed(1);
  return `~${years} years of history`;
}
