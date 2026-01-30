import type { ChangeEvent } from 'react';
import type { CommitHistoryInfo } from '@/types';

// Placeholders should match constants in GitIndexWizard.tsx
const PLACEHOLDER_FILE_PATTERNS = 'e.g. **/*.py, **/*.md (default: all files)';
const PLACEHOLDER_EXCLUDE_PATTERNS = 'e.g. **/node_modules/**, **/__pycache__/**';

interface IndexConfigFieldsProps {
  isLoading: boolean;
  filePatterns: string;
  setFilePatterns: (val: string) => void;
  excludePatterns: string;
  setExcludePatterns: (val: string) => void;
  chunkSize: number;
  setChunkSize: (val: number) => void;
  chunkOverlap: number;
  setChunkOverlap: (val: number) => void;
  maxFileSizeKb: number;
  setMaxFileSizeKb: (val: number) => void;
  ocrMode: 'disabled' | 'tesseract' | 'ollama';
  setOcrMode: (val: 'disabled' | 'tesseract' | 'ollama') => void;

  /**
   * Whether Ollama is available as an LLM provider.
   * When false, only Tesseract OCR option is shown.
   * When Ollama is selected, the global OCR vision model setting is used.
   */
  ollamaAvailable?: boolean;

  // Git-specific props (optional)
  gitCloneTimeoutMinutes?: number;
  setGitCloneTimeoutMinutes?: (val: number) => void;
  setTimeoutManuallySet?: (val: boolean) => void;
  reindexIntervalHours?: number;
  setReindexIntervalHours?: (val: number) => void;
  gitHistoryDepth?: number;
  setGitHistoryDepth?: (val: number) => void;

  /**
   * Optional commit history for displaying date estimates.
   * If provided, will use getDepthDateEstimate logic (Review Mode).
   * If undefined/null, will use generic text (Create/Edit Mode).
   */
  commitHistory?: CommitHistoryInfo;

  /**
   * Optional helper function to format the depth date estimate.
   * If not provided, will default to null/generic text.
   */
  getDepthDateEstimate?: (depth: number, history?: CommitHistoryInfo) => string | null;
}

export function IndexConfigFields({
  isLoading,
  filePatterns,
  setFilePatterns,
  excludePatterns,
  setExcludePatterns,
  chunkSize,
  setChunkSize,
  chunkOverlap,
  setChunkOverlap,
  maxFileSizeKb,
  setMaxFileSizeKb,
  gitCloneTimeoutMinutes,
  setGitCloneTimeoutMinutes,
  setTimeoutManuallySet,
  ocrMode,
  setOcrMode,
  ollamaAvailable = false,
  reindexIntervalHours,
  setReindexIntervalHours,
  gitHistoryDepth,
  setGitHistoryDepth,
  commitHistory,
  getDepthDateEstimate
}: IndexConfigFieldsProps) {

  const handleOcrModeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setOcrMode(e.target.value as 'disabled' | 'tesseract' | 'ollama');
  };

  const handleHistoryChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (setGitHistoryDepth) {
      const sliderVal = parseInt(e.target.value, 10);
      // 1 = shallow (depth 1), 1000 = one step below full, 1001 = full history (depth 0)
      setGitHistoryDepth(sliderVal === 1001 ? 0 : sliderVal);
    }
  };

  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
        <div className="form-group">
          <label>File Patterns (comma-separated)</label>
          <input
            type="text"
            value={filePatterns}
            onChange={(e) => setFilePatterns(e.target.value)}
            placeholder={PLACEHOLDER_FILE_PATTERNS}
            disabled={isLoading}
          />
        </div>
        <div className="form-group">
          <label>Exclude Patterns</label>
          <input
            type="text"
            value={excludePatterns}
            onChange={(e) => setExcludePatterns(e.target.value)}
            placeholder={PLACEHOLDER_EXCLUDE_PATTERNS}
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label>Chunk Size</label>
          <input
            type="number"
            value={chunkSize}
            onChange={(e) => setChunkSize(parseInt(e.target.value, 10) || 1000)}
            min={100}
            max={4000}
            disabled={isLoading}
          />
        </div>
        <div className="form-group">
          <label>Chunk Overlap</label>
          <input
            type="number"
            value={chunkOverlap}
            onChange={(e) => setChunkOverlap(parseInt(e.target.value, 10) || 200)}
            min={0}
            max={1000}
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label>Max File Size (KB)</label>
          <input
            type="number"
            value={maxFileSizeKb}
            onChange={(e) => setMaxFileSizeKb(parseInt(e.target.value, 10) || 500)}
            min={10}
            max={10000}
            disabled={isLoading}
          />
          <small style={{ color: '#888', fontSize: '0.8rem' }}>Files larger than this are skipped</small>
        </div>

        {gitCloneTimeoutMinutes !== undefined && setGitCloneTimeoutMinutes && (
          <div className="form-group">
            <label>Clone Timeout (min)</label>
            <input
              type="number"
              value={gitCloneTimeoutMinutes}
              onChange={(e) => {
                setGitCloneTimeoutMinutes(parseInt(e.target.value, 10) || 5);
                if (setTimeoutManuallySet) setTimeoutManuallySet(true);
              }}
              min={1}
              max={480}
              disabled={isLoading}
            />
            <small style={{ color: '#888', fontSize: '0.8rem' }}>Auto-scales with history depth</small>
          </div>
        )}

        <div className="form-group">
          <label>OCR Mode</label>
          <select
            value={ocrMode}
            onChange={handleOcrModeChange}
            disabled={isLoading}
          >
            <option value="disabled">Disabled - Skip image files</option>
            <option value="tesseract">Tesseract - Fast traditional OCR</option>
            {ollamaAvailable && (
              <option value="ollama">Ollama Vision - Semantic OCR (uses global settings)</option>
            )}
          </select>
          <small style={{ color: '#888', fontSize: '0.8rem' }}>
            {ocrMode === 'ollama'
              ? 'Uses global OCR vision model setting.'
              : ocrMode === 'tesseract'
              ? 'Fast basic text extraction.'
              : 'Skip images.'}
          </small>
        </div>

        {reindexIntervalHours !== undefined && setReindexIntervalHours && (
          <div className="form-group">
            <label>Auto Re-index Interval</label>
            <select
              value={reindexIntervalHours}
              onChange={(e) => setReindexIntervalHours(parseInt(e.target.value, 10))}
              disabled={isLoading}
            >
              <option value={0}>Manual only</option>
              <option value={1}>Every hour</option>
              <option value={6}>Every 6 hours</option>
              <option value={12}>Every 12 hours</option>
              <option value={24}>Every 24 hours (daily)</option>
              <option value={168}>Every week</option>
            </select>
          </div>
        )}
      </div>

      {gitHistoryDepth !== undefined && setGitHistoryDepth && (
        <div className="form-group" style={{ marginBottom: '16px' }}>
          <label>Git History Depth</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input
              type="range"
              min={1}
              max={1001}
              value={gitHistoryDepth === 0 ? 1001 : gitHistoryDepth}
              onChange={handleHistoryChange}
              disabled={isLoading}
              style={{ flex: 1 }}
            />
            <span style={{ minWidth: '80px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
              {gitHistoryDepth === 0 ? 'Full' : gitHistoryDepth === 1 ? '1 (shallow)' : `${gitHistoryDepth} commits`}
            </span>
          </div>
          <small style={{ color: '#888', fontSize: '0.8rem' }}>
            {gitHistoryDepth === 0
              ? `Full history: Indexes all commits.${(commitHistory?.total_commits) ? ` (${commitHistory.total_commits.toLocaleString()} commits)` : ''} Large repos may take 30+ min to clone.`
              : gitHistoryDepth === 1
                ? 'Shallow clone: Only latest commit. Fastest, but no git history search.'
                : (() => {
                    const dateEstimate = getDepthDateEstimate ? getDepthDateEstimate(gitHistoryDepth, commitHistory) : null;
                    return dateEstimate
                      ? `Indexes last ${gitHistoryDepth} commits (${dateEstimate}). Clone time scales with depth.`
                      : `Indexes last ${gitHistoryDepth} commits. Clone time scales with depth.`;
                  })()}
          </small>
        </div>
      )}
    </>
  );
}
