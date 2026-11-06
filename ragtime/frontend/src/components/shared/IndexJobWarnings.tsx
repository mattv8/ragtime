import type { IndexingWarnings } from '@/types';

const warningExplanations: Record<string, string> = {
  timeout: 'OCR did not finish before the time limit.',
  output_exhausted: 'OCR output ended before it was complete.',
  incomplete_response: 'OCR did not return a complete response.',
  invalid_output: 'OCR produced unusable text.',
};

function warningCountLabel(count: number): string {
  return `${count.toLocaleString()} ${count === 1 ? 'source' : 'sources'} skipped during OCR`;
}

export function getIndexJobWarningCount(warnings: IndexingWarnings | null | undefined): number {
  return warnings?.count ?? 0;
}

interface IndexJobWarningsProps {
  warnings: IndexingWarnings | null | undefined;
  jobIdentity: string;
}

export function IndexJobWarnings({ warnings, jobIdentity }: IndexJobWarningsProps) {
  const count = getIndexJobWarningCount(warnings);
  const items = warnings?.items ?? [];
  const summaryId = `${jobIdentity}-warning-summary`;
  const listId = `${jobIdentity}-warning-list`;

  if (count === 0) return null;

  return (
    <section
      id={`${jobIdentity}-warnings`}
      className="index-job-warnings"
      aria-labelledby={summaryId}
    >
      <div id={summaryId} className="status-message warning">
        {warningCountLabel(count)} in this run. The index may not reflect all current source
        content.
      </div>
      <h4>Sources skipped during OCR</h4>
      {items.length > 0 ? (
        <ul id={listId} className="index-job-warnings-list">
          {items.map((warning) => (
            <li
              key={warning.source}
              className="index-job-warning-item"
              data-index-job-warning={`${jobIdentity}:${warning.source}`}
            >
              <code className="index-job-warning-source">{warning.source}</code>
              <span>
                {warningExplanations[warning.category] ?? 'OCR could not process this source.'}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p>Warning details are unavailable for this job.</p>
      )}
      {items.length > 0 && count > items.length && (
        <p>
          Showing {items.length.toLocaleString()} of {count.toLocaleString()} warnings.
        </p>
      )}
    </section>
  );
}
