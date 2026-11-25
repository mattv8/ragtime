import { CheckCircle, XCircle } from 'lucide-react';
import { getSolidworksErrorGuidance } from '../api/solidworks';
import type { ToolTestResponse } from '../types/api';

export interface SolidWorksTestResultProps {
  result: ToolTestResponse | null;
}

function describeVersions(value: unknown): string {
  if (!Array.isArray(value)) return 'Unknown';
  return value.length > 0 ? value.join(', ') : 'None found';
}

function describeSession(value: unknown): string {
  if (value === true) return 'Logged in';
  if (value === false) return 'Not logged in';
  return 'Unknown';
}

function describeText(value: unknown): string {
  return typeof value === 'string' && value !== '' ? value : 'Unknown';
}

export function SolidWorksTestResult({ result }: SolidWorksTestResultProps) {
  if (!result) return null;
  const details = result.details ?? {};
  const code = typeof details.error_code === 'string' ? details.error_code : undefined;
  const guidance = result.success ? null : getSolidworksErrorGuidance(code);
  const tone = result.success ? 'success' : 'danger';
  const Icon = result.success ? CheckCircle : XCircle;
  // Hide a guidance title that only repeats the message shown above it.
  const showGuidanceTitle = guidance !== null && guidance.title !== result.message;

  return (
    <div
      id="solidworks-test-result"
      className="sw-panel"
      data-testid="solidworks-test-result"
      data-sw-test-state={result.success ? 'success' : 'error'}
      data-sw-error-code={code}
      role={result.success ? 'status' : 'alert'}
    >
      <header className="sw-panel-header">
        <h3>Test result</h3>
        <span className="sw-pill" data-tone={tone}>
          <Icon size={14} aria-hidden="true" />
          {result.success ? 'Passed' : 'Failed'}
        </span>
      </header>
      <p>{result.message}</p>
      {result.success ? (
        <dl id="solidworks-test-facts" className="rc-facts-list">
          <div>
            <dt>Agent version</dt>
            <dd>{describeText(details.host_version)}</dd>
          </div>
          <div>
            <dt>Account</dt>
            <dd>{describeText(details.account)}</dd>
          </div>
          <div>
            <dt>SolidWorks found</dt>
            <dd>{describeVersions(details.solidworks_versions)}</dd>
          </div>
          <div>
            <dt>Desktop session</dt>
            <dd>{describeSession(details.session_interactive)}</dd>
          </div>
        </dl>
      ) : (
        <>
          {guidance && (
            <div id="solidworks-test-guidance">
              {showGuidanceTitle && <strong>{guidance.title}</strong>}
              <p>{guidance.hint}</p>
            </div>
          )}
          {code && (
            <p>
              Error code: <code className="font-mono">{code}</code>
            </p>
          )}
          <details>
            <summary>Show error details</summary>
            <div className="rc-command-block">
              <pre>{JSON.stringify(details, null, 2)}</pre>
            </div>
          </details>
        </>
      )}
    </div>
  );
}
