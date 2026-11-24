import { AlertCircle, Clock, FileX, Lock } from 'lucide-react';
import type { artifactStateFromStatus } from '@/utils/modelArtifact';

export type ArtifactUnavailableReason = ReturnType<typeof artifactStateFromStatus>;

export interface ArtifactNoticeState {
  reason: ArtifactUnavailableReason;
  expiresAt?: string;
  message?: string;
}

/** Short reason titles shared by the notice and the disabled message-bar actions. */
export const ARTIFACT_UNAVAILABLE_TITLES: Record<ArtifactUnavailableReason, string> = {
  expired: 'Export expired',
  forbidden: 'Access denied',
  notfound: 'Export not available',
  signin: 'Sign in to continue',
  error: "Couldn't load this export",
};

export function ArtifactNotice({
  state,
  onRetry,
  fileName,
  titleAs: Title = 'p',
}: {
  state: ArtifactNoticeState;
  onRetry?: () => void;
  fileName?: string;
  titleAs?: 'h1' | 'p';
}) {
  const { reason, expiresAt, message } = state;
  const date = expiresAt ? ` on ${new Date(expiresAt).toLocaleDateString()}` : '';
  const copy = {
    expired: { Icon: Clock, body: `This export expired${date}. Ask for a new export.` },
    forbidden: { Icon: Lock, body: "You don't have access to this export." },
    notfound: { Icon: FileX, body: 'This export is no longer available.' },
    signin: { Icon: Lock, body: 'Sign in to view or download this model.' },
    error: { Icon: AlertCircle, body: message ?? 'Request failed.' },
  }[reason];
  const Icon = copy.Icon;
  const title = ARTIFACT_UNAVAILABLE_TITLES[reason];
  return (
    <div
      className="model-artifact-page-state"
      role={reason === 'error' ? 'alert' : 'status'}
      aria-label={fileName}
    >
      <Icon size={40} aria-hidden="true" />
      <Title className="model-artifact-notice-title">{title}</Title>
      <p>{copy.body}</p>
      {(reason === 'signin' || reason === 'error') && onRetry && (
        <button type="button" className="btn btn-primary" onClick={onRetry}>
          {reason === 'signin' ? 'Sign in' : 'Try again'}
        </button>
      )}
    </div>
  );
}
