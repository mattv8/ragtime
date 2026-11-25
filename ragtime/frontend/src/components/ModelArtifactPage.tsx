import { lazy, Suspense, useEffect, useState, type MouseEvent } from 'react';
import {
  AlertCircle,
  ArrowLeft,
  Box,
  Clock,
  Download,
  FileArchive,
  FileBox,
  type LucideIcon,
} from 'lucide-react';
import { api } from '@/api';
import type { ModelArtifactMetadata } from '@/api/client';
import {
  ArtifactRequestError,
  artifactStateFromStatus,
  formatArtifactExpiry,
  formatArtifactSize,
} from '@/utils/modelArtifact';
import { isViewableModelArtifact, ModelViewerFallback } from './modelViewerSupport';
import { ArtifactNotice } from './ArtifactNotice';

const LazyModelViewer = lazy(() => import('./ModelViewer'));

const ARTIFACT_ID_PATTERN = /^[a-f0-9]{32}$/i;

type BlockedReason = ReturnType<typeof artifactStateFromStatus>;

type PageState =
  | { status: 'loading' }
  | { status: 'ready'; artifact: ModelArtifactMetadata }
  | { status: 'blocked'; reason: BlockedReason; expiresAt?: string; message?: string };

function initialState(artifactId: string): PageState {
  return ARTIFACT_ID_PATTERN.test(artifactId)
    ? { status: 'loading' }
    : { status: 'blocked', reason: 'notfound' };
}

function stateForMetadata(artifact: ModelArtifactMetadata): PageState {
  if (artifact.expired || formatArtifactExpiry(artifact.expires_at).expired) {
    return { status: 'blocked', reason: 'expired', expiresAt: artifact.expires_at };
  }
  return { status: 'ready', artifact };
}

function stateForRequestError(error: unknown): PageState {
  if (error instanceof ArtifactRequestError) {
    return {
      status: 'blocked',
      reason: artifactStateFromStatus(error.status),
      message: error.message,
    };
  }
  return {
    status: 'blocked',
    reason: 'error',
    message: error instanceof Error ? error.message : 'Request failed.',
  };
}

function isNativeSolidWorksFormat(format: string): boolean {
  return format.startsWith('sld');
}

function goBack(event: MouseEvent<HTMLAnchorElement>) {
  const sameOriginReferrer =
    document.referrer !== '' && new URL(document.referrer).origin === window.location.origin;
  if (!sameOriginReferrer || window.history.length <= 1) return;
  event.preventDefault();
  window.history.back();
}

function PlaceholderStage({
  artifact,
  caption,
}: {
  artifact: ModelArtifactMetadata;
  caption?: string;
}) {
  const Icon: LucideIcon = isNativeSolidWorksFormat(artifact.format) ? FileBox : Box;
  const label = caption ?? `No preview image for ${artifact.file_name}`;
  return (
    <div className="model-artifact-page-stage-wrap">
      <div className="model-artifact-placeholder" role="img" aria-label={label}>
        <Icon size={40} aria-hidden="true" />
        <span>{artifact.format.toUpperCase()}</span>
      </div>
      {caption && <p className="model-artifact-caption">{caption}</p>}
    </div>
  );
}

function PreviewStage({ artifact }: { artifact: ModelArtifactMetadata }) {
  const src =
    artifact.preview_image_url ?? (artifact.kind === 'preview' ? artifact.content_url : undefined);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  if (!src) return <PlaceholderStage artifact={artifact} />;
  if (failed) return <PlaceholderStage artifact={artifact} caption="Preview image unavailable" />;
  return (
    <figure
      className="model-artifact-page-stage"
      data-model-artifact-page-stage
      aria-busy={!loaded}
    >
      {!loaded && <div className="model-artifact-skeleton" aria-hidden="true" />}
      <img
        className="model-artifact-image"
        src={src}
        alt={`${artifact.file_name} preview`}
        onLoad={() => setLoaded(true)}
        onError={() => setFailed(true)}
      />
    </figure>
  );
}

interface ArtifactBodyProps {
  artifact: ModelArtifactMetadata;
  onViewerError: (error: unknown) => void;
}

function ArtifactBody({ artifact, onViewerError }: ArtifactBodyProps) {
  const [viewerMessage, setViewerMessage] = useState<string | null>(null);
  const showViewer =
    viewerMessage === null && artifact.kind === 'model' && isViewableModelArtifact(artifact);

  if (showViewer) {
    return (
      <div className="model-artifact-page-viewer" data-model-artifact-page-viewer>
        <Suspense fallback={<ModelViewerFallback />}>
          <LazyModelViewer
            artifact={artifact}
            height="100%"
            showCloseButton={false}
            onError={(error: unknown) => {
              if (error instanceof ArtifactRequestError) {
                const reason = artifactStateFromStatus(error.status);
                if (reason !== 'error') {
                  onViewerError(error);
                  return;
                }
              }
              setViewerMessage(error instanceof Error ? error.message : 'Unknown error.');
            }}
          />
        </Suspense>
      </div>
    );
  }

  return (
    <>
      {viewerMessage !== null && (
        <p className="model-artifact-error" role="alert" title={viewerMessage}>
          <AlertCircle size={16} aria-hidden="true" />
          Couldn't load the 3D view.
        </p>
      )}
      {artifact.kind === 'native' && (
        <div className="model-artifact-native" data-model-artifact-page-native>
          {artifact.format === 'zip' ? (
            <FileArchive size={40} aria-hidden="true" />
          ) : (
            <FileBox size={40} aria-hidden="true" />
          )}
          <p>
            Native SolidWorks file. Open it in SolidWorks to edit. Assemblies include referenced
            files.
          </p>
        </div>
      )}
      {(artifact.kind !== 'native' || artifact.preview_image_url) && (
        <PreviewStage artifact={artifact} />
      )}
    </>
  );
}

function ArtifactHeader({ artifact }: { artifact: ModelArtifactMetadata }) {
  const expiry = formatArtifactExpiry(artifact.expires_at);
  return (
    <header className="model-artifact-page-header">
      <div className="model-artifact-page-nav">
        <a className="btn btn-secondary btn-sm" href="/" onClick={goBack}>
          <ArrowLeft size={14} aria-hidden="true" />
          Back to chat
        </a>
        <a
          className="btn btn-primary btn-sm"
          href={artifact.content_url}
          aria-label={`Download ${artifact.file_name}`}
          data-action="download"
        >
          <Download size={14} aria-hidden="true" />
          Download
        </a>
      </div>
      <div className="model-artifact-page-heading">
        <h1 className="model-artifact-page-title" title={artifact.file_name}>
          {artifact.file_name}
        </h1>
        <div className="model-artifact-meta">
          <span className="badge badge-muted model-artifact-format">
            {artifact.format.toUpperCase()}
          </span>
          <span className="model-artifact-size">{formatArtifactSize(artifact.size_bytes)}</span>
          {artifact.tool_config_name && (
            <span className="model-artifact-tool">{artifact.tool_config_name}</span>
          )}
          <time
            className={
              expiry.soon
                ? 'model-artifact-expiry model-artifact-expiry-soon'
                : 'model-artifact-expiry'
            }
            dateTime={artifact.expires_at}
          >
            {expiry.soon && <Clock size={12} aria-hidden="true" />}
            {expiry.label}
          </time>
        </div>
      </div>
    </header>
  );
}

export function ModelArtifactPage({ artifactId }: { artifactId: string }) {
  const [state, setState] = useState<PageState>(() => initialState(artifactId));

  useEffect(() => {
    if (!ARTIFACT_ID_PATTERN.test(artifactId)) return;
    let cancelled = false;
    api.getModelArtifact(artifactId).then(
      (artifact) => {
        if (!cancelled) setState(stateForMetadata(artifact));
      },
      (error: unknown) => {
        if (!cancelled) setState(stateForRequestError(error));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [artifactId]);

  // The viewer reports 401/403/404/410 for the same artifact; the whole page moves to that state.
  const handleViewerError = (error: unknown) => {
    if (!(error instanceof ArtifactRequestError)) return;
    setState((current) =>
      current.status === 'ready'
        ? {
            status: 'blocked',
            reason: artifactStateFromStatus(error.status),
            expiresAt: current.artifact.expires_at,
          }
        : stateForRequestError(error),
    );
  };

  const dataState = state.status === 'blocked' ? state.reason : state.status;

  return (
    <main
      id="model-artifact-page"
      className="model-artifact-page"
      data-artifact-id={artifactId}
      data-state={dataState}
    >
      {state.status === 'loading' && (
        <div className="model-artifact-page-state" role="status">
          <div className="spinner" aria-hidden="true" />
          <p>Loading export…</p>
        </div>
      )}
      {state.status === 'blocked' && (
        <>
          <ArtifactNotice state={state} titleAs="h1" onRetry={() => window.location.reload()} />
          {state.reason !== 'signin' && state.reason !== 'error' && (
            <a className="btn btn-secondary" href="/">
              Back to chat
            </a>
          )}
        </>
      )}
      {state.status === 'ready' && (
        <>
          <ArtifactHeader artifact={state.artifact} />
          <section
            className="model-artifact-page-body"
            data-model-artifact-page-section
            aria-label="Model preview"
          >
            <ArtifactBody artifact={state.artifact} onViewerError={handleViewerError} />
          </section>
        </>
      )}
    </main>
  );
}
