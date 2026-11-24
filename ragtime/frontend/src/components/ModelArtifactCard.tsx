import { lazy, Suspense, useCallback, useEffect, useId, useRef, useState } from 'react';
import { AlertCircle, Box, Download, ExternalLink, FileArchive, FileBox } from 'lucide-react';
import { api } from '@/api';
import { ApiError } from '@/api/client';
import {
  ArtifactRequestError,
  artifactStateFromStatus,
  formatArtifactExpiry,
  formatArtifactSize,
  type ModelArtifact,
} from '@/utils/modelArtifact';
import {
  isViewableModelArtifact,
  releaseModelViewer,
  setActiveModelViewer,
  useActiveModelViewer,
} from './modelViewerSupport';

const LazyModelViewer = lazy(() => import('./ModelViewer'));

type ArtifactState = 'ready' | 'expired' | 'signin' | 'forbidden' | 'notfound';

interface ModelArtifactCardProps {
  artifact: ModelArtifact;
  onZoomImage: (url: string, alt: string) => void;
}

export function ModelArtifactCard({ artifact, onZoomImage }: ModelArtifactCardProps) {
  const expiry = new Date(artifact.expires_at);
  const expiryInfo = formatArtifactExpiry(artifact.expires_at);
  const [remoteState, setState] = useState<ArtifactState>('ready');
  const state = expiryInfo.expired ? 'expired' : remoteState;
  const [imageFailed, setImageFailed] = useState(false);
  const [imageLoading, setImageLoading] = useState(Boolean(artifact.preview_image_url));
  const [viewerError, setViewerError] = useState('');
  const activeViewer = useActiveModelViewer();
  const viewerKey = `card:${artifact.id}:${useId()}`;
  const loadButton = useRef<HTMLButtonElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const viewerOpen = activeViewer === viewerKey && state === 'ready';
  const localExpiry = expiry.toLocaleString();
  const preview = artifact.preview_image_url;

  useEffect(() => () => releaseModelViewer(viewerKey), [viewerKey]);
  useEffect(() => {
    if (state !== 'ready') releaseModelViewer(viewerKey);
  }, [viewerKey, state]);
  useEffect(() => {
    if (viewerOpen) stageRef.current?.focus();
  }, [viewerOpen]);

  const closeViewer = () => {
    releaseModelViewer(viewerKey);
    requestAnimationFrame(() => loadButton.current?.focus());
  };
  const loadViewer = () => {
    setViewerError('');
    setActiveModelViewer(viewerKey);
  };
  const checkAccess = useCallback(async (): Promise<ArtifactState | null> => {
    try {
      const metadata = await api.getModelArtifact(artifact.id);
      return metadata.expired ? 'expired' : null;
    } catch (error) {
      const status =
        error instanceof ApiError || error instanceof ArtifactRequestError
          ? artifactStateFromStatus(error.status)
          : 'error';
      return status === 'error' ? null : status;
    }
  }, [artifact.id]);
  useEffect(() => {
    if (Date.parse(artifact.expires_at) < Date.now()) return;
    let cancelled = false;
    void checkAccess().then((result) => {
      if (!cancelled && result) setState(result);
    });
    return () => {
      cancelled = true;
    };
  }, [checkAccess, artifact.expires_at]);

  const handleImageError = async () => {
    setImageFailed(true);
    setImageLoading(false);
    const result = await checkAccess();
    if (result) setState(result);
  };

  const notice = {
    expired: `This export expired on ${expiry.toLocaleDateString()}. Ask for a new export.`,
    signin: 'Sign in to view or download this model.',
    forbidden: "You don't have access to this export.",
    notfound: 'This export is no longer available.',
  };
  const isReady = state === 'ready';
  const NativeIcon = artifact.format.toLowerCase() === 'zip' ? FileArchive : FileBox;

  return (
    <div
      className={`model-artifact-card${state === 'expired' ? ' model-artifact-card-expired' : ''}`}
      id={`model-artifact-card-${artifact.id}`}
      data-model-artifact-card
      data-artifact-id={artifact.id}
      data-artifact-kind={artifact.kind}
      data-artifact-format={artifact.format}
      data-state={state}
    >
      <div className="model-artifact-header">
        <div className="model-artifact-heading">
          <div className="model-artifact-title" title={artifact.file_name}>
            {artifact.file_name}
          </div>
          <div className="model-artifact-meta">
            <span className="badge badge-muted model-artifact-format">
              {artifact.format.toUpperCase()}
            </span>
            <span className="model-artifact-size">{formatArtifactSize(artifact.size_bytes)}</span>
            <time
              className={`model-artifact-expiry${expiryInfo.soon && state !== 'expired' ? ' model-artifact-expiry-soon' : ''}`}
              dateTime={artifact.expires_at}
              title={`Expires ${localExpiry}`}
            >
              {state === 'expired' ? `Expired ${expiry.toLocaleDateString()}` : expiryInfo.label}
            </time>
          </div>
        </div>
        {isReady && (
          <div className="model-artifact-actions" data-model-artifact-actions>
            <a
              className="btn btn-secondary btn-sm"
              href={artifact.content_url}
              download
              data-action="download"
              aria-label={`Download ${artifact.file_name}`}
            >
              <Download size={16} />
              <span>Download</span>
            </a>
            <a
              className="btn btn-secondary btn-sm"
              href={artifact.page_url}
              target="_blank"
              rel="noopener"
              data-action="open-page"
              aria-label={`Open in page: ${artifact.file_name}`}
            >
              <ExternalLink size={16} />
              <span>Open in page</span>
            </a>
          </div>
        )}
      </div>
      {!isReady ? (
        <div className="model-artifact-notice" role="status">
          <p>{notice[state]}</p>
          {state === 'signin' && (
            <a href={artifact.page_url} className="btn btn-secondary btn-sm" data-action="sign-in">
              Sign in
            </a>
          )}
        </div>
      ) : (
        <>
          {artifact.kind === 'native' && (
            <div className="model-artifact-native" data-model-artifact-native>
              <NativeIcon size={24} />
              <div>
                <p>
                  Native SolidWorks file. Open it in SolidWorks to edit. Assemblies include
                  referenced files.
                </p>
              </div>
            </div>
          )}
          {(artifact.kind !== 'native' || preview) && (
            <div
              className="model-artifact-stage"
              data-model-artifact-stage
              ref={stageRef}
              tabIndex={-1}
              aria-busy={imageLoading && !viewerOpen}
            >
              {viewerError && (
                <div className="model-artifact-error" role="alert" title={viewerError}>
                  <AlertCircle size={16} aria-hidden="true" /> Couldn't load the 3D view. Please try
                  again.
                </div>
              )}
              {viewerOpen ? (
                <Suspense
                  fallback={
                    <div className="model-artifact-viewer" role="status">
                      Loading 3D view…
                    </div>
                  }
                >
                  <LazyModelViewer
                    artifact={artifact}
                    onClose={closeViewer}
                    onError={(error: unknown) => {
                      const status =
                        error instanceof ArtifactRequestError
                          ? artifactStateFromStatus(error.status)
                          : 'error';
                      if (status !== 'error') setState(status);
                      else setViewerError(typeof error === 'string' ? error : 'Please try again.');
                      releaseModelViewer(viewerKey);
                    }}
                  />
                </Suspense>
              ) : (
                <>
                  {imageLoading && <div className="model-artifact-skeleton" aria-hidden="true" />}
                  {preview && !imageFailed ? (
                    <button
                      type="button"
                      className="model-artifact-image-button"
                      data-action="zoom-image"
                      aria-label={`Zoom preview of ${artifact.file_name}`}
                      onClick={() =>
                        onZoomImage(preview, `${artifact.file_name} preview, full view`)
                      }
                    >
                      <img
                        className="tool-call-screenshot-image model-artifact-image"
                        src={preview}
                        alt={`${artifact.file_name} preview`}
                        loading="lazy"
                        onLoad={() => setImageLoading(false)}
                        onError={handleImageError}
                      />
                    </button>
                  ) : (
                    <div
                      className="model-artifact-placeholder"
                      role="img"
                      aria-label={
                        imageFailed
                          ? 'Preview image unavailable'
                          : `No preview image for ${artifact.file_name}`
                      }
                    >
                      <Box size={40} />
                      <span>
                        {imageFailed ? 'Preview image unavailable' : artifact.format.toUpperCase()}
                      </span>
                    </div>
                  )}
                  {isViewableModelArtifact(artifact) && (
                    <button
                      ref={loadButton}
                      type="button"
                      className="btn btn-sm model-artifact-load3d"
                      data-action={viewerError ? 'retry' : 'load-3d'}
                      aria-label={
                        viewerError
                          ? `Try again: load 3D view of ${artifact.file_name}`
                          : `Load 3D view of ${artifact.file_name}`
                      }
                      onClick={loadViewer}
                    >
                      <Box size={16} /> {viewerError ? 'Try again' : 'Load 3D view'}
                    </button>
                  )}
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
