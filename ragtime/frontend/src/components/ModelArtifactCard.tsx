import { lazy, Suspense, useContext, useEffect, useId, useRef, useState } from 'react';
import { AlertCircle, Box, Clock, FileArchive, FileBox } from 'lucide-react';
import { ArtifactNotice } from './ArtifactNotice';
import { openModelArtifactModal } from './modelArtifactModalStore';
import { useArtifactAccess } from '@/hooks/useArtifactAccess';
import { ModelArtifactActions, ModelArtifactActionsHost } from './ModelArtifactActions';
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

interface ModelArtifactCardProps {
  artifact: ModelArtifact;
}

export function ModelArtifactCard({ artifact }: ModelArtifactCardProps) {
  const expiry = new Date(artifact.expires_at);
  const expiryInfo = formatArtifactExpiry(artifact.expires_at);
  const { state, refresh, setState } = useArtifactAccess(artifact.id, artifact.expires_at);
  const hasActionsHost = useContext(ModelArtifactActionsHost);
  const fallbackId = useId();
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
  const handleImageError = async () => {
    setImageFailed(true);
    setImageLoading(false);
    await refresh();
  };

  const isReady = state === 'ready';
  const NativeIcon = artifact.format.toLowerCase() === 'zip' ? FileArchive : FileBox;

  return (
    <>
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
                {expiryInfo.soon && state !== 'expired' && <Clock size={12} aria-hidden="true" />}
                {state === 'expired' ? `Expired ${expiry.toLocaleDateString()}` : expiryInfo.label}
              </time>
            </div>
          </div>
        </div>
        {!isReady ? (
          <ArtifactNotice
            state={{ reason: state, expiresAt: artifact.expires_at }}
            onRetry={() => {
              if (state === 'error') void refresh();
              else window.location.assign(artifact.page_url);
            }}
          />
        ) : (
          <>
            {artifact.kind === 'native' && (
              <div className="model-artifact-native" data-model-artifact-native>
                <NativeIcon size={40} aria-hidden="true" />
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
                    <AlertCircle size={16} aria-hidden="true" /> Couldn't load the 3D view. Please
                    try again.
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
                      poster={artifact.preview_image_url}
                      onClose={closeViewer}
                      onExpand={() =>
                        openModelArtifactModal(artifact, document.activeElement as HTMLElement)
                      }
                      onError={(error: unknown) => {
                        const status =
                          error instanceof ArtifactRequestError
                            ? artifactStateFromStatus(error.status)
                            : 'error';
                        if (status !== 'error') setState(status);
                        else
                          setViewerError(typeof error === 'string' ? error : 'Please try again.');
                        releaseModelViewer(viewerKey);
                      }}
                    />
                  </Suspense>
                ) : (
                  <div
                    className={`model-artifact-media${preview && !imageFailed ? ' model-artifact-media-has-preview' : ''}`}
                    data-model-artifact-media
                  >
                    {imageLoading && <div className="model-artifact-skeleton" aria-hidden="true" />}
                    {preview && !imageFailed ? (
                      <button
                        type="button"
                        className="model-artifact-image-button"
                        data-action="zoom-image"
                        aria-label={`View larger: ${artifact.file_name}`}
                        onClick={(event) => openModelArtifactModal(artifact, event.currentTarget)}
                      >
                        <img
                          className="model-artifact-image"
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
                          {imageFailed
                            ? 'Preview image unavailable'
                            : artifact.format.toUpperCase()}
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
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
      {!hasActionsHost && (
        <div
          className="chat-message-actions chat-message-actions-left visible"
          data-model-artifact-actions-fallback
        >
          <ModelArtifactActions
            artifacts={[artifact]}
            menuId={`model-artifact-menu-${fallbackId}`}
          />
        </div>
      )}
    </>
  );
}
