import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Download, FileArchive, Maximize2, Scan, X } from 'lucide-react';
import { api, type ModelArtifactMetadata } from '@/api/client';
import { useDialogFocusTrap } from '@/hooks/useDialogFocusTrap';
import {
  ArtifactRequestError,
  artifactStateFromStatus,
  formatArtifactExpiry,
  formatArtifactSize,
  type ModelArtifact,
} from '@/utils/modelArtifact';
import { ArtifactNotice, type ArtifactNoticeState } from './ArtifactNotice';
import { closeModelArtifactModal } from './modelArtifactModalStore';
import {
  isViewableModelArtifact,
  releaseModelViewer,
  setActiveModelViewer,
} from './modelViewerSupport';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

const LazyModelViewer = lazy(() => import('./ModelViewer'));

export function ModelArtifactModal({
  artifact,
  opener,
}: {
  artifact: ModelArtifact;
  opener: HTMLElement | null;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const initialFocusRef = useRef<HTMLButtonElement>(null);
  const mediaRef = useRef<HTMLDivElement>(null);
  const backdropDown = useRef(false);
  const [metadata, setMetadata] = useState<ModelArtifactMetadata | null>(null);
  const [notice, setNotice] = useState<ArtifactNoticeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [actualSize, setActualSize] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useDialogFocusTrap(dialogRef, initialFocusRef, { onEscape: closeModelArtifactModal });

  useEffect(() => {
    const key = `modal:${artifact.id}`;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    setActiveModelViewer(key);
    return () => {
      releaseModelViewer(key);
      document.body.style.overflow = overflow;
      const card = document.getElementById(`model-artifact-card-${artifact.id}`);
      const fallback = card?.querySelector<HTMLElement>('[data-action]') ?? card;
      const target = opener?.isConnected ? opener : fallback;
      if (target === card && card && !card.hasAttribute('tabindex')) card.tabIndex = -1;
      target?.focus();
    };
  }, [artifact.id, opener]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setNotice(null);
    api.getModelArtifact(artifact.id).then(
      (result) => {
        if (cancelled) return;
        setMetadata(result);
        if (result.expired) setNotice({ reason: 'expired', expiresAt: result.expires_at });
        setLoading(false);
      },
      (error: unknown) => {
        if (cancelled) return;
        setNotice({
          reason: artifactStateFromStatus(error instanceof ArtifactRequestError ? error.status : 0),
        });
        setLoading(false);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [artifact.id, attempt]);

  useEffect(() => {
    if (!loading && !notice && (artifact.kind === 'preview' || isViewableModelArtifact(artifact))) {
      mediaRef.current?.focus();
    }
  }, [artifact, loading, notice]);

  const info = metadata ?? artifact;
  const preview = artifact.kind === 'preview' ? info.content_url : info.preview_image_url;
  const state = loading ? 'loading' : (notice?.reason ?? 'ready');
  const loadingView = (
    <div className="model-artifact-modal-loading" role="status">
      <MiniLoadingSpinner ariaHidden /> Loading…
    </div>
  );

  return (
    <div
      className="modal-overlay model-artifact-modal-overlay"
      data-model-artifact-modal-overlay
      onMouseDown={(event) => {
        backdropDown.current = event.target === event.currentTarget;
      }}
      onClick={(event) => {
        if (backdropDown.current && event.target === event.currentTarget) closeModelArtifactModal();
        backdropDown.current = false;
      }}
    >
      <div
        ref={dialogRef}
        id={`model-artifact-modal-${artifact.id}`}
        className="modal-content model-artifact-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`model-artifact-modal-title-${artifact.id}`}
        data-model-artifact-modal
        data-artifact-id={artifact.id}
        data-artifact-kind={artifact.kind}
        data-state={state}
      >
        <header className="modal-header model-artifact-modal-header">
          <div className="model-artifact-modal-heading">
            <h3 id={`model-artifact-modal-title-${artifact.id}`} title={info.file_name}>
              {info.file_name}
            </h3>
            <div className="model-artifact-meta" data-model-artifact-meta>
              <span className="badge badge-muted model-artifact-format">
                {info.format.toUpperCase()}
              </span>
              <span>{formatArtifactSize(info.size_bytes)}</span>
              {metadata?.tool_config_name && (
                <span className="model-artifact-modal-tool-name">{metadata.tool_config_name}</span>
              )}
              {info.expires_at && (
                <time dateTime={info.expires_at}>
                  {formatArtifactExpiry(info.expires_at).label}
                </time>
              )}
            </div>
          </div>
          <div className="model-artifact-modal-actions">
            <a
              className="btn btn-secondary btn-sm"
              data-action="download"
              href={info.content_url}
              aria-label={`Download ${info.file_name}`}
              title={`Download ${info.file_name}`}
              download
            >
              <Download size={14} aria-hidden="true" />
              <span className="model-artifact-modal-download-label">Download</span>
            </a>
            <button
              ref={initialFocusRef}
              type="button"
              className="modal-close"
              data-action="close"
              aria-label="Close"
              onClick={closeModelArtifactModal}
            >
              <X size={18} aria-hidden="true" />
            </button>
          </div>
        </header>
        <div className="modal-body model-artifact-modal-body" data-model-artifact-modal-body>
          {loading ? (
            loadingView
          ) : notice ? (
            <ArtifactNotice
              state={notice}
              fileName={info.file_name}
              onRetry={() => {
                if (notice.reason === 'signin') window.location.assign(artifact.page_url);
                else setAttempt((value) => value + 1);
              }}
            />
          ) : isViewableModelArtifact(info) ? (
            <div
              className="model-artifact-modal-viewer"
              ref={mediaRef}
              tabIndex={0}
              aria-label={`3D view of ${info.file_name}`}
            >
              <Suspense fallback={loadingView}>
                <LazyModelViewer
                  artifact={info}
                  poster={artifact.preview_image_url}
                  height="100%"
                  showCloseButton={false}
                  allowFullscreen={false}
                  onError={(error) => {
                    if (error instanceof ArtifactRequestError)
                      setNotice({ reason: artifactStateFromStatus(error.status) });
                  }}
                />
              </Suspense>
            </div>
          ) : artifact.kind === 'preview' ? (
            <div className="model-artifact-modal-image-container">
              <div
                className={`model-artifact-modal-image-region${actualSize ? ' is-actual' : ''}`}
                ref={mediaRef}
                tabIndex={0}
                role="region"
                aria-label={`Image preview of ${info.file_name}`}
                onKeyDown={(event) => {
                  if (event.key.toLowerCase() === 'f') setActualSize(false);
                  if (event.key === '1') setActualSize(true);
                }}
                onClick={() => setActualSize((value) => !value)}
              >
                <img src={preview} alt={info.file_name} />
              </div>
              <div
                className="model-artifact-modal-image-toolbar"
                role="group"
                aria-label="Image zoom"
              >
                <button
                  type="button"
                  className="model-artifact-tool-btn"
                  aria-label="Fit to window"
                  aria-pressed={!actualSize}
                  onClick={() => setActualSize(false)}
                >
                  <Scan size={16} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="model-artifact-tool-btn"
                  aria-label="Actual size"
                  aria-pressed={actualSize}
                  onClick={() => setActualSize(true)}
                >
                  <Maximize2 size={16} aria-hidden="true" />
                </button>
              </div>
            </div>
          ) : (
            <div className="model-artifact-modal-native">
              <FileArchive size={32} aria-hidden="true" />
              {preview && <img src={preview} alt={`Preview of ${info.file_name}`} />}
              <p>Native SolidWorks file. Download to open it in SolidWorks.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
