import { createContext, useEffect, useRef, useState } from 'react';
import { Download, FileBox, Maximize2 } from 'lucide-react';
import { useArtifactAccess } from '@/hooks/useArtifactAccess';
import { formatArtifactSize, parseModelArtifact, type ModelArtifact } from '@/utils/modelArtifact';
import type { ChatMessage } from '@/types';
import { ARTIFACT_UNAVAILABLE_TITLES } from './ArtifactNotice';
import { openModelArtifactModal } from './modelArtifactModalStore';
import { isViewableModelArtifact } from './modelViewerSupport';

export const ModelArtifactActionsHost = createContext(false);

const artifactsByEvents = new WeakMap<
  NonNullable<ChatMessage['events']>,
  { length: number; result: ModelArtifact[] }
>();

export function messageModelArtifacts(events: ChatMessage['events']): ModelArtifact[] {
  if (!events) return [];
  const cached = artifactsByEvents.get(events);
  if (cached && cached.length === events.length) return cached.result;
  const artifacts = new Map<string, ModelArtifact>();
  for (const event of events) {
    if (event.type !== 'tool') continue;
    if (
      String(event.presentation?.visibility ?? '')
        .trim()
        .toLowerCase() === 'hidden'
    )
      continue;
    const artifact = parseModelArtifact(event);
    if (artifact) artifacts.set(artifact.id, artifact);
  }
  const result = [...artifacts.values()];
  artifactsByEvents.set(events, { length: events.length, result });
  return result;
}

function ArtifactGroup({
  artifact,
  onView,
}: {
  artifact: ModelArtifact;
  onView?: (artifact: ModelArtifact) => void;
}) {
  const { state } = useArtifactAccess(artifact.id, artifact.expires_at);
  const reason = state === 'ready' ? undefined : ARTIFACT_UNAVAILABLE_TITLES[state];
  const viewReason = state === 'signin' ? undefined : reason;
  const viewLabel = `View ${isViewableModelArtifact(artifact) ? '3D model ' : ''}${artifact.file_name}`;
  const downloadLabel = `Download ${artifact.file_name}`;
  const size = formatArtifactSize(artifact.size_bytes);
  return (
    <span
      className="chat-message-artifact-group"
      role="group"
      aria-label={artifact.file_name}
      aria-disabled={viewReason ? true : undefined}
      title={reason}
      data-artifact-id={artifact.id}
      data-artifact-format={artifact.format}
    >
      <span className="badge badge-muted chat-message-artifact-format">
        {artifact.format.toUpperCase()}
      </span>
      <span className="chat-message-artifact-name" title={artifact.file_name}>
        {artifact.file_name}
      </span>
      <button
        type="button"
        className="chat-action-icon-btn"
        data-action="open-viewer"
        aria-label={viewLabel}
        title={viewReason ?? `${viewLabel} (${size})`}
        aria-disabled={viewReason ? true : undefined}
        onClick={(event) => {
          if (!viewReason) {
            if (onView) onView(artifact);
            else openModelArtifactModal(artifact, event.currentTarget);
          }
        }}
      >
        <Maximize2 size={12} aria-hidden="true" />
      </button>
      {reason ? (
        <button
          type="button"
          className="chat-action-icon-btn"
          data-action="download"
          aria-label={downloadLabel}
          aria-disabled="true"
          title={state === 'signin' ? 'Sign in to download' : reason}
        >
          <Download size={12} aria-hidden="true" />
        </button>
      ) : (
        <a
          className="chat-action-icon-btn"
          data-action="download"
          href={artifact.content_url}
          download
          aria-label={downloadLabel}
          title={`${downloadLabel} (${size})`}
        >
          <Download size={12} aria-hidden="true" />
        </a>
      )}
    </span>
  );
}

export function ModelArtifactActions({
  artifacts,
  menuId,
}: {
  artifacts: ModelArtifact[];
  menuId: string;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = () => {
      setOpen(false);
      trigger.current?.focus();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    const onPointer = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) close();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, [open]);
  if (!artifacts.length) return null;
  return (
    <span
      ref={root}
      className="chat-message-artifact-actions"
      role="group"
      aria-label="Model files"
      data-message-artifacts
    >
      {artifacts.length < 3 ? (
        artifacts.map((artifact) => <ArtifactGroup key={artifact.id} artifact={artifact} />)
      ) : (
        <>
          <button
            ref={trigger}
            type="button"
            className="chat-action-icon-btn chat-message-artifact-disclosure"
            aria-label={`Model files, ${artifacts.length}`}
            aria-expanded={open}
            aria-controls={menuId}
            onClick={() => setOpen(!open)}
          >
            <FileBox size={12} aria-hidden="true" /> Model files
            <span className="badge badge-muted">{artifacts.length}</span>
          </button>
          {open && (
            <span className="chat-message-artifact-menu" id={menuId}>
              {artifacts.map((artifact) => (
                <ArtifactGroup
                  key={artifact.id}
                  artifact={artifact}
                  onView={(selected) => {
                    setOpen(false);
                    openModelArtifactModal(selected, trigger.current);
                  }}
                />
              ))}
            </span>
          )}
        </>
      )}
    </span>
  );
}
