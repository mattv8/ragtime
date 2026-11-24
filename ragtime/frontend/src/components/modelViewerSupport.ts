import { createElement, useSyncExternalStore } from 'react';
import type { ModelArtifact } from '@/utils/modelArtifact';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

export function ModelViewerFallback() {
  return createElement(
    'div',
    { className: 'model-artifact-viewer model-artifact-viewer-fallback', role: 'status' },
    createElement(MiniLoadingSpinner, { ariaHidden: true }),
    ' Loading 3D view…',
  );
}

let activeViewer: string | null = null;
const listeners = new Set<() => void>();

export function isViewableModelArtifact(artifact: ModelArtifact): boolean {
  return (
    artifact.kind === 'model' &&
    ['step', 'stl', '3mf', 'glb'].includes(artifact.format.toLowerCase())
  );
}

export function setActiveModelViewer(id: string | null): void {
  if (activeViewer === id) return;
  activeViewer = id;
  listeners.forEach((listener) => listener());
}

export function releaseModelViewer(id: string): void {
  if (activeViewer === id) setActiveModelViewer(null);
}

export function useActiveModelViewer(): string | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => activeViewer,
    () => null,
  );
}
