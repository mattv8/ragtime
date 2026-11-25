import { useSyncExternalStore } from 'react';
import type { ModelArtifact } from '@/utils/modelArtifact';

export interface ModelArtifactModalState {
  artifact: ModelArtifact;
  /** Element that opened the modal; focus returns here on close when it is still in the document. */
  opener: HTMLElement | null;
}

let current: ModelArtifactModalState | null = null;
const listeners = new Set<() => void>();

function publish(next: ModelArtifactModalState | null): void {
  current = next;
  listeners.forEach((listener) => listener());
}

export function openModelArtifactModal(
  artifact: ModelArtifact,
  opener: HTMLElement | null = null,
): void {
  publish({ artifact, opener });
}

export function closeModelArtifactModal(): void {
  if (current) publish(null);
}

export function useModelArtifactModal(): ModelArtifactModalState | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => current,
    () => null,
  );
}
