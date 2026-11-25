import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import {
  isViewableModelArtifact,
  releaseModelViewer,
  setActiveModelViewer,
  useActiveModelViewer,
} from './modelViewerSupport';
import type { ModelArtifact } from '@/utils/modelArtifact';

afterEach(() => setActiveModelViewer(null));

describe('model viewer support', () => {
  it('notifies subscribers and releases only the matching viewer', () => {
    const { result } = renderHook(() => useActiveModelViewer());
    act(() => setActiveModelViewer('one'));
    expect(result.current).toBe('one');
    act(() => releaseModelViewer('other'));
    expect(result.current).toBe('one');
    act(() => releaseModelViewer('one'));
    expect(result.current).toBeNull();
  });

  it.each(['step', 'stl', '3mf', 'glb'])('accepts model format %s', (format) => {
    expect(isViewableModelArtifact({ kind: 'model', format } as ModelArtifact)).toBe(true);
    expect(isViewableModelArtifact({ kind: 'native', format } as ModelArtifact)).toBe(false);
  });
});
