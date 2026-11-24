import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ModelViewer from './ModelViewer';
import type { ModelArtifact } from '@/utils/modelArtifact';
import { ArtifactRequestError } from '@/utils/modelArtifact';

const mocks = vi.hoisted(() => ({
  load: vi.fn(),
  destroy: vi.fn(),
  lose: vi.fn(),
  orbit: vi.fn(),
  zoom: vi.fn(),
  update: vi.fn(),
  callbacks: null as null | { onModelFinished: (result: unknown, object: unknown) => void },
  disposeObject: vi.fn(),
}));
vi.mock('online-3d-viewer', () => ({
  RGBAColor: class {},
  RGBColor: class {},
  EdgeSettings: class {},
  InputFilesFromFileObjects: (files: File[]) => files,
  ImportSettings: class {},
  Direction: { Y: 1 },
  DisposeThreeObjects: mocks.disposeObject,
  ThreeModelLoader: class {
    LoadModel(files: File[], _settings: unknown, callbacks: typeof mocks.callbacks) {
      mocks.load(files);
      mocks.callbacks = callbacks;
    }
    Destroy = mocks.destroy;
  },
  Viewer: class {
    renderer = { forceContextLoss: mocks.lose };
    navigation = { Orbit: mocks.orbit, Zoom: mocks.zoom, Update: mocks.update };
    Init() {
      document.addEventListener('mousemove', () => {});
      document.addEventListener('mouseup', () => {});
      document.addEventListener('mouseleave', () => {});
    }
    Resize() {}
    SetBackgroundColor() {}
    SetEdgeSettings() {}
    Render() {}
    Destroy() {}
  },
}));

const artifact: ModelArtifact = {
  id: 'a'.repeat(32),
  kind: 'model',
  format: 'step',
  file_name: 'part.step',
  size_bytes: 8,
  expires_at: '2099-01-01T00:00:00Z',
  content_url: `/indexes/solidworks/artifacts/${'a'.repeat(32)}/content`,
  page_url: `/?artifact=${'a'.repeat(32)}`,
};

describe('ModelViewer', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['model']) }),
    );
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('fetches inline and loads a named file, then destroys the viewer', async () => {
    const { unmount } = render(<ModelViewer artifact={artifact} />);
    await waitFor(() => expect(mocks.load).toHaveBeenCalledOnce());
    expect(fetch).toHaveBeenCalledWith(
      `${artifact.content_url}?inline=1`,
      expect.objectContaining({ credentials: 'include' }),
    );
    expect(mocks.load.mock.calls[0][0][0].name).toBe(`${artifact.id}.step`);
    const parent = document.querySelector('.model-artifact-viewer-canvas')!;
    unmount();
    expect(mocks.destroy).toHaveBeenCalledOnce();
    expect(mocks.lose).toHaveBeenCalledOnce();
    expect(parent.childElementCount).toBe(0);
  });

  it('disposes a model that finishes after the viewer is closed', async () => {
    const { unmount } = render(<ModelViewer artifact={artifact} />);
    await waitFor(() => expect(mocks.load).toHaveBeenCalledOnce());
    const callback = mocks.callbacks!.onModelFinished;
    unmount();
    const model = {};
    expect(() => callback({}, model)).not.toThrow();
    expect(mocks.disposeObject).toHaveBeenCalledWith(model);
  });

  it('removes the navigation document listeners after repeated mounts', async () => {
    const add = vi.spyOn(document, 'addEventListener');
    const remove = vi.spyOn(document, 'removeEventListener');
    const windowAdd = vi.spyOn(window, 'addEventListener');
    const windowRemove = vi.spyOn(window, 'removeEventListener');
    for (let i = 0; i < 2; i++) {
      const { unmount } = render(<ModelViewer artifact={artifact} />);
      await waitFor(() => expect(mocks.load).toHaveBeenCalledTimes(i + 1));
      unmount();
    }
    for (const name of ['mousemove', 'mouseup', 'mouseleave']) {
      const handlers = add.mock.calls.filter(([type]) => type === name);
      expect(handlers).toHaveLength(2);
      for (const [, handler] of handlers) {
        expect(remove).toHaveBeenCalledWith(name, handler, undefined);
      }
    }
    const windowListeners = windowAdd.mock.calls.filter(([type]) => type === 'resize');
    for (const [type, handler, options] of windowListeners) {
      expect(windowRemove).toHaveBeenCalledWith(type, handler, options);
    }
    add.mockRestore();
    remove.mockRestore();
    windowAdd.mockRestore();
    windowRemove.mockRestore();
  });

  it('reports fetch failure to the parent', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false, status: 403 } as Response);
    const onError = vi.fn();
    render(<ModelViewer artifact={artifact} onError={onError} />);
    await waitFor(() => expect(onError).toHaveBeenCalledOnce());
    expect(onError.mock.calls[0][0]).toBeInstanceOf(ArtifactRequestError);
    expect(onError.mock.calls[0][0].status).toBe(403);
  });

  it('updates navigation after keyboard orbit and zoom in both directions', async () => {
    render(<ModelViewer artifact={artifact} />);
    await waitFor(() => expect(mocks.load).toHaveBeenCalledOnce());
    const canvas = screen.getByRole('group', { name: /Interactive 3D model/ });
    fireEvent.keyDown(canvas, { key: '+' });
    fireEvent.keyDown(canvas, { key: '-' });
    fireEvent.keyDown(canvas, { key: 'ArrowLeft' });
    expect(mocks.zoom.mock.calls).toEqual([[0.1], [-0.1]]);
    expect(mocks.orbit).toHaveBeenCalledWith(-5, 0);
    expect(mocks.update).toHaveBeenCalledTimes(3);
  });
});
