import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelArtifactCard } from './ModelArtifactCard';
import { ArtifactRequestError, type ModelArtifact } from '@/utils/modelArtifact';
import { api } from '@/api';
import { ApiError } from '@/api/client';

vi.mock('./ModelViewer', () => ({
  default: ({
    artifact,
    onError,
    onClose,
  }: {
    artifact: ModelArtifact;
    onError: (error: unknown) => void;
    onClose: () => void;
  }) => (
    <div data-testid="mock-model-viewer">
      <span>{artifact.file_name} viewer</span>
      <button onClick={() => onError('Parsing failed')}>Fail viewer</button>
      <button onClick={() => onError(new ArtifactRequestError(410))}>Expire viewer</button>
      <button onClick={onClose}>Close viewer</button>
    </div>
  ),
}));

const id = 'a'.repeat(32);
const artifact: ModelArtifact = {
  id,
  kind: 'model',
  format: 'step',
  file_name: 'bracket.step',
  size_bytes: 123,
  expires_at: '2099-01-01T00:00:00Z',
  content_url: `/indexes/solidworks/artifacts/${id}/content`,
  preview_image_url: `/indexes/solidworks/artifacts/${id}/content`,
  page_url: `/?artifact=${id}`,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('ModelArtifactCard', () => {
  it('shows actions and zooms the preview', () => {
    const zoom = vi.fn();
    render(<ModelArtifactCard artifact={artifact} onZoomImage={zoom} />);
    expect(screen.getByRole('link', { name: 'Download bracket.step' }).getAttribute('href')).toBe(
      artifact.content_url,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Zoom preview of bracket.step' }));
    expect(zoom).toHaveBeenCalledWith(
      artifact.preview_image_url,
      'bracket.step preview, full view',
    );
  });
  it('shows a placeholder without a preview', () => {
    render(
      <ModelArtifactCard
        artifact={{ ...artifact, preview_image_url: undefined }}
        onZoomImage={vi.fn()}
      />,
    );
    expect(screen.getByRole('img', { name: 'No preview image for bracket.step' })).toBeDefined();
  });
  it('shows expiry and removes actions', () => {
    render(
      <ModelArtifactCard
        artifact={{ ...artifact, expires_at: '2000-01-01T00:00:00Z' }}
        onZoomImage={vi.fn()}
      />,
    );
    expect(screen.getByRole('status').textContent).toContain('This export expired');
    expect(screen.queryByRole('link')).toBeNull();
  });
  it('shows native file guidance', () => {
    render(
      <ModelArtifactCard
        artifact={{ ...artifact, kind: 'native', preview_image_url: undefined }}
        onZoomImage={vi.fn()}
      />,
    );
    expect(screen.getByText(/Native SolidWorks file/)).toBeDefined();
  });
  it('keeps only the most recently opened viewer mounted', async () => {
    const otherId = 'b'.repeat(32);
    const other = { ...artifact, id: otherId, file_name: 'other.step' };
    render(
      <>
        <ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />
        <ModelArtifactCard artifact={other} onZoomImage={vi.fn()} />
      </>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    expect(await screen.findByText('bracket.step viewer')).toBeDefined();
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of other.step' }));
    expect(await screen.findByText('other.step viewer')).toBeDefined();
    expect(screen.queryByText('bracket.step viewer')).toBeNull();
  });
  it('keeps only one viewer when the same artifact is shown twice', async () => {
    const { container } = render(
      <>
        <ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />
        <ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />
      </>,
    );
    fireEvent.click(screen.getAllByRole('button', { name: 'Load 3D view of bracket.step' })[0]);
    await screen.findByTestId('mock-model-viewer');
    const cards = container.querySelectorAll('[data-model-artifact-card]');
    expect(cards[0].querySelector('[data-testid="mock-model-viewer"]')).not.toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    await waitFor(() =>
      expect(cards[1].querySelector('[data-testid="mock-model-viewer"]')).not.toBeNull(),
    );
    expect(cards[0].querySelector('[data-testid="mock-model-viewer"]')).toBeNull();
  });
  it('does not offer a viewer for native or preview artifacts', () => {
    render(<ModelArtifactCard artifact={{ ...artifact, kind: 'preview' }} onZoomImage={vi.fn()} />);
    expect(screen.queryByRole('button', { name: /Load 3D view/ })).toBeNull();
  });
  it('falls back to the preview and allows retry after a viewer error', async () => {
    render(<ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Fail viewer' }));
    expect(screen.getByRole('alert').textContent).toContain("Couldn't load the 3D view.");
    expect(screen.getByAltText('bracket.step preview')).toBeDefined();
    fireEvent.click(
      screen.getByRole('button', { name: 'Try again: load 3D view of bracket.step' }),
    );
    expect(await screen.findByRole('button', { name: 'Fail viewer' })).toBeDefined();
    expect(screen.queryByRole('alert')).toBeNull();
  });
  it('returns to the preview when the viewer is closed', async () => {
    render(<ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Close viewer' }));
    expect(screen.getByAltText('bracket.step preview')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Load 3D view of bracket.step' })).toBeDefined();
  });
  it('switches to expired when the model download returns 410', async () => {
    render(<ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Expire viewer' }));
    expect(screen.getByRole('status').textContent).toContain('This export expired');
    expect(screen.queryByRole('link', { name: 'Download bracket.step' })).toBeNull();
  });
  it.each([
    [401, 'Sign in'],
    [403, "You don't have access"],
    [404, 'no longer available'],
    [200, 'This export expired'],
  ] as const)('handles metadata status %s', async (status, message) => {
    const mock = vi.spyOn(api, 'getModelArtifact');
    if (status === 200)
      mock.mockResolvedValue({ ...artifact, expired: true, tool_config_name: null });
    else mock.mockRejectedValue(new ApiError('failed', status));
    render(<ModelArtifactCard artifact={artifact} onZoomImage={vi.fn()} />);
    fireEvent.error(screen.getByAltText('bracket.step preview'));
    await waitFor(() => expect(screen.getByRole('status').textContent).toContain(message));
    if (status === 401)
      expect(screen.getByRole('link', { name: 'Sign in' }).getAttribute('href')).toBe(
        artifact.page_url,
      );
    vi.unstubAllGlobals();
  });
});
