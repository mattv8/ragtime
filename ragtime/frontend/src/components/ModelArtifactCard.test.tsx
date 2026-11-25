import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelArtifactCard } from './ModelArtifactCard';
import { ArtifactRequestError, type ModelArtifact } from '@/utils/modelArtifact';
import { api } from '@/api';
import { ApiError } from '@/api/client';
import { openModelArtifactModal } from './modelArtifactModalStore';
import { clearArtifactAccessCache } from '@/hooks/useArtifactAccess';
import { ModelArtifactActionsHost } from './ModelArtifactActions';

vi.mock('./modelArtifactModalStore', () => ({ openModelArtifactModal: vi.fn() }));

vi.mock('./ModelViewer', () => ({
  default: ({
    artifact,
    onError,
    onClose,
    poster,
  }: {
    artifact: ModelArtifact;
    onError: (error: unknown) => void;
    onClose: () => void;
    poster?: string;
  }) => (
    <div data-testid="mock-model-viewer" data-poster={poster}>
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

beforeEach(() => {
  vi.spyOn(api, 'getModelArtifact').mockResolvedValue({
    ...artifact,
    expired: false,
    tool_config_name: null,
  });
});

afterEach(() => {
  cleanup();
  clearArtifactAccessCache();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('ModelArtifactCard', () => {
  it('renders fallback actions only without a message host', () => {
    const { container, rerender } = render(<ModelArtifactCard artifact={artifact} />);
    expect(container.querySelector('[data-model-artifact-actions-fallback]')).not.toBeNull();
    rerender(
      <ModelArtifactActionsHost.Provider value>
        <ModelArtifactCard artifact={artifact} />
      </ModelArtifactActionsHost.Provider>,
    );
    expect(container.querySelector('[data-model-artifact-actions-fallback]')).toBeNull();
  });
  it('opens the preview in the artifact modal', () => {
    render(<ModelArtifactCard artifact={artifact} />);
    const button = screen.getByRole('button', { name: 'View larger: bracket.step' });
    fireEvent.click(button);
    expect(openModelArtifactModal).toHaveBeenCalledWith(artifact, button);
  });
  it('shows a placeholder without a preview', () => {
    const { container } = render(
      <ModelArtifactCard artifact={{ ...artifact, preview_image_url: undefined }} />,
    );
    expect(screen.getByRole('img', { name: 'No preview image for bracket.step' })).toBeDefined();
    const media = container.querySelector('[data-model-artifact-media]');
    expect(media?.classList.contains('model-artifact-media-has-preview')).toBe(false);
    expect(
      media?.contains(screen.getByRole('button', { name: 'Load 3D view of bracket.step' })),
    ).toBe(true);
  });
  it('anchors the load button to the preview media', () => {
    const { container } = render(<ModelArtifactCard artifact={artifact} />);
    const media = container.querySelector('[data-model-artifact-media]');
    expect(media?.classList.contains('model-artifact-media-has-preview')).toBe(true);
    expect(media?.contains(screen.getByRole('button', { name: 'View larger: bracket.step' }))).toBe(
      true,
    );
    expect(
      media?.contains(screen.getByRole('button', { name: 'Load 3D view of bracket.step' })),
    ).toBe(true);
  });
  it('shows expiry and removes actions', () => {
    render(<ModelArtifactCard artifact={{ ...artifact, expires_at: '2000-01-01T00:00:00Z' }} />);
    expect(screen.getByRole('status').textContent).toContain('This export expired');
    expect(screen.queryByRole('link')).toBeNull();
  });
  it('shows native file guidance', () => {
    render(
      <ModelArtifactCard
        artifact={{ ...artifact, kind: 'native', preview_image_url: undefined }}
      />,
    );
    expect(screen.getByText(/Native SolidWorks file/)).toBeDefined();
  });
  it('keeps only the most recently opened viewer mounted', async () => {
    const otherId = 'b'.repeat(32);
    const other = { ...artifact, id: otherId, file_name: 'other.step' };
    render(
      <>
        <ModelArtifactCard artifact={artifact} />
        <ModelArtifactCard artifact={other} />
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
        <ModelArtifactCard artifact={artifact} />
        <ModelArtifactCard artifact={artifact} />
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
    render(<ModelArtifactCard artifact={{ ...artifact, kind: 'preview' }} />);
    expect(screen.queryByRole('button', { name: /Load 3D view/ })).toBeNull();
  });
  it('falls back to the preview and allows retry after a viewer error', async () => {
    render(<ModelArtifactCard artifact={artifact} />);
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
    render(<ModelArtifactCard artifact={artifact} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Close viewer' }));
    expect(screen.getByAltText('bracket.step preview')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Load 3D view of bracket.step' })).toBeDefined();
  });
  it('passes the preview poster to the viewer', async () => {
    render(<ModelArtifactCard artifact={artifact} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load 3D view of bracket.step' }));
    expect((await screen.findByTestId('mock-model-viewer')).getAttribute('data-poster')).toBe(
      artifact.preview_image_url,
    );
  });
  it('retries transient access errors without navigating', async () => {
    const request = vi
      .spyOn(api, 'getModelArtifact')
      .mockRejectedValueOnce(new Error('network'))
      .mockResolvedValueOnce({ ...artifact, expired: false, tool_config_name: null });
    render(<ModelArtifactCard artifact={artifact} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(screen.getByAltText('bracket.step preview')).toBeDefined());
    expect(request).toHaveBeenCalledTimes(2);
  });
  it('switches to expired when the model download returns 410', async () => {
    render(<ModelArtifactCard artifact={artifact} />);
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
    render(<ModelArtifactCard artifact={artifact} />);
    fireEvent.error(screen.getByAltText('bracket.step preview'));
    await waitFor(() => expect(screen.getByRole('status').textContent).toContain(message));
    if (status === 401) expect(screen.getByRole('button', { name: 'Sign in' })).toBeDefined();
    vi.unstubAllGlobals();
  });
});
