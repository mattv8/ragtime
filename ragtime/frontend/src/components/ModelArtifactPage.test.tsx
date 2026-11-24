import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelArtifactPage } from './ModelArtifactPage';
import type { ModelArtifactMetadata } from '@/api/client';
import { ArtifactRequestError, type ModelArtifact } from '@/utils/modelArtifact';

const { getModelArtifact } = vi.hoisted(() => ({ getModelArtifact: vi.fn() }));

vi.mock('@/api', () => ({
  api: { getModelArtifact },
}));

// Each button reports one viewer failure through the same onError contract as ModelViewer.
vi.mock('./ModelViewer', () => ({
  default: ({
    artifact,
    onError,
  }: {
    artifact: ModelArtifact;
    onError?: (error: unknown) => void;
  }) => (
    <div>
      <span data-testid="model-viewer">{artifact.file_name}</span>
      <button type="button" onClick={() => onError?.(new Error('boom'))}>
        Fail viewer
      </button>
      {[401, 403, 404, 410, 500].map((status) => (
        <button
          key={status}
          type="button"
          onClick={() => onError?.(new ArtifactRequestError(status))}
        >
          Fail viewer {status}
        </button>
      ))}
    </div>
  ),
}));

const id = 'a'.repeat(32);
const contentUrl = `/indexes/solidworks/artifacts/${id}/content`;
const expiresFar = '2099-01-01T00:00:00Z';

function metadata(overrides: Partial<ModelArtifactMetadata> = {}): ModelArtifactMetadata {
  return {
    id,
    kind: 'model',
    format: 'step',
    file_name: 'bracket.step',
    size_bytes: 2048,
    expires_at: expiresFar,
    content_url: contentUrl,
    preview_image_url: contentUrl,
    page_url: `/?artifact=${id}`,
    expired: false,
    tool_config_name: 'SolidWorks Desktop',
    ...overrides,
  };
}

function renderPage(artifactId = id) {
  return render(<ModelArtifactPage artifactId={artifactId} />);
}

function pageState(): string | null {
  return document.getElementById('model-artifact-page')?.getAttribute('data-state') ?? null;
}

beforeEach(() => {
  getModelArtifact.mockReset();
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ModelArtifactPage', () => {
  it('shows the loading state while metadata is pending', () => {
    getModelArtifact.mockReturnValue(new Promise(() => {}));
    renderPage();
    expect(screen.getByRole('status').textContent).toContain('Loading export');
    expect(pageState()).toBe('loading');
  });

  it('loads the viewer automatically for viewable models and exposes the page hooks', async () => {
    getModelArtifact.mockResolvedValue(metadata());
    renderPage();
    expect(await screen.findByTestId('model-viewer')).toBeDefined();
    const page = document.getElementById('model-artifact-page');
    expect(page?.getAttribute('data-artifact-id')).toBe(id);
    expect(document.querySelector('[data-model-artifact-page-section]')).not.toBeNull();
    expect(document.querySelector('[data-model-artifact-page-viewer]')).not.toBeNull();
    expect(pageState()).toBe('ready');
    expect(screen.getByRole('heading', { level: 1, name: 'bracket.step' })).toBeDefined();
    expect(screen.getByText('STEP')).toBeDefined();
    expect(screen.getByText('SolidWorks Desktop')).toBeDefined();
    expect(screen.getByRole('link', { name: 'Download bracket.step' }).getAttribute('href')).toBe(
      contentUrl,
    );
  });

  it('shows the preview image for preview artifacts without a viewer', async () => {
    getModelArtifact.mockResolvedValue(
      metadata({ kind: 'preview', format: 'png', preview_image_url: undefined }),
    );
    renderPage();
    const image = await screen.findByRole('img', { name: 'bracket.step preview' });
    expect(image.getAttribute('src')).toBe(contentUrl);
    expect(image.closest('[data-model-artifact-page-stage]')).not.toBeNull();
    expect(screen.queryByTestId('model-viewer')).toBeNull();
  });

  it('shows a skeleton until the preview image loads', async () => {
    getModelArtifact.mockResolvedValue(
      metadata({ kind: 'preview', format: 'png', preview_image_url: undefined }),
    );
    const { container } = renderPage();
    await screen.findByRole('img', { name: 'bracket.step preview' });
    expect(container.querySelector('.model-artifact-skeleton')).not.toBeNull();
    fireEvent.load(screen.getByRole('img', { name: 'bracket.step preview' }));
    expect(container.querySelector('.model-artifact-skeleton')).toBeNull();
  });

  it('shows native file guidance with download and no viewer', async () => {
    getModelArtifact.mockResolvedValue(
      metadata({ kind: 'native', format: 'zip', preview_image_url: undefined }),
    );
    renderPage();
    expect(await screen.findByText(/Native SolidWorks file/)).toBeDefined();
    expect(document.querySelector('[data-model-artifact-page-native]')).not.toBeNull();
    expect(screen.queryByTestId('model-viewer')).toBeNull();
    expect(screen.getByRole('link', { name: 'Download bracket.step' })).toBeDefined();
  });

  it('marks expiry within a day as soon, with an icon', async () => {
    const expiresIn5Hours = new Date(Date.now() + 5 * 3_600_000).toISOString();
    getModelArtifact.mockResolvedValue(metadata({ expires_at: expiresIn5Hours }));
    renderPage();
    const expiry = await screen.findByText('Expires in 5 hours');
    expect(expiry.className).toContain('model-artifact-expiry-soon');
    expect(expiry.querySelector('svg')).not.toBeNull();
  });

  it('reports an expired export without actions and without repeating the sentence', async () => {
    getModelArtifact.mockResolvedValue(
      metadata({ expired: true, expires_at: '2000-01-01T00:00:00Z' }),
    );
    renderPage();
    expect(await screen.findByRole('heading', { level: 1, name: 'Export expired' })).toBeDefined();
    expect(screen.getByText(/This export expired on .*Ask for a new export\./)).toBeDefined();
    expect(screen.queryByRole('link', { name: 'Download bracket.step' })).toBeNull();
    expect(pageState()).toBe('expired');
  });

  it('treats a past expiry timestamp as expired', async () => {
    getModelArtifact.mockResolvedValue(metadata({ expires_at: '2000-01-01T00:00:00Z' }));
    renderPage();
    expect(await screen.findByText(/Ask for a new export/)).toBeDefined();
  });

  it('reports forbidden for a 403 response', async () => {
    getModelArtifact.mockRejectedValue(new ArtifactRequestError(403));
    renderPage();
    expect(await screen.findByText("You don't have access to this export.")).toBeDefined();
    expect(pageState()).toBe('forbidden');
  });

  it('reports not found for a 404 response', async () => {
    getModelArtifact.mockRejectedValue(new ArtifactRequestError(404));
    renderPage();
    expect(await screen.findByRole('heading', { name: 'Export not available' })).toBeDefined();
    expect(screen.getByRole('link', { name: 'Back to chat' }).getAttribute('href')).toBe('/');
    expect(pageState()).toBe('notfound');
  });

  it('offers sign-in for a 401 response', async () => {
    getModelArtifact.mockRejectedValue(new ArtifactRequestError(401));
    renderPage();
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeDefined();
    expect(screen.getByText('Sign in to view or download this model.')).toBeDefined();
    expect(pageState()).toBe('signin');
  });

  it('shows a retryable error for other failures', async () => {
    getModelArtifact.mockRejectedValue(new Error('Network down'));
    renderPage();
    expect(await screen.findByRole('alert')).toBeDefined();
    expect(screen.getByText('Network down')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeDefined();
  });

  it('treats an invalid artifact id as not found without calling the API', async () => {
    renderPage('not-a-real-id');
    expect(await screen.findByRole('heading', { name: 'Export not available' })).toBeDefined();
    expect(getModelArtifact).not.toHaveBeenCalled();
    expect(pageState()).toBe('notfound');
  });

  it('falls back to the image when the 3D viewer fails with a generic error', async () => {
    getModelArtifact.mockResolvedValue(metadata());
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Fail viewer' }));
    expect(screen.getByRole('alert').textContent).toContain("Couldn't load the 3D view.");
    expect(screen.getByRole('alert').getAttribute('title')).toBe('boom');
    expect(screen.queryByTestId('model-viewer')).toBeNull();
    expect(screen.getByRole('img', { name: 'bracket.step preview' })).toBeDefined();
    expect(pageState()).toBe('ready');
  });

  it.each([
    [401, 'signin', 'Sign in to view or download this model.'],
    [403, 'forbidden', "You don't have access to this export."],
    [404, 'notfound', 'This export is no longer available.'],
    [410, 'expired', 'Ask for a new export.'],
  ])('moves the page to the %s viewer state', async (status, state, text) => {
    getModelArtifact.mockResolvedValue(metadata());
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: `Fail viewer ${status}` }));
    expect(await screen.findByText(text, { exact: false })).toBeDefined();
    expect(pageState()).toBe(state);
    expect(screen.queryByTestId('model-viewer')).toBeNull();
    expect(screen.queryByRole('link', { name: 'Download bracket.step' })).toBeNull();
  });

  it('treats a 500 from the viewer as a generic failure with the fallback image', async () => {
    getModelArtifact.mockResolvedValue(metadata());
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Fail viewer 500' }));
    expect(screen.getByRole('alert').textContent).toContain("Couldn't load the 3D view.");
    expect(screen.getByRole('alert').getAttribute('title')).toBe('Artifact request failed (500).');
    expect(pageState()).toBe('ready');
    expect(screen.getByRole('img', { name: 'bracket.step preview' })).toBeDefined();
  });

  it('shows a placeholder when the preview image fails to load', async () => {
    getModelArtifact.mockResolvedValue(
      metadata({ kind: 'preview', format: 'png', preview_image_url: undefined }),
    );
    renderPage();
    const image = await screen.findByRole('img', { name: 'bracket.step preview' });
    fireEvent.error(image);
    expect(screen.getByText('Preview image unavailable')).toBeDefined();
    expect(screen.getByRole('link', { name: 'Download bracket.step' })).toBeDefined();
  });
});
