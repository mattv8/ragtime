import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ArtifactRequestError } from '@/utils/modelArtifact';
import { ModelArtifactModalHost } from './ModelArtifactModalHost';
import { closeModelArtifactModal, openModelArtifactModal } from './modelArtifactModalStore';
import { useActiveModelViewer } from './modelViewerSupport';

const { getModelArtifact } = vi.hoisted(() => ({ getModelArtifact: vi.fn() }));
vi.mock('@/api/client', () => ({ api: { getModelArtifact } }));
vi.mock('./ModelViewer', () => ({
  default: ({ poster }: { poster?: string }) => <div data-testid="viewer" data-poster={poster} />,
}));

const artifact = {
  id: 'a'.repeat(32),
  kind: 'preview' as const,
  format: 'png',
  file_name: 'part.png',
  size_bytes: 1024,
  expires_at: '2099-01-01T00:00:00Z',
  content_url: '/image',
  page_url: '#signin',
  expired: false,
  tool_config_name: 'Desktop',
};

beforeEach(() => {
  getModelArtifact.mockResolvedValue(artifact);
});
afterEach(() => {
  closeModelArtifactModal();
  cleanup();
  vi.clearAllMocks();
  document.body.style.overflow = '';
  window.location.hash = '';
});

function open(opener: HTMLElement | null = null) {
  render(<ModelArtifactModalHost />);
  openModelArtifactModal(artifact, opener);
}

describe('ModelArtifactModal', () => {
  it('labels the dialog, locks scroll, traps focus, and restores focus on Escape', async () => {
    document.body.style.overflow = 'auto';
    const opener = document.createElement('button');
    document.body.append(opener);
    opener.focus();
    open(opener);
    const dialog = await screen.findByRole('dialog', { name: 'part.png' });
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-labelledby')).toBe(
      `model-artifact-modal-title-${artifact.id}`,
    );
    expect(screen.getByRole('link', { name: 'Download part.png' }).getAttribute('title')).toBe(
      'Download part.png',
    );
    expect(document.body.style.overflow).toBe('hidden');
    const close = screen.getByRole('button', { name: 'Close' });
    close.focus();
    fireEvent.keyDown(document, { key: 'Tab' });
    expect(dialog.contains(document.activeElement)).toBe(true);
    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(document.activeElement).toBe(opener);
    expect(document.body.style.overflow).toBe('auto');
    opener.remove();
  });

  it('returns focus to the card when the opener is removed', async () => {
    const card = document.createElement('div');
    card.id = `model-artifact-card-${artifact.id}`;
    card.innerHTML = '<button data-action="open">Open</button>';
    document.body.append(card);
    const opener = document.createElement('button');
    document.body.append(opener);
    open(opener);
    await screen.findByRole('dialog');
    opener.remove();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await waitFor(() => expect(document.activeElement).toBe(card.firstElementChild));
    card.remove();
  });

  it('requires both backdrop events to target the overlay', async () => {
    open();
    const dialog = await screen.findByRole('dialog');
    const overlay = dialog.parentElement as HTMLElement;
    fireEvent.mouseDown(dialog);
    fireEvent.click(overlay);
    expect(screen.getByRole('dialog')).toBeDefined();
    fireEvent.mouseDown(overlay);
    fireEvent.click(overlay);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('supports image zoom buttons, keys and click', async () => {
    open();
    const image = await screen.findByRole('region', { name: 'Image preview of part.png' });
    const fit = screen.getByRole('button', { name: 'Fit to window' });
    const actual = screen.getByRole('button', { name: 'Actual size' });
    expect(fit.classList.contains('model-artifact-tool-btn')).toBe(true);
    expect(actual.classList.contains('model-artifact-tool-btn')).toBe(true);
    fireEvent.click(actual);
    expect(actual.getAttribute('aria-pressed')).toBe('true');
    fireEvent.keyDown(image, { key: 'f' });
    expect(fit.getAttribute('aria-pressed')).toBe('true');
    fireEvent.keyDown(image, { key: '1' });
    expect(actual.getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(image);
    expect(fit.getAttribute('aria-pressed')).toBe('true');
  });

  it.each([
    [401, 'Sign in to continue'],
    [410, 'Export expired'],
  ])('shows the %s notice', async (status, title) => {
    getModelArtifact.mockRejectedValue(new ArtifactRequestError(status));
    open();
    expect(await screen.findByText(title)).toBeDefined();
    expect(screen.queryByRole('heading', { name: title })).toBeNull();
  });

  it('navigates to the artifact page to sign in', async () => {
    getModelArtifact.mockRejectedValue(new ArtifactRequestError(401));
    open();
    fireEvent.click(await screen.findByRole('button', { name: 'Sign in' }));
    expect(window.location.hash).toBe('#signin');
    expect(getModelArtifact).toHaveBeenCalledTimes(1);
  });

  it('retries metadata after a transient error', async () => {
    getModelArtifact.mockRejectedValueOnce(new Error('network')).mockResolvedValueOnce(artifact);
    open();
    fireEvent.click(await screen.findByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('region', { name: 'Image preview of part.png' })).toBeDefined();
    expect(getModelArtifact).toHaveBeenCalledTimes(2);
  });

  it('passes the original preview as the viewer poster', async () => {
    const model = {
      ...artifact,
      kind: 'model' as const,
      format: 'step',
      preview_image_url: '/poster',
    };
    getModelArtifact.mockResolvedValue(model);
    render(<ModelArtifactModalHost />);
    openModelArtifactModal(model, null);
    expect((await screen.findByTestId('viewer')).getAttribute('data-poster')).toBe('/poster');
    expect(document.querySelector('.model-artifact-modal-viewer-preview')).toBeNull();
  });

  it('shows one download link for native artifacts', async () => {
    const native = {
      ...artifact,
      kind: 'native' as const,
      format: 'sldprt',
      file_name: 'part.sldprt',
    };
    getModelArtifact.mockResolvedValue(native);
    render(<ModelArtifactModalHost />);
    openModelArtifactModal(native, null);
    expect(
      await screen.findByText('Native SolidWorks file. Download to open it in SolidWorks.'),
    ).toBeDefined();
    const links = screen.getAllByRole('link', { name: /Download/ });
    expect(links).toHaveLength(1);
    expect(links[0].getAttribute('aria-label')).toBe('Download part.sldprt');
  });

  it('hands off and releases the active viewer', async () => {
    function Active() {
      return <span data-testid="active">{useActiveModelViewer()}</span>;
    }
    render(
      <>
        <ModelArtifactModalHost />
        <Active />
      </>,
    );
    openModelArtifactModal(artifact, null);
    await waitFor(() =>
      expect(screen.getByTestId('active').textContent).toBe(`modal:${artifact.id}`),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await waitFor(() => expect(screen.getByTestId('active').textContent).toBe(''));
  });
});
