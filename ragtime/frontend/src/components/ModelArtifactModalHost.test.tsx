import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ModelArtifactModalHost } from './ModelArtifactModalHost';
import { closeModelArtifactModal, openModelArtifactModal } from './modelArtifactModalStore';

vi.mock('@/api/client', () => ({ api: { getModelArtifact: () => new Promise(() => {}) } }));

afterEach(() => {
  closeModelArtifactModal();
  cleanup();
});

it('portals the modal into the document body', async () => {
  const { container } = render(<ModelArtifactModalHost />);
  openModelArtifactModal(
    {
      id: 'a'.repeat(32),
      kind: 'preview',
      format: 'png',
      file_name: 'part.png',
      size_bytes: 1,
      expires_at: '2099-01-01T00:00:00Z',
      content_url: '/image',
      page_url: '/page',
    },
    null,
  );
  const dialog = await screen.findByRole('dialog');
  expect(container.contains(dialog)).toBe(false);
  expect(dialog.parentElement?.parentElement).toBe(document.body);
  closeModelArtifactModal();
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
});
