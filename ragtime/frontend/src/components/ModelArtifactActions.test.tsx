import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { api } from '@/api';
import { ArtifactRequestError, type ModelArtifact } from '@/utils/modelArtifact';
import { clearArtifactAccessCache } from '@/hooks/useArtifactAccess';
import { messageModelArtifacts, ModelArtifactActions } from './ModelArtifactActions';
import { openModelArtifactModal } from './modelArtifactModalStore';

vi.mock('./modelArtifactModalStore', () => ({ openModelArtifactModal: vi.fn() }));

function artifact(id: string): ModelArtifact {
  return {
    id,
    kind: 'model',
    format: 'step',
    file_name: `${id}.step`,
    size_bytes: 1024,
    expires_at: '2099-01-01T00:00:00Z',
    content_url: `/indexes/solidworks/artifacts/${id}/content`,
    page_url: `/?artifact=${id}`,
  };
}

afterEach(() => {
  cleanup();
  clearArtifactAccessCache();
  vi.restoreAllMocks();
});

it('deduplicates visible artifact events and omits hidden tool calls', () => {
  const file = artifact('a'.repeat(32));
  const output = JSON.stringify({
    ok: true,
    operation: 'model.export',
    result: { artifact: file },
  });
  const events = [
    { type: 'tool' as const, tool: 'solidworks_export', output },
    { type: 'tool' as const, tool: 'solidworks_export', output },
    {
      type: 'tool' as const,
      tool: 'solidworks_export',
      output: JSON.stringify({
        ok: true,
        operation: 'model.export',
        result: { artifact: artifact('b'.repeat(32)) },
      }),
      presentation: { visibility: 'hidden' },
    },
  ];
  expect(messageModelArtifacts(events)).toEqual([file]);
  expect(messageModelArtifacts(events)).toBe(messageModelArtifacts(events));
  const second = artifact('c'.repeat(32));
  events.push({
    type: 'tool',
    tool: 'solidworks_export',
    output: JSON.stringify({ ok: true, operation: 'model.export', result: { artifact: second } }),
  });
  expect(messageModelArtifacts(events)).toEqual([file, second]);
});

it.each([1, 2])('shows %i inline file groups and opens with the opener', (count) => {
  const files = Array.from({ length: count }, (_, index) => artifact(`file${index}`));
  vi.spyOn(api, 'getModelArtifact').mockResolvedValue({
    ...files[0],
    expired: false,
    tool_config_name: null,
  });
  const { container } = render(<ModelArtifactActions artifacts={files} menuId="files" />);
  expect(container.querySelectorAll('.chat-message-artifact-group')).toHaveLength(count);
  const button = screen.getByRole('button', { name: 'View 3D model file0.step' });
  fireEvent.click(button);
  expect(openModelArtifactModal).toHaveBeenCalledWith(files[0], button);
  expect(
    screen.getByRole('link', { name: 'Download file0.step' }).getAttribute('download'),
  ).not.toBeNull();
});

it('opens the overflow above the bar and closes on Escape', () => {
  const files = ['a', 'b', 'c'].map(artifact);
  vi.spyOn(api, 'getModelArtifact').mockResolvedValue({
    ...files[0],
    expired: false,
    tool_config_name: null,
  });
  render(<ModelArtifactActions artifacts={files} menuId="files" />);
  const trigger = screen.getByRole('button', { name: 'Model files, 3' });
  fireEvent.click(trigger);
  expect(trigger.getAttribute('aria-expanded')).toBe('true');
  expect(
    document.getElementById('files')?.querySelectorAll('.chat-message-artifact-group'),
  ).toHaveLength(3);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(document.activeElement).toBe(trigger);
  expect(trigger.getAttribute('aria-expanded')).toBe('false');
  fireEvent.click(trigger);
  fireEvent.pointerDown(document.body);
  expect(trigger.getAttribute('aria-expanded')).toBe('false');
  expect(document.activeElement).toBe(trigger);
});

it('disables download with a sign in reason', async () => {
  vi.spyOn(api, 'getModelArtifact').mockRejectedValue(new ArtifactRequestError(401));
  render(<ModelArtifactActions artifacts={[artifact('a')]} menuId="files" />);
  await waitFor(() =>
    expect(screen.getByLabelText('Download a.step').getAttribute('title')).toBe(
      'Sign in to download',
    ),
  );
  expect(screen.getByLabelText('Download a.step').getAttribute('aria-disabled')).toBe('true');
  expect(screen.getByRole('button', { name: 'Download a.step' }).tagName).toBe('BUTTON');
});

it('closes the overflow and uses its trigger as modal opener', () => {
  const files = ['a', 'b', 'c'].map(artifact);
  vi.spyOn(api, 'getModelArtifact').mockResolvedValue({
    ...files[0],
    expired: false,
    tool_config_name: null,
  });
  render(<ModelArtifactActions artifacts={files} menuId="files" />);
  const trigger = screen.getByRole('button', { name: 'Model files, 3' });
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole('button', { name: 'View 3D model a.step' }));
  expect(openModelArtifactModal).toHaveBeenCalledWith(files[0], trigger);
  expect(trigger.getAttribute('aria-expanded')).toBe('false');
  expect(document.getElementById('files')).toBeNull();
});

it('keeps file actions separate from the hover actions', () => {
  const file = artifact('visible');
  vi.spyOn(api, 'getModelArtifact').mockResolvedValue({
    ...file,
    expired: false,
    tool_config_name: null,
  });
  const { container } = render(
    <div className="chat-message-actions chat-message-actions-left">
      <ModelArtifactActions artifacts={[file]} menuId="files" />
      <span className="chat-message-hover-actions">Copy</span>
    </div>,
  );
  const bar = container.querySelector('.chat-message-actions');
  expect(bar?.firstElementChild?.className).toBe('chat-message-artifact-actions');
  expect(bar?.querySelector('.chat-message-hover-actions')).not.toBeNull();
  expect(bar?.classList.contains('visible')).toBe(false);
});
