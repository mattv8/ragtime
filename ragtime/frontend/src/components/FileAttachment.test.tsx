import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AttachmentPreviewList, FileAttachment, type AttachmentFile } from './FileAttachment';

const attachment: AttachmentFile = {
  id: 'attachment-1',
  type: 'file',
  name: 'notes.txt',
  size: 12,
  mimeType: 'text/plain',
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('AttachmentPreviewList', () => {
  it('removes an attachment through its supplied callback and respects disabled state', () => {
    const onRemove = vi.fn();
    const { rerender } = render(
      <AttachmentPreviewList attachments={[attachment]} onRemove={onRemove} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Remove attachment' }));
    expect(onRemove).toHaveBeenCalledWith(attachment.id);

    rerender(<AttachmentPreviewList attachments={[attachment]} onRemove={onRemove} disabled />);
    expect(screen.getByRole('button', { name: 'Remove attachment' }).hasAttribute('disabled')).toBe(
      true,
    );
  });

  it('can leave previews to a separate composer row', () => {
    render(
      <FileAttachment
        attachments={[attachment]}
        onAttachmentsChange={vi.fn()}
        showPreviews={false}
      />,
    );

    expect(document.querySelector('.attachment-preview-list')).toBeNull();
    expect(screen.getByTitle('Attach files or images')).toBeDefined();
  });
});

describe('FileAttachment', () => {
  const oversized = (name: string) => {
    const file = new File(['content'], name, { type: 'text/plain' });
    Object.defineProperty(file, 'size', { value: 20 * 1024 * 1024 + 1 });
    return file;
  };

  it('shows one dialog listing failed files from the same batch without using window.alert', async () => {
    const alertSpy = vi.spyOn(window, 'alert');
    const { container } = render(
      <FileAttachment
        attachments={[]}
        onAttachmentsChange={() => undefined}
        conversationId="chat-1"
      />,
    );
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, {
      target: { files: [oversized('first.txt'), oversized('second.txt')] },
    });

    expect(
      await screen.findByRole('alertdialog', { name: 'Some files could not be attached' }),
    ).toBeTruthy();
    expect(screen.getByText('File "first.txt" is too large. Maximum size is 20MB.')).toBeTruthy();
    expect(screen.getByText('File "second.txt" is too large. Maximum size is 20MB.')).toBeTruthy();
    expect(alertSpy).not.toHaveBeenCalled();
  });

  it('keeps failures from an earlier batch when a later batch also fails', async () => {
    const { container } = render(
      <FileAttachment
        attachments={[]}
        onAttachmentsChange={() => undefined}
        conversationId="chat-1"
      />,
    );
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, { target: { files: [oversized('first.txt')] } });
    expect(await screen.findByRole('alertdialog', { name: 'File too large' })).toBeTruthy();

    fireEvent.change(fileInput, { target: { files: [oversized('second.txt')] } });

    expect(
      await screen.findByRole('alertdialog', { name: 'Some files could not be attached' }),
    ).toBeTruthy();
    expect(screen.getByText('File "first.txt" is too large. Maximum size is 20MB.')).toBeTruthy();
    expect(screen.getByText('File "second.txt" is too large. Maximum size is 20MB.')).toBeTruthy();
  });
});
