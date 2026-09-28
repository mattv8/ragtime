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

afterEach(cleanup);

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
