import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, expect, it, vi } from 'vitest';

import { MasterDetailLayout } from './MasterDetailLayout';

const items = [{ id: 'finance', name: 'Finance' }];

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function Detail() {
  return (
    <>
      <h4 id="detail-title" tabIndex={-1}>
        Finance details
      </h4>
      <button type="button">Delete level</button>
    </>
  );
}

function LayoutHarness({ removeOnClose = false }: { removeOnClose?: boolean }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [visibleItems, setVisibleItems] = useState(items);

  return (
    <MasterDetailLayout
      id="access-levels"
      heading="Access levels"
      headerAction={
        <button type="button" data-new-access-level>
          Add level
        </button>
      }
      items={visibleItems}
      selectedId={selectedId}
      onSelect={setSelectedId}
      onCloseDetail={() => {
        if (removeOnClose) setVisibleItems([]);
        setSelectedId(null);
      }}
      detail={selectedId ? <Detail /> : undefined}
      detailLabelledBy="detail-title"
      rowHookAttribute="data-access-level-id"
    />
  );
}

it('closes an open detail with Escape and returns focus to its row', async () => {
  const user = userEvent.setup();
  render(<LayoutHarness />);

  const row = screen.getByRole('button', { name: 'Finance' });
  await user.click(row);
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Finance details' })),
  );

  await user.keyboard('{Escape}');

  await waitFor(() => expect(document.activeElement).toBe(row));
});

it('does not steal focus when a persisted detail is restored after a tab remount', async () => {
  const outside = document.createElement('button');
  document.body.append(outside);
  outside.focus();

  render(
    <MasterDetailLayout
      id="access-levels"
      heading="Access levels"
      items={items}
      selectedId="finance"
      onSelect={vi.fn()}
      onCloseDetail={vi.fn()}
      detail={<Detail />}
      detailLabelledBy="detail-title"
      rowHookAttribute="data-access-level-id"
    />,
  );

  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  expect(document.activeElement).toBe(outside);
  outside.remove();
});

it('does not overwrite focus moved after detail focus was scheduled', () => {
  const pendingFrames: FrameRequestCallback[] = [];
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
    pendingFrames.push(callback);
    return pendingFrames.length;
  });
  render(<LayoutHarness />);

  const row = screen.getByRole('button', { name: 'Finance' });
  row.focus();
  fireEvent.click(row);
  expect(pendingFrames).toHaveLength(1);

  const deleteButton = screen.getByRole('button', { name: 'Delete level' });
  deleteButton.focus();
  act(() => pendingFrames.shift()?.(performance.now()));

  expect(document.activeElement).toBe(deleteButton);
});

it('returns focus to Add when the selected item no longer exists after close', async () => {
  const user = userEvent.setup();
  render(<LayoutHarness removeOnClose />);

  await user.click(screen.getByRole('button', { name: 'Finance' }));
  await user.keyboard('{Escape}');

  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Add level' })),
  );
});

it('leaves prevented Escape to an active editor and ignores it while busy', async () => {
  const user = userEvent.setup();
  const onCloseDetail = vi.fn();
  const { rerender } = render(
    <MasterDetailLayout
      id="access-levels"
      heading="Access levels"
      items={items}
      selectedId="finance"
      onSelect={vi.fn()}
      onCloseDetail={onCloseDetail}
      detail={<Detail />}
      detailLabelledBy="detail-title"
      rowHookAttribute="data-access-level-id"
      rootProps={{ onKeyDown: (event) => event.preventDefault() }}
    />,
  );

  const detail = screen.getByRole('heading', { name: 'Finance details' });
  detail.focus();
  await user.keyboard('{Escape}');
  expect(onCloseDetail).not.toHaveBeenCalled();

  rerender(
    <MasterDetailLayout
      id="access-levels"
      heading="Access levels"
      items={items}
      selectedId="finance"
      onSelect={vi.fn()}
      onCloseDetail={onCloseDetail}
      detail={<Detail />}
      detailLabelledBy="detail-title"
      rowHookAttribute="data-access-level-id"
      busy
    />,
  );
  detail.focus();
  await user.keyboard('{Escape}');
  expect(onCloseDetail).not.toHaveBeenCalled();
});

it('prevents Escape before delegating detail close so an ancestor does not close the modal', async () => {
  const user = userEvent.setup();
  const onCloseDetail = vi.fn();
  const onModalClose = vi.fn();
  const onDocumentKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && !event.defaultPrevented) onModalClose();
  };
  document.addEventListener('keydown', onDocumentKeyDown);

  render(
    <MasterDetailLayout
      id="access-levels"
      heading="Access levels"
      items={items}
      selectedId="finance"
      onSelect={vi.fn()}
      onCloseDetail={onCloseDetail}
      detail={<Detail />}
      detailLabelledBy="detail-title"
      rowHookAttribute="data-access-level-id"
    />,
  );

  screen.getByRole('heading', { name: 'Finance details' }).focus();
  await user.keyboard('{Escape}');
  document.removeEventListener('keydown', onDocumentKeyDown);

  expect(onCloseDetail).toHaveBeenCalledOnce();
  expect(onModalClose).not.toHaveBeenCalled();
});
