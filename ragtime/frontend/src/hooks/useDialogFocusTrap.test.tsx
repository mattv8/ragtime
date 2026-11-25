import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { useRef, type RefObject } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { getFocusableElements, useDialogFocusTrap } from './useDialogFocusTrap';

function TestDialog({ returnFocusRef }: { returnFocusRef?: RefObject<HTMLElement | null> }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const initialFocusRef = useRef<HTMLButtonElement>(null);
  useDialogFocusTrap(dialogRef, initialFocusRef, { returnFocusRef });
  return (
    <div ref={dialogRef}>
      <button ref={initialFocusRef}>Initial</button>
    </div>
  );
}

afterEach(cleanup);

describe('getFocusableElements', () => {
  it('excludes inactive tabs and controls hidden from the accessibility tree', () => {
    const container = document.createElement('div');
    container.innerHTML = `
      <button id="active">Active</button>
      <button id="inactive-tab" tabindex="-1">Inactive tab</button>
      <div hidden><button id="hidden">Hidden</button></div>
      <div inert><button id="inert">Inert</button></div>
      <div aria-hidden="true"><button id="aria-hidden">Aria hidden</button></div>
    `;

    expect(getFocusableElements(container).map((element) => element.id)).toEqual(['active']);
  });
});

describe('useDialogFocusTrap', () => {
  it('focuses the initial element and restores the explicit return target', async () => {
    const target = document.createElement('button');
    document.body.append(target);
    const view = render(<TestDialog returnFocusRef={{ current: target }} />);
    await waitFor(() => expect(document.activeElement).toBe(screen.getByText('Initial')));
    view.unmount();
    expect(document.activeElement).toBe(target);
    target.remove();
  });

  it('falls back to the previously focused element when the return target is disconnected', () => {
    const previous = document.createElement('button');
    document.body.append(previous);
    previous.focus();
    const view = render(
      <TestDialog returnFocusRef={{ current: document.createElement('button') }} />,
    );
    view.unmount();
    expect(document.activeElement).toBe(previous);
    previous.remove();
  });
});
