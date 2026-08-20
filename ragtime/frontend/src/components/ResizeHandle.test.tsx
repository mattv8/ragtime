import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import { ResizeHandle } from './ResizeHandle';

const pointerCaptures = new WeakMap<Element, Set<number>>();

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', {
    configurable: true,
    value(pointerId: number) {
      const captures = pointerCaptures.get(this) ?? new Set<number>();
      captures.add(pointerId);
      pointerCaptures.set(this, captures);
    },
  });

  Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', {
    configurable: true,
    value(pointerId: number) {
      pointerCaptures.get(this)?.delete(pointerId);
    },
  });

  Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', {
    configurable: true,
    value(pointerId: number) {
      return pointerCaptures.get(this)?.has(pointerId) ?? false;
    },
  });
});

afterEach(() => {
  cleanup();
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
});

describe('ResizeHandle', () => {
  it('exposes separator semantics and keyboard resizing for pixel values', () => {
    const onResize = vi.fn();
    const onResizeTo = vi.fn();
    const onResizeEnd = vi.fn();

    render(
      <ResizeHandle
        direction="horizontal"
        ariaLabel="Resize chat sidebar"
        value={280}
        min={180}
        max={480}
        valueUnit="pixels"
        onResize={onResize}
        onResizeTo={onResizeTo}
        onResizeEnd={onResizeEnd}
      />,
    );

    const separator = screen.getByRole('separator', { name: 'Resize chat sidebar' });

    expect(separator.getAttribute('aria-orientation')).toBe('vertical');
    expect(separator.getAttribute('aria-valuemin')).toBe('180');
    expect(separator.getAttribute('aria-valuemax')).toBe('480');
    expect(separator.getAttribute('aria-valuenow')).toBe('280');
    expect(separator.getAttribute('aria-valuetext')).toBe('280 pixels');
    expect(separator.getAttribute('tabindex')).toBe('0');

    fireEvent.keyDown(separator, { key: 'ArrowRight' });
    fireEvent.keyDown(separator, { key: 'ArrowLeft', shiftKey: true });
    fireEvent.keyDown(separator, { key: 'Home' });
    fireEvent.keyDown(separator, { key: 'End' });

    expect(onResize).toHaveBeenNthCalledWith(1, 8);
    expect(onResize).toHaveBeenNthCalledWith(2, -32);
    expect(onResizeTo).toHaveBeenNthCalledWith(1, 180);
    expect(onResizeTo).toHaveBeenNthCalledWith(2, 480);
    expect(onResizeEnd).toHaveBeenCalledTimes(4);
  });

  it('toggles collapsed state with Enter and exposes collapsed aria text', () => {
    const onResize = vi.fn();
    const onResizeTo = vi.fn();
    const onResizeEnd = vi.fn();

    const { rerender } = render(
      <ResizeHandle
        direction="vertical"
        ariaLabel="Resize workspace editor and chat"
        value={60}
        min={10}
        max={90}
        valueUnit="percent"
        collapsible={{ side: 'after', restoreValue: 60 }}
        onResize={onResize}
        onResizeTo={onResizeTo}
        onResizeEnd={onResizeEnd}
      />,
    );

    const separator = screen.getByRole('separator', { name: 'Resize workspace editor and chat' });

    expect(separator.getAttribute('aria-valuetext')).toBe('60%');

    fireEvent.keyDown(separator, { key: 'Enter' });

    expect(onResize).not.toHaveBeenCalled();
    expect(onResizeTo).toHaveBeenNthCalledWith(1, 0);
    expect(onResizeEnd).toHaveBeenCalledTimes(1);

    rerender(
      <ResizeHandle
        direction="vertical"
        ariaLabel="Resize workspace editor and chat"
        value={0}
        min={10}
        max={90}
        valueUnit="percent"
        collapsed="after"
        collapsible={{ side: 'after', restoreValue: 60 }}
        onResize={onResize}
        onResizeTo={onResizeTo}
        onResizeEnd={onResizeEnd}
      />,
    );

    const collapsedSeparator = screen.getByRole('separator', {
      name: 'Resize workspace editor and chat',
    });

    expect(collapsedSeparator.getAttribute('aria-valuenow')).toBe('0');
    expect(collapsedSeparator.getAttribute('aria-valuetext')).toBe('Collapsed');

    fireEvent.keyDown(collapsedSeparator, { key: 'Enter' });

    expect(onResizeTo).toHaveBeenNthCalledWith(2, 60);
    expect(onResizeEnd).toHaveBeenCalledTimes(2);
  });

  it('flushes pointer resizing and restores body styles when the drag completes', () => {
    const onResize = vi.fn();
    const onResizeTo = vi.fn();
    const onResizeEnd = vi.fn();

    render(
      <ResizeHandle
        direction="horizontal"
        ariaLabel="Resize public chat input"
        value={120}
        min={96}
        max={240}
        valueUnit="pixels"
        onResize={onResize}
        onResizeTo={onResizeTo}
        onResizeEnd={onResizeEnd}
      />,
    );

    const separator = screen.getByRole('separator', { name: 'Resize public chat input' });

    fireEvent.pointerDown(separator, { pointerId: 1, clientX: 100 });

    expect(document.body.style.cursor).toBe('col-resize');
    expect(document.body.style.userSelect).toBe('none');

    fireEvent.pointerMove(separator, { pointerId: 1, clientX: 124 });
    fireEvent.pointerUp(separator, { pointerId: 1, clientX: 124 });

    expect(onResize).toHaveBeenCalledTimes(1);
    expect(onResizeTo).not.toHaveBeenCalled();
    expect(onResizeEnd).toHaveBeenCalledTimes(1);
    expect(document.body.style.cursor).toBe('');
    expect(document.body.style.userSelect).toBe('');
  });

  it('cleans up body drag styles when unmounted mid-drag', () => {
    const onResize = vi.fn();
    const onResizeTo = vi.fn();

    const { unmount } = render(
      <ResizeHandle
        direction="vertical"
        ariaLabel="Resize chat input"
        value={120}
        min={96}
        max={240}
        valueUnit="pixels"
        onResize={onResize}
        onResizeTo={onResizeTo}
      />,
    );

    const separator = screen.getByRole('separator', { name: 'Resize chat input' });
    fireEvent.pointerDown(separator, { pointerId: 2, clientY: 100 });

    expect(document.body.style.cursor).toBe('row-resize');
    expect(document.body.style.userSelect).toBe('none');

    unmount();

    expect(document.body.style.cursor).toBe('');
    expect(document.body.style.userSelect).toBe('');
  });
});
