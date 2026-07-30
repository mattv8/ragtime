import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, createEvent, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ChatMessageNavigator, type ChatMessageNavigationEntry } from './ChatMessageNavigator';

const originalScrollIntoView = window.HTMLElement.prototype.scrollIntoView;

const entries: ChatMessageNavigationEntry[] = [
  { key: 'user-1', messageIndex: 0, preview: 'First user message' },
  { key: 'user-2', messageIndex: 3, preview: 'Second user message' },
  { key: 'user-3', messageIndex: 7, preview: 'Third user message' },
];

function setScrollableMetrics(element: HTMLElement, clientHeight: number, scrollHeight: number) {
  let scrollTop = 0;
  Object.defineProperty(element, 'clientHeight', {
    configurable: true,
    value: clientHeight,
  });
  Object.defineProperty(element, 'scrollHeight', {
    configurable: true,
    value: scrollHeight,
  });
  Object.defineProperty(element, 'scrollTop', {
    configurable: true,
    get: () => scrollTop,
    set: (value: number) => {
      scrollTop = value;
    },
  });
}

describe('ChatMessageNavigator', () => {
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('returns null when there are fewer than two entries', () => {
    const { container, rerender } = render(
      <ChatMessageNavigator entries={[]} activeKey={null} onNavigate={vi.fn()} />,
    );
    expect(container.firstChild).toBeNull();

    rerender(<ChatMessageNavigator entries={[entries[0]]} activeKey={null} onNavigate={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it('preserves entry order and opens on hover and focus', async () => {
    const user = userEvent.setup();
    render(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    const popover = navigator.querySelector('.chat-message-navigator-popover');
    const tickButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-tick',
    );
    const previewButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-item',
    );

    expect(navigator.classList.contains('is-open')).toBe(false);
    expect(popover?.getAttribute('aria-hidden')).toBe('true');
    expect(tickButtons).toHaveLength(3);
    expect(previewButtons).toHaveLength(3);
    expect(Array.from(previewButtons, (button) => button.tabIndex)).toEqual([-1, -1, -1]);

    fireEvent.mouseEnter(navigator);
    expect(navigator.classList.contains('is-open')).toBe(true);

    const visiblePreviewButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-item',
    );
    expect(Array.from(visiblePreviewButtons, (button) => button.textContent)).toEqual([
      'First user message',
      'Second user message',
      'Third user message',
    ]);
    expect(
      Array.from(visiblePreviewButtons, (button) => {
        const textSpans = button.querySelectorAll<HTMLSpanElement>(
          '.chat-message-navigator-item-text',
        );

        return {
          spanCount: textSpans.length,
          text: textSpans[0]?.textContent,
        };
      }),
    ).toEqual([
      { spanCount: 1, text: 'First user message' },
      { spanCount: 1, text: 'Second user message' },
      { spanCount: 1, text: 'Third user message' },
    ]);
    expect(popover?.getAttribute('aria-hidden')).toBe('false');
    expect(Array.from(visiblePreviewButtons, (button) => button.tabIndex)).toEqual([0, 0, 0]);

    await user.tab();
    expect(document.activeElement).toBe(tickButtons[0]);
    expect(navigator.classList.contains('is-open')).toBe(true);

    await user.tab();
    await user.tab();
    await user.tab();
    expect(document.activeElement).toBe(
      navigator.querySelectorAll<HTMLButtonElement>('.chat-message-navigator-item')[0],
    );
    expect(navigator.classList.contains('is-open')).toBe(true);
  });

  it('delays pointer-leave close, cancels it on re-entry, and still closes immediately on blur and escape', async () => {
    vi.useFakeTimers();
    render(
      <>
        <ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />
        <button type="button">Outside navigator</button>
      </>,
    );

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    const tickButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-tick',
    );
    const firstTick = tickButtons[0];
    const outsideButton = screen.getByRole('button', { name: 'Outside navigator' });
    const popover = navigator.querySelector('.chat-message-navigator-popover');
    expect(popover).not.toBeNull();

    fireEvent.mouseEnter(navigator);
    expect(navigator.classList.contains('is-open')).toBe(true);

    fireEvent.mouseLeave(navigator);
    expect(navigator.classList.contains('is-open')).toBe(true);

    await act(async () => {
      vi.advanceTimersByTime(149);
    });
    expect(navigator.classList.contains('is-open')).toBe(true);

    fireEvent.mouseEnter(popover as HTMLElement);
    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(navigator.classList.contains('is-open')).toBe(true);

    fireEvent.mouseLeave(navigator);
    await act(async () => {
      vi.advanceTimersByTime(149);
    });
    expect(navigator.classList.contains('is-open')).toBe(true);

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(navigator.classList.contains('is-open')).toBe(false);

    act(() => {
      firstTick.focus();
    });
    expect(document.activeElement).toBe(firstTick);
    expect(navigator.classList.contains('is-open')).toBe(true);

    const firstPreviewButton = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-item',
    )[0];
    act(() => {
      firstPreviewButton.focus();
    });
    expect(document.activeElement).toBe(firstPreviewButton);
    expect(navigator.classList.contains('is-open')).toBe(true);

    act(() => {
      outsideButton.focus();
    });
    expect(document.activeElement).toBe(outsideButton);
    expect(navigator.classList.contains('is-open')).toBe(false);

    act(() => {
      firstPreviewButton.focus();
    });
    expect(document.activeElement).toBe(firstPreviewButton);
    expect(navigator.classList.contains('is-open')).toBe(true);

    fireEvent.keyDown(firstPreviewButton, { key: 'Escape' });
    expect(navigator.classList.contains('is-open')).toBe(false);
    expect(document.activeElement).toBe(firstTick);
  });

  it('isolates wheel scrolling and synchronizes the preview list with the tick rail from the first closed-wheel event', () => {
    render(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });

    const list = navigator.querySelector('.chat-message-navigator-list');
    const ticks = navigator.querySelector('.chat-message-navigator-ticks');
    expect(list).not.toBeNull();
    expect(ticks).not.toBeNull();

    setScrollableMetrics(list as HTMLElement, 100, 400);
    setScrollableMetrics(ticks as HTMLElement, 40, 100);

    const wheelEvent = createEvent.wheel(navigator, { deltaY: 150 });
    const dispatchResult = fireEvent(navigator, wheelEvent);

    expect(dispatchResult).toBe(false);
    expect((list as HTMLElement).scrollTop).toBe(150);
    expect((ticks as HTMLElement).scrollTop).toBe(30);
    expect(navigator.classList.contains('is-open')).toBe(false);

    (list as HTMLElement).scrollTop = 225;
    fireEvent.scroll(list as HTMLElement);
    expect((ticks as HTMLElement).scrollTop).toBe(45);
  });

  it('normalizes wheel line deltas using a 16px multiplier', () => {
    render(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    const list = navigator.querySelector('.chat-message-navigator-list');
    const ticks = navigator.querySelector('.chat-message-navigator-ticks');
    expect(list).not.toBeNull();
    expect(ticks).not.toBeNull();

    setScrollableMetrics(list as HTMLElement, 100, 400);
    setScrollableMetrics(ticks as HTMLElement, 40, 100);

    const wheelEvent = createEvent.wheel(navigator, {
      deltaY: 3,
      deltaMode: WheelEvent.DOM_DELTA_LINE,
    });
    const dispatchResult = fireEvent(navigator, wheelEvent);

    expect(dispatchResult).toBe(false);
    expect((list as HTMLElement).scrollTop).toBe(48);
    expect((ticks as HTMLElement).scrollTop).toBeCloseTo(9.6);
  });

  it('installs the wheel listener after rerendering from fewer than two entries', () => {
    const { rerender } = render(
      <ChatMessageNavigator entries={[entries[0]]} activeKey={null} onNavigate={vi.fn()} />,
    );

    rerender(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    fireEvent.mouseEnter(navigator);

    const list = navigator.querySelector('.chat-message-navigator-list');
    const ticks = navigator.querySelector('.chat-message-navigator-ticks');
    expect(list).not.toBeNull();
    expect(ticks).not.toBeNull();

    setScrollableMetrics(list as HTMLElement, 100, 400);
    setScrollableMetrics(ticks as HTMLElement, 40, 100);

    const wheelEvent = createEvent.wheel(navigator, { deltaY: 120 });
    const dispatchResult = fireEvent(navigator, wheelEvent);

    expect(dispatchResult).toBe(false);
    expect((list as HTMLElement).scrollTop).toBe(120);
    expect((ticks as HTMLElement).scrollTop).toBe(24);
  });

  it('previews the matching row from tick hover and focus, keeps aria-current on the true active entry, and notifies navigation clicks', async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    const unsafeEntries: ChatMessageNavigationEntry[] = [
      entries[0],
      {
        key: 'legacy-user-message-2:2026-07-29T12:00:00.000Z"]',
        messageIndex: 3,
        preview: 'Unsafe key',
      },
      entries[2],
    ];
    const scrollIntoView = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollIntoView;

    const { rerender } = render(
      <ChatMessageNavigator
        entries={unsafeEntries}
        activeKey={unsafeEntries[1].key}
        onNavigate={onNavigate}
      />,
    );

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    fireEvent.mouseEnter(navigator);

    const tickButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-tick',
    );
    const previewButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-item',
    );
    const secondTick = tickButtons[1];
    const secondPreviewButton = previewButtons[1];
    const thirdPreviewButton = previewButtons[2];

    expect(secondPreviewButton.getAttribute('aria-current')).toBe('location');
    expect(secondTick.getAttribute('aria-current')).toBe('location');
    expect(navigator.querySelectorAll('.chat-message-navigator-item.is-active')).toHaveLength(1);
    expect(navigator.querySelectorAll('.chat-message-navigator-tick.is-active')).toHaveLength(1);
    expect(secondPreviewButton.classList.contains('is-previewed')).toBe(false);
    expect(thirdPreviewButton.getAttribute('aria-current')).toBeNull();

    fireEvent.mouseEnter(secondTick);
    expect(secondPreviewButton.classList.contains('is-previewed')).toBe(true);
    expect(secondPreviewButton.getAttribute('aria-current')).toBe('location');

    fireEvent.mouseLeave(secondTick);
    expect(secondPreviewButton.classList.contains('is-previewed')).toBe(false);

    act(() => {
      tickButtons[2].focus();
    });
    expect(thirdPreviewButton.classList.contains('is-previewed')).toBe(true);
    expect(thirdPreviewButton.getAttribute('aria-current')).toBeNull();
    expect(secondPreviewButton.getAttribute('aria-current')).toBe('location');

    act(() => {
      tickButtons[2].blur();
    });
    expect(thirdPreviewButton.classList.contains('is-previewed')).toBe(false);

    await user.click(tickButtons[2]);
    expect(onNavigate).toHaveBeenCalledWith(unsafeEntries[2]);

    rerender(
      <ChatMessageNavigator entries={unsafeEntries} activeKey="user-3" onNavigate={onNavigate} />,
    );
    expect(scrollIntoView).toHaveBeenCalledTimes(6);
  });

  it('does not toggle closed when a tick is clicked while the navigator is hovered', async () => {
    const user = userEvent.setup();
    render(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    const tickButtons = navigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-tick',
    );

    fireEvent.mouseEnter(navigator);
    expect(navigator.classList.contains('is-open')).toBe(true);

    await user.click(tickButtons[0]);
    expect(navigator.classList.contains('is-open')).toBe(true);
  });

  it('resets to closed when entries drop below the render threshold and then reappear', () => {
    const { rerender } = render(
      <ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />,
    );

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    fireEvent.mouseEnter(navigator);
    expect(navigator.classList.contains('is-open')).toBe(true);

    rerender(<ChatMessageNavigator entries={[entries[0]]} activeKey={null} onNavigate={vi.fn()} />);
    expect(screen.queryByRole('navigation', { name: 'User message navigation' })).toBeNull();

    rerender(<ChatMessageNavigator entries={entries} activeKey={null} onNavigate={vi.fn()} />);

    const nextNavigator = screen.getByRole('navigation', { name: 'User message navigation' });
    const popover = nextNavigator.querySelector('.chat-message-navigator-popover');
    const previewButtons = nextNavigator.querySelectorAll<HTMLButtonElement>(
      '.chat-message-navigator-item',
    );

    expect(nextNavigator.classList.contains('is-open')).toBe(false);
    expect(popover?.getAttribute('aria-hidden')).toBe('true');
    expect(Array.from(previewButtons, (button) => button.tabIndex)).toEqual([-1, -1, -1]);
  });

  it('closes on escape, returns focus to the active tick, and reopens on subsequent tab navigation', async () => {
    const user = userEvent.setup();
    render(
      <>
        <ChatMessageNavigator entries={entries} activeKey={entries[1].key} onNavigate={vi.fn()} />
        <button type="button">After navigator</button>
      </>,
    );

    await user.tab();
    await user.tab();
    await user.tab();
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getAllByRole('button', { name: /jump to user message 1/i })[1],
    );

    await user.keyboard('{Escape}');

    const navigator = screen.getByRole('navigation', { name: 'User message navigation' });
    expect(navigator.classList.contains('is-open')).toBe(false);
    expect(document.activeElement).toBe(
      navigator.querySelectorAll<HTMLButtonElement>('.chat-message-navigator-tick')[1],
    );

    await user.tab();
    expect(document.activeElement).toBe(
      navigator.querySelectorAll<HTMLButtonElement>('.chat-message-navigator-tick')[2],
    );
    expect(navigator.classList.contains('is-open')).toBe(true);

    await user.tab({ shift: true });
    expect(document.activeElement).toBe(
      navigator.querySelectorAll<HTMLButtonElement>('.chat-message-navigator-tick')[1],
    );
    expect(navigator.classList.contains('is-open')).toBe(true);
  });
});
