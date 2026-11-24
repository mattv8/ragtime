import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ContextMenu, type ContextMenuItem, type ContextMenuPlacement } from './ContextMenu';

afterEach(() => {
  cleanup();
});

function makeItems(
  overrides: Partial<Record<'a' | 'b' | 'c' | 'd', Partial<ContextMenuItem>>> = {},
) {
  const onSelect = {
    a: vi.fn(),
    b: vi.fn(),
    c: vi.fn(),
    d: vi.fn(),
  };
  const items: ContextMenuItem[] = [
    { label: 'Alpha', onSelect: onSelect.a, ...overrides.a },
    { label: 'Bravo', disabled: true, onSelect: onSelect.b, ...overrides.b },
    { label: 'Charlie', onSelect: onSelect.c, ...overrides.c },
    { label: 'Delta', onSelect: onSelect.d, ...overrides.d },
  ];
  return { items, onSelect };
}

describe('measured placement', () => {
  const VIEWPORT_WIDTH = 1000;
  const VIEWPORT_HEIGHT = 800;
  const MENU_SIZE = { width: 272, height: 140 };
  let restore: Array<() => void> = [];

  beforeEach(() => {
    const originalWidth = window.innerWidth;
    const originalHeight = window.innerHeight;
    window.innerWidth = VIEWPORT_WIDTH;
    window.innerHeight = VIEWPORT_HEIGHT;
    const rectSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () =>
        ({
          top: 0,
          left: 0,
          right: MENU_SIZE.width,
          bottom: MENU_SIZE.height,
          ...MENU_SIZE,
        }) as DOMRect,
    );
    restore = [
      () => rectSpy.mockRestore(),
      () => {
        window.innerWidth = originalWidth;
        window.innerHeight = originalHeight;
      },
    ];
  });

  afterEach(() => {
    restore.forEach((fn) => fn());
  });

  function renderPlaced(placement: ContextMenuPlacement) {
    const { items } = makeItems();
    render(<ContextMenu {...placement} items={items} />);
    return document.querySelector('.userspace-tool-context-menu') as HTMLElement;
  }

  it.each([
    ['pointer with room', { x: 100, y: 100 }, '108px', '108px'],
    ['pointer near the right edge', { x: 990, y: 100 }, '710px', '108px'],
    ['pointer near the bottom', { x: 100, y: 780 }, '108px', '632px'],
  ])('places a menu at a %s', (_label, placement, left, top) => {
    const menu = renderPlaced(placement);
    expect(menu.style.left).toBe(left);
    expect(menu.style.top).toBe(top);
  });

  it.each([
    [
      'anchor near the right edge aligns right edges',
      { left: 960, right: 984, top: 100, bottom: 124 },
      '712px',
      '124px',
    ],
    [
      'anchor with room opens below and aligns left edges',
      { left: 100, right: 124, top: 100, bottom: 124 },
      '100px',
      '124px',
    ],
    [
      'anchor near the bottom opens above',
      { left: 100, right: 124, top: 700, bottom: 724 },
      '100px',
      '560px',
    ],
  ])('places a menu by anchor rect: %s', (_label, anchorRect, left, top) => {
    const menu = renderPlaced({ anchorRect });
    expect(menu.style.left).toBe(left);
    expect(menu.style.top).toBe(top);
  });

  it.each([
    [
      'anchor near the bottom opens above with the space above it',
      { anchorRect: { left: 100, right: 124, top: 700, bottom: 724 } },
      '692px',
    ],
    ['pointer with room caps height at the space below', { x: 100, y: 100 }, '684px'],
  ])('caps the menu height to the free space: %s', (_label, placement, maxHeight) => {
    const menu = renderPlaced(placement as ContextMenuPlacement);
    expect(menu.style.maxHeight).toBe(maxHeight);
  });

  it('restores the placed position after measuring when the menu re-renders with the same size', () => {
    const { items } = makeItems();
    const anchorRect = { left: 100, right: 124, top: 100, bottom: 124 };
    const { rerender } = render(<ContextMenu items={items.slice(0, 3)} anchorRect={anchorRect} />);
    const menu = document.querySelector('.userspace-tool-context-menu') as HTMLElement;
    const maxHeightBefore = menu.style.maxHeight;

    // Same stubbed size, so the computed placement is identical and React will not rewrite the style.
    rerender(<ContextMenu items={items} anchorRect={anchorRect} />);

    expect(menu.style.left).toBe('100px');
    expect(menu.style.top).toBe('124px');
    expect(menu.style.maxHeight).toBe(maxHeightBefore);
  });

  it('keeps the anchor position when the menu already fits (regression)', () => {
    const menu = renderPlaced({ x: 20, y: 100 });
    expect(menu.style.top).toBe('108px');
  });
});

function itemElement(label: string): HTMLElement {
  return screen.getByText(label).closest('[role="button"]') as HTMLElement;
}

function focusedLabel(): string | null {
  const active = document.activeElement as HTMLElement | null;
  return active?.querySelector('.userspace-tool-context-menu-label')?.textContent ?? null;
}

function isFocusedInnerInput(): boolean {
  return document.activeElement instanceof HTMLInputElement;
}

describe('ContextMenu keyboard behaviour', () => {
  it('focuses the first enabled item on mount when autoFocus is set', () => {
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    expect(focusedLabel()).toBe('Alpha');
  });

  it('does not steal focus on mount without autoFocus', () => {
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} />);

    expect(document.activeElement).not.toBe(itemElement('Alpha'));
  });

  it('moves focus with ArrowDown and ArrowUp, skipping disabled items and wrapping', async () => {
    const user = userEvent.setup();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    await user.keyboard('{ArrowDown}');
    expect(focusedLabel()).toBe('Charlie');

    await user.keyboard('{ArrowDown}');
    expect(focusedLabel()).toBe('Delta');

    await user.keyboard('{ArrowDown}');
    expect(focusedLabel()).toBe('Alpha');

    await user.keyboard('{ArrowUp}');
    expect(focusedLabel()).toBe('Delta');

    await user.keyboard('{ArrowUp}');
    expect(focusedLabel()).toBe('Charlie');
  });

  it('moves focus to the first and last enabled items with Home and End', async () => {
    const user = userEvent.setup();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    await user.keyboard('{End}');
    expect(focusedLabel()).toBe('Delta');

    await user.keyboard('{Home}');
    expect(focusedLabel()).toBe('Alpha');
  });

  it('calls onClose on Escape', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus onClose={onClose} />);

    await user.keyboard('{Escape}');

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Tab leaves from the last enabled item', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus onClose={onClose} />);

    await user.keyboard('{End}');
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard('{Tab}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Shift+Tab leaves from the first enabled item', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus onClose={onClose} />);

    expect(focusedLabel()).toBe('Alpha');

    await user.keyboard('{Shift>}{Tab}{/Shift}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('moves focus from one enabled item to the next on Tab, skipping disabled items', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus onClose={onClose} />);

    await user.keyboard('{Tab}');

    expect(focusedLabel()).toBe('Charlie');
    expect(onClose).not.toHaveBeenCalled();
  });
});

describe('ContextMenu with toggle and checkbox items', () => {
  function makeControlItems() {
    const onSelect = {
      check: vi.fn(),
      toggle: vi.fn(),
      plain: vi.fn(),
    };
    const items: ContextMenuItem[] = [
      { label: 'Plain', onSelect: onSelect.plain },
      { label: 'Check one', type: 'checkbox', checked: false, onSelect: onSelect.check },
      { label: 'Toggle one', type: 'toggle', checked: false, onSelect: onSelect.toggle },
    ];
    return { items, onSelect };
  }

  it('never lands Tab focus on an inner checkbox input', async () => {
    const user = userEvent.setup();
    const { items } = makeControlItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    expect(focusedLabel()).toBe('Plain');

    await user.keyboard('{Tab}');
    expect(isFocusedInnerInput()).toBe(false);
    expect(focusedLabel()).toBe('Check one');

    await user.keyboard('{Tab}');
    expect(isFocusedInnerInput()).toBe(false);
    expect(focusedLabel()).toBe('Toggle one');
  });

  it('moves focus with ArrowUp and ArrowDown from a toggle item', async () => {
    const user = userEvent.setup();
    const { items } = makeControlItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    await user.keyboard('{End}');
    expect(focusedLabel()).toBe('Toggle one');

    await user.keyboard('{ArrowUp}');
    expect(focusedLabel()).toBe('Check one');

    await user.keyboard('{ArrowDown}');
    expect(focusedLabel()).toBe('Toggle one');

    await user.keyboard('{ArrowDown}');
    expect(focusedLabel()).toBe('Plain');
  });

  it('calls onClose when Tab leaves from the last toggle item without focusing its input', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { items } = makeControlItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus onClose={onClose} />);

    await user.keyboard('{End}');
    await user.keyboard('{Tab}');

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(isFocusedInnerInput()).toBe(false);
  });

  it('calls onSelect once when the checkbox input is clicked', async () => {
    const user = userEvent.setup();
    const { items, onSelect } = makeControlItems();
    const { container } = render(<ContextMenu items={items} x={0} y={0} />);

    const checkbox = container.ownerDocument.querySelector<HTMLInputElement>(
      '.userspace-tool-context-menu input[type="checkbox"]',
    );
    expect(checkbox).not.toBeNull();

    await user.click(checkbox as HTMLInputElement);

    expect(onSelect.check).toHaveBeenCalledTimes(1);
    expect(onSelect.plain).not.toHaveBeenCalled();
  });
});

describe('ContextMenu selection', () => {
  it('calls onSelect on click, but not on a disabled item', async () => {
    const user = userEvent.setup();
    const { items, onSelect } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} />);

    await user.click(itemElement('Alpha'));
    await user.click(itemElement('Bravo'));

    expect(onSelect.a).toHaveBeenCalledTimes(1);
    expect(onSelect.b).not.toHaveBeenCalled();
  });

  it('calls onSelect on Enter and Space for enabled items only', async () => {
    const user = userEvent.setup();
    const { items, onSelect } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} autoFocus />);

    await user.keyboard('{Enter}');
    expect(onSelect.a).toHaveBeenCalledTimes(1);

    await user.keyboard('{ArrowDown}');
    await user.keyboard(' ');
    expect(onSelect.c).toHaveBeenCalledTimes(1);

    itemElement('Bravo').focus();
    await user.keyboard('{Enter}');
    await user.keyboard(' ');
    expect(onSelect.b).not.toHaveBeenCalled();
  });
});

describe('ContextMenu accessibility', () => {
  it('exposes ariaLabel as the label of a group container', () => {
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} ariaLabel="Tool actions for Demo" />);

    const group = screen.getByRole('group', { name: 'Tool actions for Demo' });
    expect(group.classList).toContain('userspace-tool-context-menu');
  });

  it('keeps items as role button with roving tab stops for disabled items', () => {
    const { items } = makeItems();
    render(<ContextMenu items={items} x={0} y={0} />);

    expect(itemElement('Alpha').getAttribute('tabindex')).toBe('0');
    expect(itemElement('Bravo').getAttribute('tabindex')).toBe('-1');
  });
});
