export type AnchoredMenuPlacement = 'bottom-start' | 'bottom-end' | 'top-start' | 'top-end';

export interface AnchoredMenuPosition {
  top: number;
  left: number;
  maxHeight: number;
  maxWidth: number;
}

interface Viewport {
  width: number;
  height: number;
}

export function computeAnchoredMenuPosition(
  trigger: Pick<DOMRect, 'top' | 'bottom' | 'left' | 'right'>,
  menu: { width: number; height: number },
  placement: AnchoredMenuPlacement,
  viewport: Viewport = { width: window.innerWidth, height: window.innerHeight },
  margin = 8,
): AnchoredMenuPosition {
  const maxWidth = Math.max(0, viewport.width - margin * 2);
  const width = Math.min(menu.width, maxWidth);
  const availableAbove = Math.max(0, trigger.top - margin);
  const availableBelow = Math.max(0, viewport.height - trigger.bottom - margin);
  const preferAbove = placement.startsWith('top');
  const hasRoomAbove = availableAbove >= menu.height;
  const hasRoomBelow = availableBelow >= menu.height;
  const opensAbove = preferAbove
    ? hasRoomAbove
      ? true
      : hasRoomBelow
        ? false
        : availableAbove >= availableBelow
    : hasRoomBelow
      ? false
      : hasRoomAbove
        ? true
        : availableAbove > availableBelow;
  const maxHeight = opensAbove ? availableAbove : availableBelow;
  const top = opensAbove ? trigger.top - Math.min(menu.height, maxHeight) : trigger.bottom;
  const preferEnd = placement.endsWith('end');
  const endLeft = trigger.right - width;
  const startLeft = trigger.left;
  const endFits = endLeft >= margin;
  const startFits = startLeft + width <= viewport.width - margin;
  const left = preferEnd
    ? !endFits && startFits
      ? startLeft
      : endLeft
    : !startFits && endFits
      ? endLeft
      : startLeft;
  const maxTop = Math.max(margin, viewport.height - Math.min(menu.height, maxHeight) - margin);
  const maxLeft = Math.max(margin, viewport.width - width - margin);

  return {
    top: Math.min(Math.max(top, margin), maxTop),
    left: Math.min(Math.max(left, margin), maxLeft),
    maxHeight,
    maxWidth,
  };
}
