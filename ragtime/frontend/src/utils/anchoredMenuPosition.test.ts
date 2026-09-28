import { describe, expect, it } from 'vitest';

import { computeAnchoredMenuPosition } from './anchoredMenuPosition';

const viewport = { width: 500, height: 500 };

function trigger(left: number, top: number, width = 40, height = 30) {
  return { left, top, right: left + width, bottom: top + height };
}

describe('computeAnchoredMenuPosition', () => {
  it('keeps end alignment when it fits', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(300, 100),
        { width: 180, height: 160 },
        'bottom-end',
        viewport,
      ),
    ).toMatchObject({ left: 160, top: 130 });
  });

  it('flips end alignment to start near the left edge', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(12, 100),
        { width: 180, height: 160 },
        'bottom-end',
        viewport,
      ),
    ).toMatchObject({ left: 12, top: 130 });
  });

  it('flips start alignment to end near the right edge', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(452, 100),
        { width: 180, height: 160 },
        'bottom-start',
        viewport,
      ),
    ).toMatchObject({ left: 312, top: 130 });
  });

  it('clamps horizontally when neither alignment fits', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(200, 100),
        { width: 600, height: 160 },
        'bottom-start',
        viewport,
      ),
    ).toMatchObject({ left: 8, maxWidth: 484 });
  });

  it('uses the preferred vertical side when it fits', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(100, 300),
        { width: 180, height: 160 },
        'top-start',
        viewport,
      ),
    ).toMatchObject({ top: 140, maxHeight: 292 });
  });

  it('flips vertically when the preferred side lacks room', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(100, 10),
        { width: 180, height: 160 },
        'top-start',
        viewport,
      ),
    ).toMatchObject({ top: 40, maxHeight: 452 });
  });

  it('uses the larger vertical side when neither side fits', () => {
    expect(
      computeAnchoredMenuPosition(
        trigger(100, 180),
        { width: 180, height: 300 },
        'bottom-start',
        viewport,
      ),
    ).toMatchObject({ top: 210, maxHeight: 282 });
  });
});
