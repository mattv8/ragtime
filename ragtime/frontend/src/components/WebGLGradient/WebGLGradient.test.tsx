import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import WebGLGradient from './WebGLGradient';

class BatteryManagerMock extends EventTarget {
  charging = false;
}

describe('WebGLGradient battery warning', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});

    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    );

    Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
      configurable: true,
      value: vi.fn().mockReturnValue(null),
    });

    Object.defineProperty(navigator, 'getBattery', {
      configurable: true,
      value: vi.fn().mockResolvedValue(new BatteryManagerMock()),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('shows the battery warning in a popover on hover and focus instead of inline text', async () => {
    render(<WebGLGradient />);

    const trigger = await screen.findByRole('button', {
      name: 'Background paused on battery',
    });
    const status = await screen.findByRole('status');

    expect(status.textContent).toContain('Background paused on battery');
    expect(screen.queryByRole('tooltip')).toBeNull();

    fireEvent.mouseEnter(trigger);

    await waitFor(() => {
      expect(screen.getByRole('tooltip').textContent).toContain('Background paused on battery');
    });

    fireEvent.mouseLeave(trigger);

    await waitFor(() => {
      expect(screen.queryByRole('tooltip')).toBeNull();
    });

    fireEvent.focus(trigger);

    await waitFor(() => {
      expect(screen.getByRole('tooltip').textContent).toContain('Background paused on battery');
    });
  });
});

describe('WebGLGradient WebGL availability', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});

    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    );

    Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
      configurable: true,
      value: vi.fn().mockReturnValue(null),
    });

    Object.defineProperty(navigator, 'getBattery', {
      configurable: true,
      value: vi.fn().mockRejectedValue(new Error('Battery API not available')),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('reports false through onWebGLAvailabilityChange when WebGL context is null', async () => {
    const onWebGLAvailabilityChange = vi.fn();
    render(<WebGLGradient onWebGLAvailabilityChange={onWebGLAvailabilityChange} />);

    await waitFor(() => {
      expect(onWebGLAvailabilityChange).toHaveBeenCalledWith(false);
    });
  });
});
