import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ModelSelector } from './ModelSelector';

const models = [
  { id: 'openai::gpt-5', name: 'GPT-5', provider: 'openai' },
  { id: 'anthropic::claude-sonnet-4', name: 'Claude Sonnet 4', provider: 'anthropic' },
];

function rect({
  left,
  top,
  width,
  height,
}: {
  left: number;
  top: number;
  width: number;
  height: number;
}) {
  return {
    x: left,
    y: top,
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    toJSON: () => ({}),
  } as DOMRect;
}

function setViewport(width: number, height: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: height });
}

function renderSelector(placement?: 'bottom-start' | 'top-end') {
  return render(
    <ModelSelector
      models={models}
      selectedModelId="openai::gpt-5"
      onModelChange={vi.fn()}
      placement={placement}
    />,
  );
}

describe('ModelSelector placement', () => {
  const rectSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect');

  afterEach(() => {
    cleanup();
    rectSpy.mockReset();
  });

  it('opens below and start-aligned by default', () => {
    setViewport(500, 500);
    rectSpy.mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('model-selector-dropdown')
        ? rect({ left: 0, top: 0, width: 180, height: 160 })
        : rect({ left: 20, top: 20, width: 120, height: 28 });
    });
    renderSelector();

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));

    const dropdown = document.querySelector('.model-selector-dropdown') as HTMLElement;
    expect(dropdown.style.left).toBe('20px');
    expect(dropdown.style.top).toBe('48px');
  });

  it('opens above and end-aligned in top-end mode', () => {
    setViewport(500, 500);
    rectSpy.mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('model-selector-dropdown')
        ? rect({ left: 0, top: 0, width: 180, height: 160 })
        : rect({ left: 360, top: 400, width: 120, height: 28 });
    });
    renderSelector('top-end');

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));

    const dropdown = document.querySelector('.model-selector-dropdown') as HTMLElement;
    expect(dropdown.style.left).toBe('300px');
    expect(dropdown.style.top).toBe('240px');
  });

  it('flips top-end to below and clamps it inside a narrow viewport', () => {
    setViewport(200, 250);
    rectSpy.mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('model-selector-dropdown')
        ? rect({ left: 0, top: 0, width: 180, height: 160 })
        : rect({ left: 150, top: 10, width: 42, height: 28 });
    });
    renderSelector('top-end');

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));

    const dropdown = document.querySelector('.model-selector-dropdown') as HTMLElement;
    expect(dropdown.style.left).toBe('12px');
    expect(dropdown.style.top).toBe('38px');
    expect(dropdown.style.maxHeight).toBe('204px');
  });

  it('repositions after search changes the root menu size and focuses search', async () => {
    setViewport(500, 300);
    rectSpy.mockImplementation(function (this: HTMLElement) {
      if (this.classList.contains('model-selector-dropdown')) {
        const hasSearch = !!document.querySelector<HTMLInputElement>('.model-selector-search-input')
          ?.value;
        return rect({ left: 0, top: 0, width: 180, height: hasSearch ? 80 : 220 });
      }
      return rect({ left: 100, top: 100, width: 120, height: 28 });
    });
    renderSelector('top-end');

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));
    const search = screen.getByLabelText('Filter models');
    await waitFor(() => expect(document.activeElement).toBe(search));
    expect((document.querySelector('.model-selector-dropdown') as HTMLElement).style.top).toBe(
      '128px',
    );

    fireEvent.change(search, { target: { value: 'gpt' } });
    expect((document.querySelector('.model-selector-dropdown') as HTMLElement).style.top).toBe(
      '20px',
    );
  });

  it('retains Escape and outside-click dismissal', () => {
    setViewport(500, 500);
    rectSpy.mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('model-selector-dropdown')
        ? rect({ left: 0, top: 0, width: 180, height: 160 })
        : rect({ left: 20, top: 20, width: 120, height: 28 });
    });
    renderSelector();

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));
    fireEvent.keyDown(screen.getByLabelText('Filter models'), { key: 'Escape' });
    expect(document.querySelector('.model-selector-dropdown')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /openai gpt-5/i }));
    fireEvent.mouseDown(document.body);
    expect(document.querySelector('.model-selector-dropdown')).toBeNull();
  });
});

describe('ModelSelector generic provider labels', () => {
  afterEach(cleanup);

  it('uses the configured host label instead of an Other group for sparse generic metadata', () => {
    render(
      <ModelSelector
        models={[
          {
            id: 'CaseSensitive/Model',
            name: 'CaseSensitive/Model',
            provider: 'openai_compatible',
            host_provider_label: 'Internal Gateway',
          },
        ]}
        selectedModelId="CaseSensitive/Model"
        onModelChange={vi.fn()}
      />,
    );

    expect(
      screen.getByRole('button', { name: /Internal Gateway CaseSensitive\/Model/i }),
    ).toBeTruthy();
  });
});
