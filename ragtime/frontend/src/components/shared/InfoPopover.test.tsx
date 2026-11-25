import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { InfoPopover } from './InfoPopover';

afterEach(cleanup);

const TRIGGER_ID = 'info-popover-test';
const trigger = () => screen.getByRole('button', { name: 'About test help' });
const content = () => document.getElementById(`${TRIGGER_ID}-content`);

function renderPopover() {
  return render(
    <>
      <InfoPopover id={TRIGGER_ID} label="About test help">
        Help text for the test
      </InfoPopover>
      <button type="button">Outside</button>
    </>,
  );
}

describe('InfoPopover', () => {
  it('is closed by default', () => {
    renderPopover();

    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(trigger().hasAttribute('aria-controls')).toBe(false);
    expect(content()).toBeNull();
  });

  it('opens on click and closes on a second click', () => {
    renderPopover();

    fireEvent.click(trigger());

    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(trigger().getAttribute('aria-controls')).toBe(`${TRIGGER_ID}-content`);
    expect(trigger().getAttribute('aria-describedby')).toBe(`${TRIGGER_ID}-content`);
    expect(content()?.textContent).toBe('Help text for the test');

    fireEvent.click(trigger());

    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(content()).toBeNull();
  });

  it('closes on Escape and returns focus to the trigger', () => {
    renderPopover();
    fireEvent.click(trigger());

    act(() => screen.getByRole('button', { name: 'Outside' }).focus());
    fireEvent.keyDown(document, { key: 'Escape' });

    expect(content()).toBeNull();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(trigger());
  });

  it('closes on mousedown outside the trigger and content', () => {
    renderPopover();
    fireEvent.click(trigger());

    fireEvent.mouseDown(screen.getByRole('button', { name: 'Outside' }));

    expect(content()).toBeNull();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
  });

  it('stays open on mousedown inside the content', () => {
    renderPopover();
    fireEvent.click(trigger());

    fireEvent.mouseDown(content() as HTMLElement);

    expect(content()).not.toBeNull();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
  });
});
