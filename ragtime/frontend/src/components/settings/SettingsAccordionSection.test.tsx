import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SettingsAccordionSection } from './SettingsAccordionSection';

afterEach(() => {
  cleanup();
});

const sectionId = 'appearance';
const bodyId = `settings-accordion-body-${sectionId}`;

function getBody(): HTMLElement {
  return screen.getByText('Body content').parentElement as HTMLElement;
}

describe('SettingsAccordionSection', () => {
  it('renders the title and children', () => {
    render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open onToggle={() => {}}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    expect(screen.getByRole('button', { name: /Appearance/ })).toBeTruthy();
    expect(screen.getByText('Body content')).toBeTruthy();
  });

  it('applies open/closed DOM attributes and classes', () => {
    const { rerender } = render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open={false} onToggle={() => {}}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    const body = getBody();
    const section = body.parentElement;
    expect(section?.getAttribute('data-settings-accordion-section')).toBe(sectionId);
    expect(section?.getAttribute('data-settings-accordion-open')).toBe('false');
    expect(body.tagName).toBe('DIV');
    expect(body.id).toBe(bodyId);
    expect(body.className).toBe('settings-accordion-body');
    expect(body.hasAttribute('hidden')).toBe(true);

    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(button.getAttribute('aria-controls')).toBe(bodyId);

    rerender(
      <SettingsAccordionSection id={sectionId} title="Appearance" open onToggle={() => {}}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    expect(body.hasAttribute('hidden')).toBe(false);
    expect(section?.getAttribute('data-settings-accordion-open')).toBe('true');
    expect(button.getAttribute('aria-expanded')).toBe('true');
  });

  it('keeps children mounted when closed', () => {
    render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open={false} onToggle={() => {}}>
        <p>Kept mounted</p>
      </SettingsAccordionSection>,
    );

    expect(screen.getByText('Kept mounted')).toBeTruthy();
  });

  it('renders status when provided', () => {
    render(
      <SettingsAccordionSection
        id={sectionId}
        title="Appearance"
        open
        onToggle={() => {}}
        status={<span data-testid="status">Active</span>}
      >
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    expect(screen.getByTestId('status')).toBeTruthy();
  });

  it('does not render status container when status is absent', () => {
    render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open onToggle={() => {}}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    expect(document.querySelector('.settings-accordion-header-status')).toBeNull();
  });

  it('calls onToggle with the section id when the header is clicked', async () => {
    const onToggle = vi.fn();
    render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open={false} onToggle={onToggle}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    await userEvent.click(screen.getByRole('button'));
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(onToggle).toHaveBeenCalledWith(sectionId);
  });

  it('supports keyboard activation via the real button', async () => {
    const onToggle = vi.fn();
    render(
      <SettingsAccordionSection id={sectionId} title="Appearance" open={false} onToggle={onToggle}>
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    await userEvent.tab();
    await userEvent.keyboard('{Enter}');
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it('passes extra className to the root', () => {
    render(
      <SettingsAccordionSection
        id={sectionId}
        title="Appearance"
        open
        onToggle={() => {}}
        className="extra-class"
      >
        <p>Body content</p>
      </SettingsAccordionSection>,
    );

    const section = getBody().parentElement;
    expect(section?.classList.contains('settings-accordion-item')).toBe(true);
    expect(section?.classList.contains('settings-accordion-item--open')).toBe(true);
    expect(section?.classList.contains('extra-class')).toBe(true);
  });

  it('updates controlled open state correctly through a wrapper', async () => {
    function Wrapper() {
      const [open, setOpen] = useState(false);
      return (
        <SettingsAccordionSection
          id={sectionId}
          title="Appearance"
          open={open}
          onToggle={() => setOpen((v) => !v)}
        >
          <p>Body content</p>
        </SettingsAccordionSection>
      );
    }

    render(<Wrapper />);
    const button = screen.getByRole('button');
    const body = getBody();

    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(body.hasAttribute('hidden')).toBe(true);

    await userEvent.click(button);

    expect(button.getAttribute('aria-expanded')).toBe('true');
    expect(body.hasAttribute('hidden')).toBe(false);
  });

  it('visually flattens direct fieldset children to avoid nested bordered panels', () => {
    const css = readFileSync(join(cwd(), 'src/styles/components.css'), 'utf8');
    const fieldsetRule = css.match(/\.settings-accordion-body\s*>\s*fieldset\s*\{(?<body>[^}]*)\}/);

    expect(fieldsetRule?.groups?.body).toContain('border: 0;');
    expect(fieldsetRule?.groups?.body).toContain('padding: 0;');
  });
});
