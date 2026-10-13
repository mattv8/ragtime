import {
  cleanup,
  createEvent,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect, type ReactNode } from 'react';
import { afterEach, expect, it, vi } from 'vitest';

import { CheckboxDropdown } from './CheckboxDropdown';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const options = [
  { id: 'engineering', label: 'Engineering', badge: 'LDAP' },
  { id: 'finance', label: 'Finance', badge: 'Internal' },
  { id: 'operations', label: 'Operations' },
];

function ModalHarness({
  children,
  onModalKeyDown,
  onDocumentKeyDown,
}: {
  children: ReactNode;
  onModalKeyDown: () => void;
  onDocumentKeyDown: () => void;
}) {
  useEffect(() => {
    document.addEventListener('keydown', onDocumentKeyDown);
    return () => document.removeEventListener('keydown', onDocumentKeyDown);
  }, [onDocumentKeyDown]);

  return (
    <div role="dialog" onKeyDown={onModalKeyDown}>
      <button type="button">Close modal</button>
      {children}
      <input aria-label="Next wizard input" />
    </div>
  );
}

function renderDropdown(props: Partial<React.ComponentProps<typeof CheckboxDropdown>> = {}) {
  return render(
    <CheckboxDropdown
      options={options}
      selectedIds={[]}
      onChange={vi.fn()}
      ariaLabel="Groups for this level"
      searchThreshold={0}
      {...props}
    />,
  );
}

function mockNaturalPanelHeight(height: number) {
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(function (
    this: HTMLElement,
  ) {
    return this.classList.contains('chk-dropdown-panel') ? 40 : 0;
  });
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (
    this: HTMLElement,
  ) {
    return this.classList.contains('chk-dropdown-list') ? 10 : 0;
  });
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function (
    this: HTMLElement,
  ) {
    return this.classList.contains('chk-dropdown-list') ? height - 30 : 0;
  });
}

it('filters option labels and badges with an always-visible search', async () => {
  const user = userEvent.setup();
  renderDropdown({ searchAriaLabel: 'Filter directory groups' });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const search = screen.getByRole('textbox', { name: 'Filter directory groups' });
  await user.type(search, 'ldap');

  expect(screen.getByLabelText('Engineering')).toBeTruthy();
  expect(screen.queryByLabelText('Finance')).toBeNull();
});

it('keeps unknown selections when an option is selected', async () => {
  const user = userEvent.setup();
  const onChange = vi.fn();
  renderDropdown({ selectedIds: ['retired-group'], onChange });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  await user.click(screen.getByLabelText('Engineering'));

  expect(onChange).toHaveBeenCalledWith(['retired-group', 'engineering']);
});

it('closes on Escape without reaching a document modal handler and restores trigger focus', async () => {
  const user = userEvent.setup();
  const documentEscape = vi.fn();
  document.addEventListener('keydown', documentEscape);
  renderDropdown();

  const trigger = screen.getByRole('button', { name: 'Groups for this level' });
  await user.click(trigger);
  await user.keyboard('{Escape}');

  expect(screen.queryByRole('dialog', { name: 'Groups for this level' })).toBeNull();
  expect(documentEscape).not.toHaveBeenCalled();
  await waitFor(() => expect(document.activeElement).toBe(trigger));
  document.removeEventListener('keydown', documentEscape);
});

it('stops middle Tab directions from React and document modal traps without preventing default', async () => {
  const user = userEvent.setup();
  const modalTrap = vi.fn();
  const wizardListener = vi.fn();
  render(
    <ModalHarness onModalKeyDown={modalTrap} onDocumentKeyDown={wizardListener}>
      <CheckboxDropdown options={options} selectedIds={[]} onChange={vi.fn()} ariaLabel="Groups" />
    </ModalHarness>,
  );

  await user.click(screen.getByRole('button', { name: 'Groups' }));
  const panel = screen.getByRole('dialog', { name: 'Groups' });
  const checks = within(panel).getAllByRole('checkbox');
  checks[1].focus();
  const forwardTab = createEvent.keyDown(checks[1], {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  fireEvent(checks[1], forwardTab);
  checks[1].focus();
  const reverseTab = createEvent.keyDown(checks[1], {
    key: 'Tab',
    shiftKey: true,
    bubbles: true,
    cancelable: true,
  });
  fireEvent(checks[1], reverseTab);

  expect(forwardTab.defaultPrevented).toBe(false);
  expect(reverseTab.defaultPrevented).toBe(false);
  expect(modalTrap).not.toHaveBeenCalled();
  expect(wizardListener).not.toHaveBeenCalled();
});

it('closes at a Tab boundary, restores trigger focus, and prevents the boundary event', async () => {
  const user = userEvent.setup();
  renderDropdown();

  const trigger = screen.getByRole('button', { name: 'Groups for this level' });
  await user.click(trigger);
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });
  const checks = within(panel).getAllByRole('checkbox');
  checks[checks.length - 1]?.focus();
  const tabEvent = createEvent.keyDown(checks[checks.length - 1], {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  fireEvent(checks[checks.length - 1], tabEvent);

  expect(tabEvent.defaultPrevented).toBe(true);
  expect(screen.queryByRole('dialog', { name: 'Groups for this level' })).toBeNull();
  await waitFor(() => expect(document.activeElement).toBe(trigger));
});

it('flips above when the capped natural panel height does not fit below', async () => {
  const user = userEvent.setup();
  const originalInnerHeight = window.innerHeight;
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 700 });
  mockNaturalPanelHeight(320);
  const { container } = renderDropdown();
  const wrap = container.querySelector('.chk-dropdown-wrap') as HTMLDivElement;
  vi.spyOn(wrap, 'getBoundingClientRect').mockReturnValue({
    bottom: 460,
    height: 40,
    left: 30,
    right: 230,
    top: 420,
    width: 200,
    x: 30,
    y: 420,
    toJSON: () => ({}),
  });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });

  expect(panel.parentElement).toBe(document.body);
  expect(panel.style.bottom).toBe('284px');
  expect(panel.style.left).toBe('30px');
  expect(panel.style.zIndex).toBe('var(--z-tooltip)');
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: originalInnerHeight });
});

it('keeps a short panel below when it fits beneath the trigger', async () => {
  const user = userEvent.setup();
  const originalInnerHeight = window.innerHeight;
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 700 });
  mockNaturalPanelHeight(120);
  const { container } = renderDropdown();
  const wrap = container.querySelector('.chk-dropdown-wrap') as HTMLDivElement;
  vi.spyOn(wrap, 'getBoundingClientRect').mockReturnValue({
    bottom: 460,
    height: 40,
    left: 30,
    right: 230,
    top: 420,
    width: 200,
    x: 30,
    y: 420,
    toJSON: () => ({}),
  });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });

  await waitFor(() => expect(panel.style.top).toBe('464px'));
  expect(panel.style.bottom).toBe('');
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: originalInnerHeight });
});

it('keeps a capped panel below when there is sufficient lower space', async () => {
  const user = userEvent.setup();
  const originalInnerHeight = window.innerHeight;
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 700 });
  mockNaturalPanelHeight(320);
  const { container } = renderDropdown();
  const wrap = container.querySelector('.chk-dropdown-wrap') as HTMLDivElement;
  vi.spyOn(wrap, 'getBoundingClientRect').mockReturnValue({
    bottom: 360,
    height: 40,
    left: 30,
    right: 230,
    top: 320,
    width: 200,
    x: 30,
    y: 320,
    toJSON: () => ({}),
  });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });

  await waitFor(() => expect(panel.style.top).toBe('364px'));
  expect(panel.style.bottom).toBe('');
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: originalInnerHeight });
});

it('clamps short and narrow viewports to available panel dimensions', async () => {
  const user = userEvent.setup();
  const originalInnerHeight = window.innerHeight;
  const originalInnerWidth = window.innerWidth;
  Object.defineProperties(window, {
    innerHeight: { configurable: true, value: 100 },
    innerWidth: { configurable: true, value: 180 },
  });
  const { container } = renderDropdown();
  const wrap = container.querySelector('.chk-dropdown-wrap') as HTMLDivElement;
  vi.spyOn(wrap, 'getBoundingClientRect').mockReturnValue({
    bottom: 92,
    height: 20,
    left: 160,
    right: 360,
    top: 72,
    width: 200,
    x: 160,
    y: 72,
    toJSON: () => ({}),
  });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });

  expect(panel.style.maxHeight).toBe('60px');
  expect(panel.style.width).toBe('164px');
  expect(panel.style.left).toBe('8px');
  Object.defineProperties(window, {
    innerHeight: { configurable: true, value: originalInnerHeight },
    innerWidth: { configurable: true, value: originalInnerWidth },
  });
});

it('does not steal focus when an outside wizard input is clicked', async () => {
  const user = userEvent.setup();
  render(
    <ModalHarness onModalKeyDown={vi.fn()} onDocumentKeyDown={vi.fn()}>
      <CheckboxDropdown options={options} selectedIds={[]} onChange={vi.fn()} ariaLabel="Groups" />
    </ModalHarness>,
  );

  await user.click(screen.getByRole('button', { name: 'Groups' }));
  const nextInput = screen.getByRole('textbox', { name: 'Next wizard input' });
  await user.click(nextInput);

  expect(screen.queryByRole('dialog', { name: 'Groups' })).toBeNull();
  expect(document.activeElement).toBe(nextInput);
});

it('focuses a no-search popup so Escape remains scoped to the dropdown', async () => {
  const user = userEvent.setup();
  const documentEscape = vi.fn();
  document.addEventListener('keydown', documentEscape);
  renderDropdown({ searchThreshold: 5 });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const panel = screen.getByRole('dialog', { name: 'Groups for this level' });
  await waitFor(() =>
    expect(document.activeElement).toBe(within(panel).getAllByRole('checkbox')[0]),
  );
  await user.keyboard('{Escape}');

  expect(documentEscape).not.toHaveBeenCalled();
  document.removeEventListener('keydown', documentEscape);
});

it('uses triggerLabel in preference to selected option labels', () => {
  renderDropdown({ selectedIds: ['engineering'], triggerLabel: '1 group selected' });

  expect(screen.getByRole('button', { name: 'Groups for this level' }).textContent).toContain(
    '1 group selected',
  );
});

it('describes same-name provider options without changing their exact labels', async () => {
  const user = userEvent.setup();
  renderDropdown({
    options: [
      { id: 'ldap-finance', label: 'Finance', badge: 'LDAP' },
      { id: 'local-finance', label: 'Finance', badge: 'Internal' },
    ],
  });

  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  const financeOptions = screen.getAllByRole('checkbox', { name: 'Finance' });

  expect(financeOptions[0].getAttribute('aria-describedby')).toContain('ldap-finance');
  expect(financeOptions[1].getAttribute('aria-describedby')).toContain('local-finance');
});

it('moves focus to the parent dialog when an open control becomes disabled', async () => {
  const user = userEvent.setup();
  const { rerender } = render(
    <ModalHarness onModalKeyDown={vi.fn()} onDocumentKeyDown={vi.fn()}>
      <CheckboxDropdown options={options} selectedIds={[]} onChange={vi.fn()} ariaLabel="Groups" />
    </ModalHarness>,
  );

  await user.click(screen.getByRole('button', { name: 'Groups' }));
  rerender(
    <ModalHarness onModalKeyDown={vi.fn()} onDocumentKeyDown={vi.fn()}>
      <CheckboxDropdown
        options={options}
        selectedIds={[]}
        onChange={vi.fn()}
        ariaLabel="Groups"
        disabled
      />
    </ModalHarness>,
  );

  expect(screen.queryByRole('dialog', { name: 'Groups' })).toBeNull();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close modal' })),
  );
});
