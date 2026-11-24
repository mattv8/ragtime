import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ToolGroupSection } from './ToolGroupSection';

afterEach(cleanup);

it('omits description relationship when there is no description', () => {
  render(
    <ToolGroupSection
      sectionId="a"
      title="A"
      toolCount={1}
      disabledCount={0}
      health={null}
      collapsed={false}
      onToggle={vi.fn()}
    />,
  );
  expect(screen.getByRole('button').hasAttribute('aria-describedby')).toBe(false);
});

it('hides the description while collapsed and drops its relationship', () => {
  const { container } = render(
    <ToolGroupSection
      sectionId="a"
      title="Servers"
      description="Details"
      toolCount={1}
      disabledCount={0}
      health={null}
      collapsed
      onToggle={vi.fn()}
    />,
  );
  expect(screen.queryByText('Details')).toBeNull();
  expect(
    screen.getByRole('button', { name: 'Servers 1 tool' }).hasAttribute('aria-describedby'),
  ).toBe(false);
  expect(container.querySelector('#tools-section-a-description')).toBeNull();
});

it('shows visible and total counts while a filter narrows the section', () => {
  render(
    <ToolGroupSection
      sectionId="a"
      title="Servers"
      toolCount={2}
      totalCount={4}
      disabledCount={0}
      health={null}
      collapsed={false}
      onToggle={vi.fn()}
    />,
  );
  expect(screen.getByText('2 of 4 tools')).toBeTruthy();
});

it('reserves the actions slot in sections without a menu so health columns align', () => {
  const { container } = render(
    <ToolGroupSection
      sectionId="ungrouped"
      title="Ungrouped"
      toolCount={1}
      disabledCount={0}
      health={{ tone: 'success', label: 'All healthy' }}
      collapsed={false}
      onToggle={vi.fn()}
    />,
  );
  expect(container.querySelector('.tools-section-actions-spacer')).not.toBeNull();
  expect(screen.queryByRole('button', { name: /Group actions/ })).toBeNull();
});

it('keeps collapsed content mounted and separates actions from the toggle', () => {
  render(
    <ToolGroupSection
      sectionId="a"
      title="Servers"
      toolCount={1}
      disabledCount={0}
      health={null}
      collapsed
      onToggle={vi.fn()}
      onOpenActions={vi.fn()}
    >
      <span>content</span>
    </ToolGroupSection>,
  );
  expect(screen.getByText('content').parentElement?.hidden).toBe(true);
  expect(screen.getByRole('button', { name: 'Servers 1 tool' }).getAttribute('aria-expanded')).toBe(
    'false',
  );
  expect(
    screen
      .getByRole('heading')
      .contains(screen.getByRole('button', { name: 'Group actions for Servers' })),
  ).toBe(false);
});

it('wires toggle, description, health and menu actions', () => {
  const onToggle = vi.fn();
  const onOpenActions = vi.fn();
  const { rerender } = render(
    <ToolGroupSection
      sectionId="a"
      title="Servers"
      description="Details"
      toolCount={5}
      disabledCount={1}
      health={{ tone: 'error', label: '2 failing' }}
      collapsed={false}
      onToggle={onToggle}
      onOpenActions={onOpenActions}
      actionsMenuOpen
    >
      <span>children</span>
    </ToolGroupSection>,
  );
  const toggle = screen.getByRole('button', { name: 'Servers 5 tools, 1 disabled' });
  expect(toggle.closest('h3')).not.toBeNull();
  expect(toggle.getAttribute('aria-expanded')).toBe('true');
  expect(toggle.getAttribute('aria-controls')).toBe('tools-section-a-body');
  expect(toggle.getAttribute('aria-describedby')).toBe('tools-section-a-description');
  expect(document.getElementById('tools-section-a-description')?.textContent).toBe('Details');
  expect(screen.getByText('2 failing').getAttribute('data-tone')).toBe('error');
  fireEvent.click(toggle);
  expect(onToggle).toHaveBeenCalledWith('a');
  const actions = screen.getByRole('button', { name: 'Group actions for Servers' });
  expect(actions.closest('h3')).toBeNull();
  expect(actions.getAttribute('aria-expanded')).toBe('true');
  fireEvent.click(actions);
  expect(onOpenActions).toHaveBeenCalledWith('a', actions);
  rerender(
    <ToolGroupSection
      sectionId="a"
      title="Servers"
      toolCount={4}
      disabledCount={0}
      health={null}
      collapsed={false}
      onToggle={onToggle}
    />,
  );
  expect(screen.getByText('4 tools')).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Group actions for Servers' })).toBeNull();
});

it('renders empty state and dispatches drop and context handlers', () => {
  const header = { onDragOver: vi.fn(), onDragLeave: vi.fn(), onDrop: vi.fn() };
  const body = { onDragOver: vi.fn(), onDragLeave: vi.fn(), onDrop: vi.fn() };
  const context = vi.fn();
  const props = {
    sectionId: 'a',
    title: 'Servers',
    toolCount: 0,
    disabledCount: 0,
    health: null,
    collapsed: false,
    onToggle: vi.fn(),
    emptyMessage: 'Empty',
    headerDropHandlers: header,
    bodyDropHandlers: body,
    onHeaderContextMenu: context,
  };
  const { container, rerender } = render(
    <ToolGroupSection {...props} dropTarget="header" dropLabel="Move to Servers">
      child
    </ToolGroupSection>,
  );
  const headerElement = container.querySelector('.tools-section-header')!;
  const bodyElement = container.querySelector('.tools-section-body')!;
  expect(headerElement.classList.contains('is-drop-target')).toBe(true);
  expect(screen.getByText('Move to Servers')).toBeTruthy();
  expect(screen.getByText('Empty')).toBeTruthy();
  expect(screen.queryByText('child')).toBeNull();
  fireEvent.dragOver(headerElement);
  fireEvent.dragLeave(headerElement);
  fireEvent.drop(headerElement);
  fireEvent.contextMenu(headerElement);
  fireEvent.dragOver(bodyElement);
  fireEvent.dragLeave(bodyElement);
  fireEvent.drop(bodyElement);
  for (const fn of [...Object.values(header), ...Object.values(body), context])
    expect(fn).toHaveBeenCalledOnce();
  rerender(
    <ToolGroupSection {...props} toolCount={1} dropTarget="body">
      child
    </ToolGroupSection>,
  );
  expect(bodyElement.classList.contains('is-drop-target')).toBe(true);
  expect(screen.queryByText('Empty')).toBeNull();
  expect(screen.getByText('child')).toBeTruthy();
});
