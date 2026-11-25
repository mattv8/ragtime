import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';

afterEach(cleanup);

function setup(overrides: Partial<React.ComponentProps<typeof ConstrainedPathBrowser>> = {}) {
  const onSelectPath = vi.fn();
  const onBrowsePath = vi.fn(async (path: string) => ({
    path,
    entries:
      path === '/'
        ? [
            { name: 'Allowed', path: '/Allowed', is_dir: true },
            { name: 'Blocked', path: '/Blocked', is_dir: true },
          ]
        : [],
  }));
  const props = {
    currentPath: '',
    rootPath: '/',
    defaultExpanded: false,
    onSelectPath,
    onBrowsePath,
    ...overrides,
  };
  const view = render(<ConstrainedPathBrowser {...props} />);
  return { ...view, onSelectPath, onBrowsePath, props };
}

async function expand() {
  fireEvent.click(screen.getByRole('button', { expanded: false }));
  await screen.findByRole('button', { name: /Allowed/ });
}

describe('ConstrainedPathBrowser', () => {
  it('loads on expansion and selects and collapses by default', async () => {
    const { onBrowsePath, onSelectPath } = setup();
    expect(onBrowsePath).not.toHaveBeenCalled();
    const header = screen.getByRole('button', { expanded: false });
    expect(header.querySelector('.mount-icon')?.getAttribute('aria-hidden')).toBe('true');
    await expand();
    expect(header.getAttribute('aria-expanded')).toBe('true');
    expect(screen.getByRole('textbox', { name: 'Filter or Create...' })).toBeTruthy();
    expect(onBrowsePath).toHaveBeenCalledWith('/');
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    expect(onSelectPath).toHaveBeenCalledWith('/');
    expect(screen.queryByRole('button', { name: 'Select' })).toBeNull();
    expect(header.getAttribute('aria-expanded')).toBe('false');
  });

  it('allows disabled entries to navigate by default', async () => {
    const { onBrowsePath } = setup({
      isPathDisabled: (path) => (path === '/Blocked' ? 'No access' : null),
    });
    await expand();
    const entry = screen.getByRole('button', { name: /Blocked/ });
    expect(entry.getAttribute('aria-disabled')).toBeNull();
    fireEvent.click(entry);
    await waitFor(() => expect(onBrowsePath).toHaveBeenCalledWith('/Blocked'));
  });

  it('keeps a string selection reason in the title only', async () => {
    setup({ canSelectPath: () => false, cannotSelectPathMessage: 'Not allowed' });
    await expand();
    expect(screen.getByRole('button', { name: 'Select' }).getAttribute('title')).toBe(
      'Not allowed',
    );
    expect(screen.queryByText('Not allowed')).toBeNull();
  });

  it('keeps the browser expanded after selecting when requested', async () => {
    const { onSelectPath } = setup({ keepExpandedOnSelect: true });
    await expand();
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    expect(onSelectPath).toHaveBeenCalledWith('/');
    expect(screen.getByRole('button', { name: 'Select' })).toBeTruthy();
  });

  it('uses the custom select label', async () => {
    setup({ selectLabel: 'Approve folder' });
    await expand();
    expect(screen.getByRole('button', { name: 'Approve folder' })).toBeTruthy();
  });

  it('uses the custom filter placeholder as its accessible label', async () => {
    setup({ filterPlaceholder: 'Filter folders' });
    await expand();
    expect(
      screen.getByRole('textbox', { name: 'Filter folders' }).getAttribute('placeholder'),
    ).toBe('Filter folders');
  });

  it('shows a computed selection reason only for an unselectable path', async () => {
    setup({
      canSelectPath: (path) => path !== '/',
      cannotSelectPathMessage: (path) => `Cannot select ${path}`,
    });
    await expand();
    expect(screen.getByRole('button', { name: 'Select' }).getAttribute('title')).toBe(
      'Cannot select /',
    );
    expect(
      screen.getByText('Cannot select /').matches('p.field-help[data-browser-select-reason]'),
    ).toBe(true);
    const reason = screen.getByText('Cannot select /');
    expect(reason.id).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Select' }).getAttribute('aria-describedby')).toBe(
      reason.id,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Allowed' }));
    await waitFor(() => expect(screen.queryByText('Cannot select /Allowed')).toBeNull());
    expect(
      screen.getByRole('button', { name: 'Select' }).getAttribute('aria-describedby'),
    ).toBeNull();
  });

  it('shows status badges only for enabled, non-pending entries', async () => {
    const pathBadge = vi.fn(() => ({
      label: 'Ready',
      tone: 'success' as const,
      title: 'Can change',
    }));
    setup({
      stagedDirectories: ['/Pending'],
      isPathDisabled: (path) => (path === '/Blocked' ? 'No access' : null),
      pathBadge,
    });
    await expand();
    const badge = within(screen.getByRole('button', { name: /Allowed/ })).getByText('Ready');
    expect(badge.getAttribute('data-tone')).toBe('success');
    expect(badge.getAttribute('title')).toBe('Can change');
    expect(within(screen.getByRole('button', { name: /Blocked/ })).queryByText('Ready')).toBeNull();
    expect(within(screen.getByRole('button', { name: /Pending/ })).getByText('New')).toBeTruthy();
    expect(pathBadge).not.toHaveBeenCalledWith('/Blocked');
    expect(pathBadge).not.toHaveBeenCalledWith('/Pending');
  });

  it('blocks navigation to disabled entries, including via the filter', async () => {
    const { onBrowsePath } = setup({
      blockDisabledNavigation: true,
      isPathDisabled: (path) => (path === '/Blocked' ? 'No access' : null),
    });
    await expand();
    const entry = screen.getByRole('button', { name: /Blocked/ });
    expect(entry.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(entry);
    const filter = screen.getByPlaceholderText('Filter or Create...');
    fireEvent.change(filter, { target: { value: 'Blocked' } });
    fireEvent.keyDown(filter, { key: 'Enter' });
    expect(onBrowsePath).not.toHaveBeenCalledWith('/Blocked');
    expect(screen.getByRole('button', { name: /Blocked/ })).toBeTruthy();
    fireEvent.change(filter, {
      target: { value: 'Blocked/' },
    });
    expect(onBrowsePath).not.toHaveBeenCalledWith('/Blocked');
    expect(screen.getByRole('button', { name: /Blocked/ })).toBeTruthy();
  });

  it('skips disabled matches on Enter only when disabled navigation is blocked', async () => {
    const isPathDisabled = (path: string) => (path === '/Allowed' ? 'No access' : null);
    const first = setup({ blockDisabledNavigation: true, isPathDisabled });
    await expand();
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    await waitFor(() => expect(first.onBrowsePath).toHaveBeenCalledWith('/Blocked'));
    first.unmount();

    const second = setup({ isPathDisabled });
    await expand();
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    await waitFor(() => expect(second.onBrowsePath).toHaveBeenCalledWith('/Allowed'));
  });

  it('reports normalized browse paths initially, on navigation and going up', async () => {
    const onBrowsePathChange = vi.fn();
    setup({ onBrowsePathChange });
    expect(onBrowsePathChange).toHaveBeenCalledWith('/');
    await expand();
    fireEvent.click(screen.getByRole('button', { name: 'Allowed' }));
    await waitFor(() => expect(onBrowsePathChange).toHaveBeenCalledWith('/Allowed'));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '..' }).hasAttribute('disabled')).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: '..' }));
    await waitFor(() =>
      expect(onBrowsePathChange.mock.calls.map(([path]) => path)).toEqual(['/', '/Allowed', '/']),
    );
  });
});
