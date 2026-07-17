import { useEffect } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';

import {
  SearchFilterBar,
  normalizeSearchFilterText,
  searchFilterTextMatchesQuery,
  useUrlSearchFilterState,
} from './SearchFilterBar';

/**
 * Queries reaching searchFilterTextMatchesQuery are already normalized by the
 * filter state hook, so mirror that here.
 */
function toQueries(...rawQueries: string[]): string[] {
  return rawQueries.map(normalizeSearchFilterText).filter(Boolean);
}

afterEach(() => {
  cleanup();
});

function SearchFilterBarHarness(props: { completionCandidates?: string[] }): JSX.Element {
  const state = useUrlSearchFilterState('test-search');
  useEffect(() => {
    window.history.replaceState(null, '', '/settings');
  }, []);

  return (
    <>
      <SearchFilterBar
        state={state}
        placeholder="Filter settings by keyword..."
        ariaLabel="Filter settings by keyword"
        completionCandidates={props.completionCandidates}
      />
      <button type="button">Next field</button>
    </>
  );
}

describe('normalizeSearchFilterText ampersand handling', () => {
  it('canonicalizes "&" and "and" to the same token', () => {
    expect(normalizeSearchFilterText('Backup & Restore')).toBe(
      normalizeSearchFilterText('Backup and Restore'),
    );
  });

  it('treats an unspaced ampersand the same as a spaced "and"', () => {
    expect(normalizeSearchFilterText('Backup&Restore')).toBe(
      normalizeSearchFilterText('Backup and Restore'),
    );
  });
});

describe('searchFilterTextMatchesQuery ampersand interchangeability', () => {
  it('matches an "&" label when the user typed "and"', () => {
    expect(
      searchFilterTextMatchesQuery('Server Backup & Restore', toQueries('backup and restore')),
    ).toBe(true);
  });

  it('matches an "&" label when the user typed "&"', () => {
    expect(
      searchFilterTextMatchesQuery('Server Backup & Restore', toQueries('backup & restore')),
    ).toBe(true);
  });

  it('matches an "and" label when the user typed "&"', () => {
    expect(searchFilterTextMatchesQuery('Backup and Restore', toQueries('backup & restore'))).toBe(
      true,
    );
  });

  it('still rejects unrelated queries', () => {
    expect(searchFilterTextMatchesQuery('Server Backup & Restore', toQueries('appearance'))).toBe(
      false,
    );
  });
});

describe('SearchFilterBar completion behavior', () => {
  it('keeps legacy Tab-to-tag behavior when no completion candidates are provided', async () => {
    const user = userEvent.setup();
    render(<SearchFilterBarHarness />);

    const input = screen.getByRole('textbox', { name: 'Filter settings by keyword' });
    await user.type(input, 'appearance');
    await user.tab();

    expect(screen.getByText('appearance')).toBeTruthy();
    expect(input).toHaveProperty('value', '');
    expect(screen.queryByText(/Tab to complete/i)).toBeNull();
  });

  it('shows a Tab completion hint and accepts the suggestion without committing a tag', async () => {
    const user = userEvent.setup();
    render(
      <SearchFilterBarHarness
        completionCandidates={['Server Backup & Restore', 'Search Configuration']}
      />,
    );

    const input = screen.getByRole('textbox', { name: 'Filter settings by keyword' });
    await user.type(input, 'server backup and');

    expect(screen.getByText(/Tab to complete:/i).textContent).toContain('Server Backup & Restore');

    await user.tab();

    expect(input).toHaveProperty('value', 'Server Backup & Restore');
    expect(document.activeElement).toBe(input);
    expect(screen.queryByText(/^Server Backup & Restore$/)).toBeNull();
  });

  it('preserves normal focus navigation when there is no completion match', async () => {
    const user = userEvent.setup();
    render(<SearchFilterBarHarness completionCandidates={['Server Backup & Restore']} />);

    const input = screen.getByRole('textbox', { name: 'Filter settings by keyword' });
    await user.type(input, 'appearance');
    await user.tab();

    expect(document.activeElement).not.toBe(input);
    expect(input).toHaveProperty('value', '');
    expect(screen.getByText(/^appearance$/)).toBeTruthy();
  });
});
