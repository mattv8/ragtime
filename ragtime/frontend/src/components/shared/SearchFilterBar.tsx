import { useEffect, useMemo, useState } from 'react';
import type { Dispatch, RefObject, SetStateAction } from 'react';
import { Search, X } from 'lucide-react';

export interface SearchFilterState {
  input: string;
  tags: string[];
  debouncedInput: string;
  queries: string[];
  hasActiveFilters: boolean;
  setInput: (value: string) => void;
  setTags: Dispatch<SetStateAction<string[]>>;
  clear: () => void;
}

interface SearchFilterBarProps {
  state: SearchFilterState;
  inputRef?: RefObject<HTMLInputElement>;
  placeholder: string;
  ariaLabel: string;
  className?: string;
  onClick?: () => void;
}

export function normalizeSearchFilterText(value: string): string {
  return value.trim().toLowerCase();
}

export function searchFilterTextMatchesQuery(text: string | null | undefined, queries: string[]): boolean {
  if (queries.length === 0) {
    return true;
  }

  const normalized = normalizeSearchFilterText(text || '');
  return queries.some((query) => normalized.includes(query));
}

function readSearchFilterStateFromUrl(queryParam: string): { input: string; tags: string[] } {
  const params = new URLSearchParams(window.location.search);
  const rawSearch = params.get(queryParam) || '';
  const tags = rawSearch
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  return { input: '', tags };
}

function writeSearchFilterStateToUrl(queryParam: string, input: string, tags: string[]): void {
  const params = new URLSearchParams(window.location.search);
  const searchValue = [...tags, input]
    .map((value) => value.trim())
    .filter(Boolean)
    .join(',');

  if (searchValue) {
    params.set(queryParam, searchValue);
  } else {
    params.delete(queryParam);
  }

  const nextSearch = params.toString();
  const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
  window.history.replaceState(null, '', nextUrl);
}

export function useUrlSearchFilterState(queryParam = 'search'): SearchFilterState {
  const initialState = useMemo(() => readSearchFilterStateFromUrl(queryParam), [queryParam]);
  const [tags, setTags] = useState<string[]>(initialState.tags);
  const [input, setInput] = useState(initialState.input);
  const [debouncedInput, setDebouncedInput] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedInput(input), 200);
    return () => clearTimeout(timer);
  }, [input]);

  useEffect(() => {
    writeSearchFilterStateToUrl(queryParam, input, tags);
  }, [queryParam, input, tags]);

  const queries = useMemo(() => {
    const liveInput = normalizeSearchFilterText(debouncedInput);
    return [...tags.map(normalizeSearchFilterText), ...(liveInput ? [liveInput] : [])].filter(Boolean);
  }, [tags, debouncedInput]);

  return {
    input,
    tags,
    debouncedInput,
    queries,
    hasActiveFilters: tags.length > 0 || input.trim().length > 0 || debouncedInput.trim().length > 0,
    setInput,
    setTags,
    clear: () => {
      setTags([]);
      setInput('');
    },
  };
}

export function SearchFilterBar({ state, inputRef, placeholder, ariaLabel, className = '', onClick }: SearchFilterBarProps) {
  const addTag = (rawTag: string) => {
    const tag = rawTag.trim();
    if (tag && !state.tags.includes(tag)) {
      state.setTags((prev) => [...prev, tag]);
    }
  };

  return (
    <div
      className={`settings-filter-search${className ? ` ${className}` : ''}`}
      role="search"
      aria-label={ariaLabel}
      onClick={() => {
        onClick?.();
        inputRef?.current?.focus();
      }}
    >
      <Search size={16} className="settings-filter-search-icon" aria-hidden="true" />
      {state.tags.map((tag, index) => (
        <span key={`${tag}-${index}`} className="settings-filter-tag">
          {tag}
          <button
            type="button"
            className="settings-filter-tag-remove"
            onClick={(event) => {
              event.stopPropagation();
              state.setTags((prev) => prev.filter((_, tagIndex) => tagIndex !== index));
            }}
            aria-label={`Remove filter: ${tag}`}
          >
            <X size={12} />
          </button>
        </span>
      ))}
      <input
        ref={inputRef}
        type="text"
        placeholder={state.tags.length === 0 ? placeholder : ''}
        value={state.input}
        onChange={(event) => {
          const value = event.target.value;
          if (value.endsWith(',')) {
            addTag(value.slice(0, -1));
            state.setInput('');
          } else {
            state.setInput(value);
          }
        }}
        onKeyDown={(event) => {
          if ((event.key === 'Tab' || event.key === 'Enter') && state.input.trim()) {
            event.preventDefault();
            addTag(state.input);
            state.setInput('');
          }
          if (event.key === 'Backspace' && !state.input && state.tags.length > 0) {
            state.setTags((prev) => prev.slice(0, -1));
          }
        }}
        onBlur={() => {
          addTag(state.input);
          state.setInput('');
        }}
        aria-label={ariaLabel}
      />
      {(state.tags.length > 0 || state.input.trim()) && (
        <button
          type="button"
          className="settings-filter-clear"
          onClick={(event) => {
            event.stopPropagation();
            state.clear();
          }}
          aria-label="Clear all filters"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
}
