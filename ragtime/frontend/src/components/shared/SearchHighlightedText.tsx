import { type ReactNode } from 'react';

export interface SearchHighlightedTextProps {
  text: string;
  query: string;
}

/**
 * Renders text with search query terms highlighted.
 * Performs case-insensitive matching and wraps matches in <mark> elements.
 */
export function SearchHighlightedText({ text, query }: SearchHighlightedTextProps) {
  const needle = query.trim();
  if (!needle) return <>{text}</>;

  const lowerText = text.toLowerCase();
  const lowerNeedle = needle.toLowerCase();
  const segments: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = lowerText.indexOf(lowerNeedle);

  while (matchIndex !== -1) {
    if (matchIndex > cursor) {
      segments.push(text.slice(cursor, matchIndex));
    }
    segments.push(
      <mark key={`${matchIndex}-${segments.length}`} className="chat-search-highlight">
        {text.slice(matchIndex, matchIndex + needle.length)}
      </mark>,
    );
    cursor = matchIndex + needle.length;
    matchIndex = lowerText.indexOf(lowerNeedle, cursor);
  }

  if (cursor < text.length) {
    segments.push(text.slice(cursor));
  }
  return <>{segments}</>;
}
