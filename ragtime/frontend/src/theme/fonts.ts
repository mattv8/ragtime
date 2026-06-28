const FALLBACK_BODY_FONT =
  "'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

export function getThemeFontFamily(): string {
  if (typeof document === 'undefined') {
    return FALLBACK_BODY_FONT;
  }
  const value = getComputedStyle(document.documentElement).getPropertyValue('--font-body').trim();
  return value || FALLBACK_BODY_FONT;
}
