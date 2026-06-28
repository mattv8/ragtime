export type ThemePackId = 'default' | 'serif';

export interface ThemePackSwatches {
  background: string;
  surface: string;
  primary: string;
  text: string;
}

export interface ThemePack {
  id: ThemePackId;
  label: string;
  description: string;
  headingFontPreview: string;
  swatches: ThemePackSwatches;
}

export const DEFAULT_THEME_PACK_ID: ThemePackId = 'default';

export const THEME_PACK_STORAGE_KEY = 'ragtime-theme-pack';

export const THEME_PACKS: ThemePack[] = [
  {
    id: 'default',
    label: 'Default',
    description: 'The original Ragtime look: cool slate surfaces, indigo accent, rounded corners.',
    headingFontPreview:
      "'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    swatches: {
      background: '#0f172a',
      surface: '#1e293b',
      primary: '#6366f1',
      text: '#f1f5f9',
    },
  },
  {
    id: 'serif',
    label: 'Serif',
    description:
      'Mature and editorial: warm parchment surfaces, serif headings, terracotta accent, sharper corners.',
    headingFontPreview: "'Source Serif 4', Georgia, 'Times New Roman', serif",
    swatches: {
      background: '#f5f4ed',
      surface: '#faf9f5',
      primary: '#c96442',
      text: '#141413',
    },
  },
];

export function isThemePackId(value: string | null | undefined): value is ThemePackId {
  return THEME_PACKS.some((pack) => pack.id === value);
}

export function getThemePack(id: ThemePackId): ThemePack {
  return THEME_PACKS.find((pack) => pack.id === id) ?? THEME_PACKS[0];
}
