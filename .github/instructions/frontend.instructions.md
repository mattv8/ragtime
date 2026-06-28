---
applyTo: 'ragtime/frontend/**'
---

# Ragtime Frontend - Design System

Ragtime's UI is token-driven. Every color, radius, shadow, spacing value, and font is a CSS custom property defined in `src/styles/theme.css` (the base contract) and optionally overridden per theme pack under `src/styles/themes/`.

Component CSS must never hard-code themeable values. It should read tokens so a theme applies app-wide without per-component rules.

There are two orthogonal axes, both set as attributes on `<html>`:

| Axis | Attribute | Values | Storage key | Default |
|------|-----------|--------|-------------|---------|
| Theme pack | `data-theme-pack` | `default` (attribute absent), `serif` | `ragtime-theme-pack` | `default` |
| Color mode | `data-theme` | `light`, `dark`, system (attribute absent) | `ragtime-theme` | system |

They compose: for example, `data-theme-pack="serif"` + `data-theme="light"` produces the serif pack's parchment light palette. Both are applied before first paint by the inline guard in `index.html` to avoid a flash of the wrong theme, and at runtime via `src/theme` (`setThemePack`, `setColorMode`).

## Persistence & Precedence

- Theme pack resolution is `user saved choice -> global default -> default`; `resolveThemePackId(...)` is the source of truth.
- The theme registry (`THEME_PACKS`) is canonical for available packs, user-menu cycling order, and Settings cards. There are currently only `default` and `serif`.
- `localStorage` is advisory only: `getStoredThemePack()` and `getStoredColorMode()` are guarded and fall back to defaults if storage is unavailable or invalid.
- `setThemePack()` and `setColorMode()` remove the corresponding HTML attribute and clear the storage key when switching back to `default` or `system`.
- User-menu theme changes are applied immediately in the UI and then saved best-effort to the account; the optimistic local theme stays in place even if the network save fails.
- The global default theme is changed in Settings and is intended for admins; a per-user choice always overrides it.

## Brand & Personality

- Default pack: the original Ragtime look - cool slate surfaces, indigo accent (`#6366f1`), sans-serif type (Nunito), rounded corners. Technical, modern, neutral.
- Serif pack: mature, contemporary, editorial - inspired by Claude's style guide. Warm parchment surfaces, serif headings (Source Serif 4), terracotta accent (`#c96442`), warm-toned neutrals, ring-style depth, and sharper corners. It should read like a well-set document rather than a dashboard.

## Color Tokens

Defined for each pack × color mode. Names are stable; values change per theme.

- Surfaces: `--color-bg-primary`, `--color-bg-secondary`, `--color-bg-tertiary`, `--color-surface`, `--color-surface-hover`, `--color-surface-active`
- Text: `--color-text-primary`, `--color-text-secondary`, `--color-text-muted`, `--color-text-inverse`
- Brand: `--color-primary`, `--color-primary-hover`, `--color-primary-light`, `--color-primary-soft`, `--color-primary-border`
- Accent: `--color-accent`, `--color-accent-hover`, `--color-accent-light`
- Semantic: `--color-success|error|warning|info` each with `-light` and `-border` variants
- Borders/inputs: `--color-border`, `--color-border-strong`, `--color-input-bg`, `--color-input-border`, `--color-input-focus`
- Surfaces of record: `--color-terminal-surface`, login gradient (`--login-*`) tokens

### Default palette anchors

Background `#0f172a` · Surface `#1e293b` · Primary `#6366f1` · Text `#f1f5f9` (light mode: Background `#f8fafc`, Surface `#ffffff`, Text `#0f172a`).

### Serif palette anchors

Light: Parchment `#f5f4ed` · Ivory `#faf9f5` · Terracotta `#c96442` · Near-black `#141413`. Dark: Deep dark `#141413` · `#1e1d1b` · Coral `#d97757` · Parchment text `#f5f4ed`. All neutrals are warm-toned; the only cool color is the focus blue `#3898ec`.

## Typography

- `--font-sans` - Nunito stack. The raw sans stack; reference it directly only for form controls and other UI chrome that should stay sans in every pack.
- `--font-serif` - Source Serif 4 (Georgia fallback). Available to any pack.
- `--font-body` - applied at the `body` level (see `layout.css`), so it cascades to all content that does not set its own font: chat messages, tool-call formatting, the topnav links, DataTables, modals, and settings text. Default pack maps it to `--font-sans`; the serif pack maps it to `--font-serif`. This is the primary lever that makes a pack feel app-wide.
- `--font-heading` - applied to `h1-h6`, `legend`, and the `.topnav-brand` wordmark. Default pack maps it to `--font-sans`; the serif pack maps it to `--font-serif`.
- `--font-mono` - code/terminal. Never themed.
- Scale: `--text-xs ... --text-2xl`; line-heights `--leading-tight|normal|relaxed`.

Source Serif 4 and Nunito are loaded from Google Fonts in `index.html`.

`layout.css` forces `button, input, select, textarea, optgroup { font-family: inherit; }` so form controls and button-based nav stay on-theme instead of falling back to browser UI fonts.

### Canvas/WebGL surfaces

Charts render to `<canvas>`, so CSS fonts/colors cannot inherit. Both chart surfaces read `--font-body` via `getThemeFontFamily()` (`src/theme/fonts.ts`) and set `Chart.defaults.font.family` (plus per-chart legend/title/tick fonts):

- Chat charts (`ChartDisplay` in `ChatPanel.tsx`) recreate on a `data-theme`/`data-theme-pack` MutationObserver.
- Admin charts (`react-chartjs-2` in `UsersPanel.tsx`) update the default font in the existing `useThemeColors` observer and re-render when colors change.
- DataTables inherit `--font-body` (also set explicitly on `.datatable-container table.dataTable`).
- Other script-rendered visuals that sample theme variables from the DOM must also observe theme attribute changes. `WebGLGradient` follows the same rule and refreshes on `data-theme`, `data-theme-pack`, and relevant root-style changes.

## Radius, Shadow, Spacing

- Radius scale: `--radius-xs (3px) · sm (4px) · md (8px) · lg (12px) · xl (16px) · full`. The serif pack overrides these to sharper values (`xs 1px · sm 2px · md 3px · lg 5px · xl 8px`). Component CSS uses the tokens, so switching packs reshapes every corner at once. `border-radius: 50%` (circles/avatars) and `0` (intentional squares) stay literal.
- Shadows: `--shadow-sm|md|lg|xl`. The serif pack uses softer, warmer, ring-style shadows.
- Spacing: `--space-xs ... --space-2xl` (unchanged across packs).

## Reusable Patterns

- Buttons, cards, inputs, modals, badges, tool-call blocks - all in `components.css`, token-driven.
- Appearance picker (`.appearance-theme-grid`, `.appearance-theme-card`, `.appearance-swatch`, `.appearance-mode-toggle`, `.appearance-mode-option`) in `components.css` - the Settings UI for choosing theme + color mode.
- Border idiom: `1px solid var(--color-border)` is the standard divider.

## Intentionally Fixed

Some values stay literal on purpose; do not tokenize them:

- Code-block / syntax-highlight palettes in `chat.css` (for example the Tokyo Night block `#1a1b26`, `#7aa2f7`, `#f7768e`) - code colors are stable across themes.
- `@media print` border in `responsive.css` (`#ccc`) - print is always on white.
- White text on colored buttons (`color: #fff`) and hover-darkened semantic colors that remain in the same hue family across themes.
- Categorical badge hues (purple/orange tags) that carry meaning by color.

## How to Add a New Theme

1. Create `src/styles/themes/<id>.css`. Override only the tokens you want to change, scoped to the pack, mirroring `serif.css`:
   ```css
   [data-theme-pack='<id>'] { /* pack-wide: fonts, radii, shadow shape */ }
   [data-theme-pack='<id>'] { /* dark color mode (base) */ }
   [data-theme-pack='<id>'][data-theme='light'] { /* explicit light */ }
   @media (prefers-color-scheme: light) {
     [data-theme-pack='<id>']:not([data-theme='dark']) { /* system light */ }
   }
   ```
2. `@import './themes/<id>.css';` in `src/styles/global.css` (after `theme.css`).
3. Register it in `src/theme/themes.ts` by adding a `ThemePack` entry (id, label, description, heading font preview, swatch colors). The Settings Appearance picker renders from this registry automatically.
4. No component CSS changes are needed - the pack inherits the full token contract and overrides what it declares.
