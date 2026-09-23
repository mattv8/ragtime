# Ragtime UX Review Reference

Use this guide to review a proposed frontend change against the implemented
Ragtime interface. It is a review aid, not a replacement for the source. Mark
applicable checklist items **Pass**, **Fail**, or **N/A**, then report failed
items as findings with user impact and an exact `path:line` reference.
Distinguish a required contract from a local legacy limitation; do not turn a
caveat into a new convention.

## Authority and source map

Resolve conflicts in this order: the relevant implementation and tests, the
token/theme files, shared CSS, then this guide. Start with:

- `ragtime/frontend/src/styles/theme.css` — base token contract and Default.
- `ragtime/frontend/src/styles/themes/modern.css` and `themes/serif.css` — pack
  overrides and Modern's canonical workbench hierarchy.
- `ragtime/frontend/src/styles/components.css`, `layout.css`, `workbench.css`,
  `workbench-chat.css`, `workbench-userspace.css`, `workbench-admin.css`,
  `chat.css`, and `responsive.css` — reusable controls and breakpoints.
- `ragtime/frontend/src/theme/` — pack/mode resolution and runtime updates.
- The component being changed plus its adjacent tests. Existing behavior is the
  baseline; verify whether it is deliberate before treating it as a pattern.

## Shared contracts

- **Tokens first.** Colors, typography, radii, shadows, spacing, focus, and
  z-index must use the CSS custom-property contract. Do not hard-code themeable
  values. Intentional exceptions include syntax highlighting, print borders,
  white text on colored controls, and categorical badge hues.
- **Type and rhythm.** Use `--font-body`, `--font-heading`, `--font-mono`, the
  `--text-*` scale, `--leading-*`, `--space-*`, and `--radius-*`. Controls
  inherit the active pack; preserve monospace for code, paths, and terminals.
- **Surfaces and borders.** Use the named surface tokens and the normal divider
  idiom, `1px solid var(--color-border)`. Reserve stronger borders, shadows, and
  `--color-surface-active` for a meaningful state change, not decoration.
- **Feedback and focus.** Success, error, warning, and info states need text or
  icon meaning in addition to color. Every keyboard action needs a visible
  token-based focus indication. Preserve semantic roles, labels, `aria-*`
  relationships, live-region urgency, and keyboard Escape/Enter/Space behavior.
  Check WCAG AA contrast at 4.5:1 for normal text and 3:1 for large text and
  meaningful non-text controls; do not assume current tokens pass.
- **Identity hooks.** Every unique primary page, section, panel, card, modal,
  form, toolbar, table/list, and repeated domain record needs a stable,
  human-readable named identity. Use a unique `id` where it is unique in the
  document; otherwise use a stable semantic `data-*` value or a durable domain
  identifier. Never use an array index, random value, or styling class as the
  hook. Incidental wrappers and leaves do not need one.
- **Responsive, touch, motion.** Test desktop and narrow layouts. Preserve
  `responsive.css`'s stacked forms, wrapped tabs, mobile modal sizing, and
  User Space pane reflow. Coarse pointers require usable targets (normally
  44px); hover-only controls must remain discoverable by keyboard and touch.
  Keep `prefers-reduced-motion` behavior intact; do not add decorative motion
  that conceals state.
- **Overlays.** Use the existing modal/popover/portal layers and token z-index
  order (`dropdown`, `sticky`, `modal`, `tooltip`, `toast`). An overlay must not
  clip behind a pane or iframe, must dismiss predictably, and must retain focus
  where the established component does so.

## Theme-pack review

The pack is `data-theme-pack` (`default` is absent) and mode is `data-theme`
(`light`, `dark`, or absent for system). Explicit light and system-light blocks
are intentionally equivalent. Review every changed surface in Default, Modern,
and Serif, in dark and light/system.

### Default

Cool slate surfaces, indigo primary, Nunito, rounded geometry, and ordinary
shadow depth. It is the base token contract in `styles/theme.css`. Check that
new UI reads as technical and neutral, without creating a competing palette.

### Modern

`styles/themes/modern.css` is authoritative: compact Droid Sans/Fira Code,
blue accent, square-ish controls, 4px grid, structural borders, and no small or
medium shadows. Its hierarchy is strict: workbench desk → transparent route or
page container → `--color-panel` section → one `--color-widget` record layer.
This desk → panel → widget hierarchy applies to Chat, Workspaces, and Admin
surfaces.
Overlays remain widget surfaces with overlay shadows. Check 35px titlebars,
32px toolbars, 28px controls, 4px sashes, 8px pane inset/radius where relevant.
Do not add soft cards, double-nested bordered containers, or a second record
elevation inside a section. Revealed controls must also appear on focus,
selection, and touch.

### Serif

Warm parchment/ivory light surfaces and near-black/warm dark surfaces,
terracotta primary, Source Serif 4 body/headings, sharper radii, and soft
ring-style depth. Preserve the cool blue focus token in light mode and legible
warm semantic feedback. Light mode uses cool blue `#3898ec`; dark mode aliases
focus to the terracotta primary. Check focus contrast in both modes. Review
dense controls especially: editorial typography must remain scannable and never
turn code or UI chrome into serif by accident.

## Recurring interaction patterns

- **Workbench panes.** Maintain a clear active pane and structural separators.
  `ResizeHandle.tsx` is the accessibility exemplar: `role="separator"`, value
  semantics, pointer capture, arrow/Home/End controls, and collapse/restore.
  `UserSpacePanel.tsx` is the responsive multi-pane exemplar; mobile hides
  sashes and reorders panes rather than forcing a squeezed desktop layout.
- **Cards, records, and summaries.** Make the primary action, status, metadata,
  and secondary actions easy to scan. Compact summaries may hide detail, but
  selected/expanded states and unavailable reasons must remain explicit.
- **Forms and wizards.** Group related fields, give errors an associated label,
  preserve entered work, and make the next action/irreversible consequence
  clear. `ToolWizard.tsx` and `WorkspaceScmWizard.tsx` are broad, stateful
  references: review step progress, validation, busy/error/retry/result states,
  and cancellation/close behavior rather than copying their structure blindly.
- **Discovery, search, filtering, and tables.** `SearchFilterBar.tsx` supports
  tag creation, debounced filtering, URL state, completion, clear, and keyboard
  handling. Results must state empty, filtered-empty, loading, and failure
  conditions. Preserve headings/cells and responsive table-to-card behavior;
  `ToolAccessEditor.tsx` shows ARIA table and listbox semantics.
- **Tabs, async progress, and destructive actions.** Tabs require correct
  tablist/tab/tabpanel relationships and a visible selected state. Long work
  needs a progress/status signal and a terminal result. Use explicit confirm,
  consequence, and recovery paths for destructive actions; do not rely on color
  or a transient toast as the only evidence.
- **File and diff views.** `FileDiffOverlay.tsx` provides per-file navigation,
  loading, error, no-diff, and changed-line summaries. Preserve filenames,
  operation/status labels, before/after context, scrollability, and narrow
  single-column behavior.
- **Charts, canvas, and iframes.** Canvas cannot inherit CSS: follow the
  theme-observing chart pattern in `ChatPanel.tsx`/`UsersPanel.tsx` and use
  `getThemeFontFamily()`. `HtmlComponentDisplay.tsx` is the iframe exemplar:
  sandbox policy, loading/ready/error states, bounded height, and theme/data
  messages must work without reloading on theme changes. Portal dropdowns such
  as `ToolSelectorDropdown.tsx` calculate viewport position to draw above
  iframes; review clipping, Escape/outside dismissal, focus, and small windows.
- **Transient status.** `Toast.tsx` has timed success/info/error feedback and
  a dismiss affordance; its current container is `aria-live="polite"` while its
  items use `role="alert"`. When reusing it, verify announcement urgency and
  duplication. `UserSpaceStatusOverlay.tsx` supplies polite persistent status;
  also check timing, dismissibility, and whether a durable in-context error is
  required.
- **Credential dialogs.** `ExternalApiCredentialDialogs.tsx` is the modal
  exemplar: dialog semantics, `aria-modal`, labelled title, focus trap, and
  curl and Power Query credential-use tabs. Secrets must not be exposed by
  visual truncation, clipboard surprises, or an unprotected secondary view.

## Audited caveats — inspect, do not copy

- `FileDiffOverlay.tsx` is a click-backdrop overlay without dialog semantics,
  focus trapping, or Escape dismissal. Treat it as a review target, not a
  general modal template.
- `ToolSelectorDropdown.tsx` portals and positions a complex menu correctly
  above iframes, but its `div` `role="button"` group header contains a nested
  checkbox; prefer valid native/control structures for new UI.
- `SearchFilterBar.tsx`'s composite tag key includes the array index because
  tags are mutable local state. This is not permission to use index-derived
  identity for primary repeated UI.
- `ToastContainer` combines a polite container with alert items. Review new
  announcements for duplication and choose urgency deliberately.

## Quick review checklist

Mark each **Pass / Fail / N/A**:

- Hierarchy is clear; surfaces, borders, spacing, and density match the pack.
- Default, Modern, and Serif work in dark, light, and system mode.
- Unique primary boundaries have stable, named IDs or semantic data hooks.
- Loading, empty, filtered-empty, success, error, disabled, and progress states
  are explicit where applicable.
- Keyboard, focus, labels, roles, live feedback, contrast, touch targets, and
  reduced motion are preserved.
- Narrow/mobile reflow, panes, forms, tabs, tables, and overlays remain usable.
- Portals, charts/canvas, and embedded iframes respect theme changes, layering,
  sizing, sandboxing, and fallback/error states.
