export const CHAT_HTML_COMPONENT_BRIDGE = 'chat-html-component-v1' as const;

export const CHAT_HTML_COMPONENT_MESSAGE_TYPES = {
  READY: 'ragtime-component-ready',
  RESIZE: 'ragtime-component-resize',
  ERROR: 'ragtime-component-error',
  DATA: 'ragtime-component-data',
  THEME: 'ragtime-component-theme',
} as const;

export const HTML_COMPONENT_MIN_HEIGHT = 200;
export const HTML_COMPONENT_DEFAULT_HEIGHT = 400;
export const HTML_COMPONENT_MAX_HEIGHT_RATIO = 0.7;
export const HTML_COMPONENT_EXPANDED_MAX_HEIGHT_RATIO = 0.85;
export const HTML_COMPONENT_READY_TIMEOUT_MS = 8000;

/**
 * Sandbox flags that are always withheld from in-chat HTML components.
 * A srcdoc iframe inherits the app origin, so allow-same-origin would hand
 * agent-authored code the user's session; top navigation could replace the
 * Ragtime UI with an arbitrary destination.
 */
export const CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS = [
  'allow-same-origin',
  'allow-top-navigation',
  'allow-top-navigation-by-user-activation',
  'allow-top-navigation-to-custom-protocols',
] as const;
