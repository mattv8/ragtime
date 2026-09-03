import { getThemeFontFamily } from '@/theme/fonts';
import { getThemeSnapshot } from '@/theme/themeSnapshot';

import { CHAT_HTML_COMPONENT_BRIDGE, CHAT_HTML_COMPONENT_MESSAGE_TYPES } from './constants';

export interface HtmlComponentTheme {
  pack: string;
  mode: 'light' | 'dark';
  tokens: Record<string, string>;
}

/**
 * Token key -> app CSS custom property it is sampled from. Order is stable so
 * the generated `:root` block is deterministic. `fontBody` is sampled through
 * getThemeFontFamily() instead (it resolves the `--font-body` alias chain).
 */
const THEME_TOKEN_SOURCES: ReadonlyArray<readonly [key: string, cssVariable: string]> = [
  ['colorTextPrimary', '--color-text-primary'],
  ['colorTextSecondary', '--color-text-secondary'],
  ['colorTextMuted', '--color-text-muted'],
  ['colorBgPrimary', '--color-bg-primary'],
  ['colorBgSecondary', '--color-bg-secondary'],
  ['colorSurface', '--color-surface'],
  ['colorBorder', '--color-border'],
  ['colorPrimary', '--color-primary'],
  ['colorAccent', '--color-accent'],
  ['fontMono', '--font-mono'],
  ['radiusMd', '--radius-md'],
];

/**
 * Used when a computed custom property is empty (jsdom, or a stylesheet that
 * has not loaded yet) so the component still receives a coherent palette.
 * Values mirror the app's default theme pack.
 */
const FALLBACK_TOKENS: Record<'light' | 'dark', Record<string, string>> = {
  dark: {
    colorTextPrimary: '#f1f5f9',
    colorTextSecondary: '#94a3b8',
    colorTextMuted: '#64748b',
    colorBgPrimary: '#0f172a',
    colorBgSecondary: '#1e293b',
    colorSurface: '#1e293b',
    colorBorder: 'rgba(255, 255, 255, 0.1)',
    colorPrimary: '#6366f1',
    colorAccent: '#0ea5e9',
    fontMono: "'JetBrains Mono', 'SF Mono', Monaco, 'Fira Code', Consolas, monospace",
    radiusMd: '8px',
  },
  light: {
    colorTextPrimary: '#0f172a',
    colorTextSecondary: '#475569',
    colorTextMuted: '#94a3b8',
    colorBgPrimary: '#f8fafc',
    colorBgSecondary: '#ffffff',
    colorSurface: '#ffffff',
    colorBorder: 'rgba(0, 0, 0, 0.08)',
    colorPrimary: '#4f46e5',
    colorAccent: '#0284c7',
    fontMono: "'JetBrains Mono', 'SF Mono', Monaco, 'Fira Code', Consolas, monospace",
    radiusMd: '8px',
  },
};

const HTML_COMPONENT_BASE_STYLE =
  'html,body{margin:0;background:transparent;color:var(--ragtime-color-text-primary);font-family:var(--ragtime-font-body)}' +
  '*,*::before,*::after{box-sizing:border-box}' +
  'img,svg,canvas,video{max-width:100%}';

export function sampleHtmlComponentTheme(): HtmlComponentTheme {
  const snapshot = getThemeSnapshot();
  const fallback = FALLBACK_TOKENS[snapshot.mode];
  const computed =
    typeof document !== 'undefined' && typeof getComputedStyle === 'function'
      ? getComputedStyle(document.documentElement)
      : null;

  const tokens: Record<string, string> = {};
  for (const [key, cssVariable] of THEME_TOKEN_SOURCES) {
    const sampled = computed?.getPropertyValue(cssVariable).trim() ?? '';
    tokens[key] = sampled || fallback[key];
  }
  tokens.fontBody = getThemeFontFamily();

  return { pack: snapshot.pack, mode: snapshot.mode, tokens };
}

/**
 * JSON-encodes a value so it can be inlined inside a `<script>` element
 * without terminating the element (`</script>`), opening an HTML comment, or
 * breaking on JS line terminators that JSON leaves unescaped.
 */
export function serializeForInlineScript(value: unknown): string {
  const json = JSON.stringify(value === undefined ? null : value) ?? 'null';
  return json
    .replace(/<\//g, '<\\/')
    .replace(/<!--/g, '<\\!--')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

/** `colorTextPrimary` -> `--ragtime-color-text-primary`. */
export function themeTokenToCssVariable(tokenKey: string): string {
  return `--ragtime-${tokenKey.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`)}`;
}

/**
 * Child side of the bridge (PRD 6.10 / 6.11). Plain ES5 so it runs in any
 * engine the sandboxed frame might use. Placeholders are substituted by
 * buildHtmlComponentSrcdoc. The source must never contain the sequence `</`,
 * otherwise the inline <script> element would be cut short by the parser.
 */
export const HTML_COMPONENT_BOOTSTRAP_SCRIPT: string = `(function () {
  'use strict';
  var BRIDGE = ${JSON.stringify(CHAT_HTML_COMPONENT_BRIDGE)};
  var TYPES = ${JSON.stringify(CHAT_HTML_COMPONENT_MESSAGE_TYPES)};
  var PREFIX = '--ragtime-';
  var data = __RAGTIME_DATA__;
  var theme = __RAGTIME_THEME__;
  var dataVersion = -1;
  var dataListeners = [];
  var themeListeners = [];
  var lastHeight = 0;
  var hasOwn = Object.prototype.hasOwnProperty;

  function post(type, payload) {
    var message = { bridge: BRIDGE, type: type };
    if (payload) {
      for (var key in payload) {
        if (hasOwn.call(payload, key)) message[key] = payload[key];
      }
    }
    try {
      window.parent.postMessage(message, '*');
    } catch (postError) {
      /* parent unavailable */
    }
  }

  function describeError(error) {
    if (error && typeof error === 'object') {
      var described = { message: String(error.message || error.name || 'Unknown error') };
      if (typeof error.stack === 'string') described.stack = error.stack;
      return described;
    }
    return { message: error === undefined ? 'Unknown error' : String(error) };
  }

  function reportError(error) {
    post(TYPES.ERROR, describeError(error));
  }

  function subscribe(list, callback) {
    if (typeof callback !== 'function') return function () {};
    list.push(callback);
    return function () {
      var index = list.indexOf(callback);
      if (index >= 0) list.splice(index, 1);
    };
  }

  function notify(list, value) {
    var snapshot = list.slice();
    for (var i = 0; i < snapshot.length; i++) {
      try {
        snapshot[i](value);
      } catch (callbackError) {
        reportError(callbackError);
      }
    }
  }

  function toKebab(key) {
    return key.replace(/[A-Z]/g, function (char) {
      return '-' + char.toLowerCase();
    });
  }

  function applyTheme(next) {
    if (!next || typeof next !== 'object') return;
    theme = next;
    ragtime.theme = next;
    var root = document.documentElement;
    if (!root) return;
    if (next.mode) root.setAttribute('data-theme', String(next.mode));
    if (next.pack) root.setAttribute('data-theme-pack', String(next.pack));
    var tokens = next.tokens || {};
    for (var key in tokens) {
      if (hasOwn.call(tokens, key)) {
        root.style.setProperty(PREFIX + toKebab(key), String(tokens[key]));
      }
    }
  }

  function applyData(next, version) {
    if (typeof version === 'number') {
      if (version < dataVersion) return;
      dataVersion = version;
    }
    data = next;
    ragtime.data = next;
    notify(dataListeners, next);
  }

  function measureHeight() {
    var root = document.documentElement;
    var body = document.body;
    return Math.max(root ? root.scrollHeight : 0, body ? body.scrollHeight : 0);
  }

  function reportSize() {
    var height = measureHeight();
    if (!(height > 0) || Math.abs(height - lastHeight) < 2) return;
    lastHeight = height;
    post(TYPES.RESIZE, { height: height });
  }

  function startSizing() {
    if (typeof ResizeObserver === 'function') {
      var observer = new ResizeObserver(function () {
        reportSize();
      });
      if (document.documentElement) observer.observe(document.documentElement);
      if (document.body) observer.observe(document.body);
    } else {
      window.addEventListener('resize', reportSize);
    }
    window.addEventListener('load', reportSize);
    reportSize();
  }

  function onReady() {
    startSizing();
    post(TYPES.READY, {});
  }

  var ragtime = {
    data: data,
    theme: theme,
    onData: function (callback) {
      var unsubscribe = subscribe(dataListeners, callback);
      if (data !== null && data !== undefined && typeof callback === 'function') {
        try {
          callback(data);
        } catch (callbackError) {
          reportError(callbackError);
        }
      }
      return unsubscribe;
    },
    onTheme: function (callback) {
      return subscribe(themeListeners, callback);
    },
    reportError: reportError
  };
  window.ragtime = ragtime;

  applyTheme(theme);

  window.addEventListener('message', function (event) {
    if (event.source !== window.parent) return;
    var message = event.data;
    if (!message || message.bridge !== BRIDGE) return;
    if (message.type === TYPES.DATA) {
      applyData(message.data, message.version);
    } else if (message.type === TYPES.THEME) {
      applyTheme(message.theme);
      notify(themeListeners, theme);
    }
  });

  window.addEventListener('error', function (event) {
    var payload = describeError(event && event.error);
    if (event && typeof event.message === 'string' && event.message) payload.message = event.message;
    if (event && typeof event.filename === 'string' && event.filename) payload.source = event.filename;
    if (event && typeof event.lineno === 'number' && event.lineno > 0) payload.line = event.lineno;
    post(TYPES.ERROR, payload);
  });

  window.addEventListener('unhandledrejection', function (event) {
    reportError(event ? event.reason : undefined);
  });

  if (document.readyState !== 'loading') {
    onReady();
  } else {
    document.addEventListener('DOMContentLoaded', onReady);
  }
})();`;

const HEAD_OPEN_TAG = /<head\b[^>]*>/i;
const HTML_OPEN_TAG = /<html\b[^>]*>/i;
const CHARSET_META = /<meta\b[^>]*charset\s*=/i;
const BOOTSTRAP_PLACEHOLDER = /__RAGTIME_(DATA|THEME)__/g;

/**
 * Inserts `fragment` immediately after the first `<head …>` tag. Without a
 * head, one is created right after `<html …>`; without either, the fragment
 * is prepended so it still precedes any author markup.
 */
export function injectIntoHead(html: string, fragment: string): string {
  const headMatch = HEAD_OPEN_TAG.exec(html);
  if (headMatch) {
    const insertAt = headMatch.index + headMatch[0].length;
    return `${html.slice(0, insertAt)}${fragment}${html.slice(insertAt)}`;
  }

  const htmlMatch = HTML_OPEN_TAG.exec(html);
  if (htmlMatch) {
    const insertAt = htmlMatch.index + htmlMatch[0].length;
    return `${html.slice(0, insertAt)}<head>${fragment}</head>${html.slice(insertAt)}`;
  }

  return `${fragment}${html}`;
}

/** Strips characters that could terminate a CSS declaration block or the style element. */
function sanitizeCssValue(value: string): string {
  return String(value).replace(/[<>{};]/g, '');
}

function buildThemeStyle(theme: HtmlComponentTheme): string {
  const declarations = Object.keys(theme.tokens)
    .sort()
    .map((key) => `${themeTokenToCssVariable(key)}:${sanitizeCssValue(theme.tokens[key])}`)
    .join(';');
  return `:root{${declarations}}${HTML_COMPONENT_BASE_STYLE}`;
}

export function buildHtmlComponentSrcdoc(input: {
  html: string;
  data: unknown;
  theme: HtmlComponentTheme;
}): string {
  const { html, data, theme } = input;
  const serializedData = serializeForInlineScript(data);
  const serializedTheme = serializeForInlineScript(theme);
  const bootstrap = HTML_COMPONENT_BOOTSTRAP_SCRIPT.replace(
    BOOTSTRAP_PLACEHOLDER,
    (_match, name) => (name === 'DATA' ? serializedData : serializedTheme),
  );

  const parts: string[] = [];
  if (!CHARSET_META.test(html)) {
    parts.push('<meta charset="utf-8">');
  }
  parts.push(`<style data-ragtime-base>${buildThemeStyle(theme)}</style>`);
  parts.push(`<script data-ragtime-bootstrap>${bootstrap}</script>`);

  return injectIntoHead(html, parts.join('\n'));
}

/** djb2 hash rendered as unsigned hex; stable across runs for identical input. */
export function hashString(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) + hash + value.charCodeAt(index)) | 0;
  }
  return (hash >>> 0).toString(16);
}
