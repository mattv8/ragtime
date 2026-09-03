import { afterEach, describe, expect, it, vi } from 'vitest';

import { CHAT_HTML_COMPONENT_BRIDGE, CHAT_HTML_COMPONENT_MESSAGE_TYPES } from './constants';
import {
  HTML_COMPONENT_BOOTSTRAP_SCRIPT,
  buildHtmlComponentSrcdoc,
  hashString,
  injectIntoHead,
  sampleHtmlComponentTheme,
  serializeForInlineScript,
  themeTokenToCssVariable,
  type HtmlComponentTheme,
} from './buildSrcdoc';

const THEME: HtmlComponentTheme = {
  pack: 'modern',
  mode: 'dark',
  tokens: {
    colorTextPrimary: '#f1f5f9',
    colorBgPrimary: '#0f172a',
    fontBody: "'Nunito', sans-serif",
    fontMono: 'monospace',
    radiusMd: '8px',
  },
};

const FULL_DOCUMENT =
  '<!doctype html><html><head><meta charset="utf-8"><title>T</title><script>window.authorRan = true;</script></head><body><div id="root"></div><script src="https://cdn.example/lib.js"></script></body></html>';

interface RagtimeApi {
  data: unknown;
  theme: HtmlComponentTheme;
  onData: (callback: (data: unknown) => void) => () => void;
  onTheme: (callback: (theme: HtmlComponentTheme) => void) => () => void;
  reportError: (error: unknown) => void;
}

function bootstrapSource(srcdoc: string): string {
  const start = srcdoc.indexOf('<script data-ragtime-bootstrap>');
  expect(start).toBeGreaterThanOrEqual(0);
  const bodyStart = start + '<script data-ragtime-bootstrap>'.length;
  const end = srcdoc.indexOf('</script>', bodyStart);
  return srcdoc.slice(bodyStart, end);
}

/** Executes the built bootstrap inside the jsdom window and returns the installed API. */
function runBootstrap(data: unknown, theme: HtmlComponentTheme = THEME): RagtimeApi {
  const srcdoc = buildHtmlComponentSrcdoc({ html: FULL_DOCUMENT, data, theme });
  new Function(bootstrapSource(srcdoc))();
  return (window as unknown as { ragtime: RagtimeApi }).ragtime;
}

function parentMessage(type: string, extra: Record<string, unknown>): MessageEvent {
  return new MessageEvent('message', {
    source: window,
    data: { bridge: CHAT_HTML_COMPONENT_BRIDGE, type, ...extra },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  delete (window as unknown as { ragtime?: unknown }).ragtime;
  document.documentElement.removeAttribute('data-theme');
  document.documentElement.removeAttribute('data-theme-pack');
  document.documentElement.removeAttribute('style');
});

describe('serializeForInlineScript', () => {
  it('escapes script terminators, comment openers and JS line separators', () => {
    const serialized = serializeForInlineScript({
      html: '</script><!-- x -->',
      text: 'a b c',
    });
    expect(serialized).not.toContain('</');
    expect(serialized).not.toContain('<!--');
    expect(serialized).not.toContain(' ');
    expect(serialized).not.toContain(' ');
    // The output is a JS expression (uses \/ and \! escapes), not strict JSON.
    expect(new Function(`return ${serialized}`)()).toEqual({
      html: '</script><!-- x -->',
      text: 'a b c',
    });
  });

  it('serializes undefined as null', () => {
    expect(serializeForInlineScript(undefined)).toBe('null');
  });
});

describe('injectIntoHead', () => {
  it('inserts right after the opening <head> tag', () => {
    expect(injectIntoHead('<html><head lang="en"><title>x</title></head></html>', '[F]')).toBe(
      '<html><head lang="en">[F]<title>x</title></head></html>',
    );
  });

  it('creates a <head> after <html> when the document has none', () => {
    expect(injectIntoHead('<html><body>hi</body></html>', '[F]')).toBe(
      '<html><head>[F]</head><body>hi</body></html>',
    );
  });

  it('prepends when there is neither <html> nor <head>', () => {
    expect(injectIntoHead('<div>fragment</div>', '[F]')).toBe('[F]<div>fragment</div>');
  });

  it('does not mistake <header> for <head>', () => {
    expect(injectIntoHead('<html><body><header>x</header></body></html>', '[F]')).toBe(
      '<html><head>[F]</head><body><header>x</header></body></html>',
    );
  });
});

describe('buildHtmlComponentSrcdoc', () => {
  it('places the bootstrap before the first author <script>', () => {
    const srcdoc = buildHtmlComponentSrcdoc({ html: FULL_DOCUMENT, data: null, theme: THEME });
    const bootstrapAt = srcdoc.indexOf('<script data-ragtime-bootstrap>');
    const authorAt = srcdoc.indexOf('<script>window.authorRan');
    expect(bootstrapAt).toBeGreaterThan(-1);
    expect(authorAt).toBeGreaterThan(bootstrapAt);
    expect(srcdoc.indexOf('<style data-ragtime-base>')).toBeLessThan(bootstrapAt);
  });

  it('handles <html>-less fragments and adds a charset meta when missing', () => {
    const srcdoc = buildHtmlComponentSrcdoc({
      html: '<div id="map"></div><script>init()</script>',
      data: null,
      theme: THEME,
    });
    expect(srcdoc.startsWith('<meta charset="utf-8">')).toBe(true);
    expect(srcdoc.indexOf('<script data-ragtime-bootstrap>')).toBeLessThan(
      srcdoc.indexOf('<script>init()'),
    );
  });

  it('creates <head> when the document only has <html>', () => {
    const srcdoc = buildHtmlComponentSrcdoc({
      html: '<html><body></body></html>',
      data: null,
      theme: THEME,
    });
    expect(srcdoc).toMatch(/^<html><head><meta charset="utf-8">/);
    expect(srcdoc).toContain('</script></head><body></body></html>');
  });

  it('does not duplicate an existing charset meta', () => {
    const srcdoc = buildHtmlComponentSrcdoc({ html: FULL_DOCUMENT, data: null, theme: THEME });
    expect(srcdoc.match(/<meta charset="utf-8">/gi)).toHaveLength(1);
  });

  it('escapes </script> and <!-- inside data so the bootstrap cannot be terminated', () => {
    const data = { note: '</script><script>alert(1)</script><!--' };
    const srcdoc = buildHtmlComponentSrcdoc({ html: FULL_DOCUMENT, data, theme: THEME });
    const script = bootstrapSource(srcdoc);
    expect(script).not.toContain('</');
    expect(script).not.toContain('<!--');
    expect(script).toContain('<\\/script>');
    expect(script).toContain(serializeForInlineScript(data));
  });

  it('inlines --ragtime-* vars, the base style, and data-theme stamping', () => {
    const srcdoc = buildHtmlComponentSrcdoc({ html: FULL_DOCUMENT, data: null, theme: THEME });
    expect(srcdoc).toContain('--ragtime-color-text-primary:#f1f5f9');
    expect(srcdoc).toContain("--ragtime-font-body:'Nunito', sans-serif");
    expect(srcdoc).toContain('color:var(--ragtime-color-text-primary)');
    expect(srcdoc).toContain("setAttribute('data-theme'");
    expect(srcdoc).toContain("setAttribute('data-theme-pack'");
  });

  it('produces identical output for identical input', () => {
    const input = { html: FULL_DOCUMENT, data: { rows: [{ a: 1 }] }, theme: THEME };
    expect(buildHtmlComponentSrcdoc(input)).toBe(buildHtmlComponentSrcdoc(input));
  });

  it('uses message type strings from constants and never contains </ in the bootstrap', () => {
    expect(HTML_COMPONENT_BOOTSTRAP_SCRIPT).not.toContain('</');
    expect(HTML_COMPONENT_BOOTSTRAP_SCRIPT).toContain(JSON.stringify(CHAT_HTML_COMPONENT_BRIDGE));
    for (const type of Object.values(CHAT_HTML_COMPONENT_MESSAGE_TYPES)) {
      expect(HTML_COMPONENT_BOOTSTRAP_SCRIPT).toContain(JSON.stringify(type));
    }
    expect(HTML_COMPONENT_BOOTSTRAP_SCRIPT).toContain('__RAGTIME_DATA__');
    expect(HTML_COMPONENT_BOOTSTRAP_SCRIPT).toContain('__RAGTIME_THEME__');
  });
});

describe('bootstrap runtime (executed in jsdom)', () => {
  it('installs window.ragtime, stamps the theme, and posts READY', () => {
    const post = vi.spyOn(window.parent, 'postMessage');
    const ragtime = runBootstrap({ rows: [1, 2] });

    expect(ragtime.data).toEqual({ rows: [1, 2] });
    expect(ragtime.theme.pack).toBe('modern');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme-pack')).toBe('modern');
    expect(document.documentElement.style.getPropertyValue('--ragtime-color-text-primary')).toBe(
      '#f1f5f9',
    );
    expect(post).toHaveBeenCalledWith(
      { bridge: CHAT_HTML_COMPONENT_BRIDGE, type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY },
      '*',
    );
  });

  it('fires onData immediately and on parent DATA messages, ignoring foreign senders', () => {
    const ragtime = runBootstrap([{ a: 1 }]);
    const seen: unknown[] = [];
    ragtime.onData((data) => seen.push(data));
    expect(seen).toEqual([[{ a: 1 }]]);

    window.dispatchEvent(
      new MessageEvent('message', {
        source: null,
        data: {
          bridge: CHAT_HTML_COMPONENT_BRIDGE,
          type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA,
          data: 'x',
          version: 1,
        },
      }),
    );
    window.dispatchEvent(
      parentMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA, { data: [{ a: 2 }], version: 1 }),
    );
    expect(seen).toEqual([[{ a: 1 }], [{ a: 2 }]]);
    expect(ragtime.data).toEqual([{ a: 2 }]);
  });

  it('does not fire onData when the initial data is null', () => {
    const ragtime = runBootstrap(null);
    const callback = vi.fn();
    const unsubscribe = ragtime.onData(callback);
    expect(callback).not.toHaveBeenCalled();
    unsubscribe();
    window.dispatchEvent(
      parentMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA, { data: 1, version: 1 }),
    );
    expect(callback).not.toHaveBeenCalled();
  });

  it('applies THEME messages in place and notifies onTheme subscribers', () => {
    const ragtime = runBootstrap(null);
    const callback = vi.fn();
    ragtime.onTheme(callback);
    const nextTheme: HtmlComponentTheme = {
      pack: 'serif',
      mode: 'light',
      tokens: { colorTextPrimary: '#0f172a', fontBody: 'Georgia, serif' },
    };
    window.dispatchEvent(
      parentMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.THEME, { theme: nextTheme }),
    );

    expect(callback).toHaveBeenCalledWith(nextTheme);
    expect(ragtime.theme).toEqual(nextTheme);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(document.documentElement.getAttribute('data-theme-pack')).toBe('serif');
    expect(document.documentElement.style.getPropertyValue('--ragtime-font-body')).toBe(
      'Georgia, serif',
    );
  });

  it('posts ERROR from reportError with the bridge marker', () => {
    const post = vi.spyOn(window.parent, 'postMessage');
    const ragtime = runBootstrap(null);
    ragtime.reportError(new Error('boom'));
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({
        bridge: CHAT_HTML_COMPONENT_BRIDGE,
        type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR,
        message: 'boom',
      }),
      '*',
    );
  });
});

describe('sampleHtmlComponentTheme', () => {
  it('reads pack/mode from the document and returns every token key', () => {
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.setAttribute('data-theme-pack', 'modern');
    const theme = sampleHtmlComponentTheme();
    expect(theme.mode).toBe('light');
    expect(theme.pack).toBe('modern');
    expect(Object.keys(theme.tokens).sort()).toEqual(
      [
        'colorTextPrimary',
        'colorTextSecondary',
        'colorTextMuted',
        'colorBgPrimary',
        'colorBgSecondary',
        'colorSurface',
        'colorBorder',
        'colorPrimary',
        'colorAccent',
        'fontBody',
        'fontMono',
        'radiusMd',
      ].sort(),
    );
    for (const value of Object.values(theme.tokens)) {
      expect(value.length).toBeGreaterThan(0);
    }
  });
});

describe('helpers', () => {
  it('maps token keys to kebab-case --ragtime- variables', () => {
    expect(themeTokenToCssVariable('colorTextPrimary')).toBe('--ragtime-color-text-primary');
    expect(themeTokenToCssVariable('fontBody')).toBe('--ragtime-font-body');
  });

  it('hashes deterministically to hex', () => {
    expect(hashString('abc')).toBe(hashString('abc'));
    expect(hashString('abc')).not.toBe(hashString('abd'));
    expect(hashString('')).toBe('1505');
    expect(hashString('abc')).toMatch(/^[0-9a-f]+$/);
  });
});
