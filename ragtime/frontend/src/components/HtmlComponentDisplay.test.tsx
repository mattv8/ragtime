import { EditorView } from '@codemirror/view';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CHAT_HTML_COMPONENT_BRIDGE,
  CHAT_HTML_COMPONENT_MESSAGE_TYPES,
  HTML_COMPONENT_MAX_HEIGHT_RATIO,
  HTML_COMPONENT_MIN_HEIGHT,
  HTML_COMPONENT_READY_TIMEOUT_MS,
} from '@/utils/htmlComponent/constants';

const { apiMock, policyMock } = vi.hoisted(() => ({
  apiMock: {
    getUserSpacePreviewSettings: vi.fn(),
  },
  policyMock: {
    useChatComponentSandboxPolicy: vi.fn(),
  },
}));

vi.mock('@/api/client', () => ({ api: apiMock }));
vi.mock('@/utils/htmlComponent/sandboxPolicy', () => ({
  useChatComponentSandboxPolicy: policyMock.useChatComponentSandboxPolicy,
}));

import type { HtmlComponentData } from './HtmlComponentDisplay';
import { HtmlComponentDisplay } from './HtmlComponentDisplay';

// jsdom has no layout: CodeMirror measures selection geometry through Range on focus, so
// give it inert implementations instead of letting a timer-driven measure throw.
if (typeof Range !== 'undefined') {
  const emptyRect = { x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 };
  if (typeof Range.prototype.getClientRects !== 'function') {
    Range.prototype.getClientRects = () =>
      ({
        length: 0,
        item: () => null,
        [Symbol.iterator]: [][Symbol.iterator],
      }) as unknown as DOMRectList;
  }
  if (typeof Range.prototype.getBoundingClientRect !== 'function') {
    Range.prototype.getBoundingClientRect = () =>
      ({ ...emptyRect, toJSON: () => emptyRect }) as DOMRect;
  }
}

const SAMPLE_HTML =
  '<!doctype html><html><head><title>Map</title></head><body><div id="map"></div><script>console.log("hi")</script></body></html>';

function makeComponent(overrides: Partial<HtmlComponentData> = {}): HtmlComponentData {
  return {
    __html_component__: true,
    title: 'Shipments by origin',
    html: SAMPLE_HTML,
    data: { columns: ['lat', 'lng'], rows: [{ lat: 31.9, lng: -99.9 }], row_count: 1 },
    ...overrides,
  };
}

function readyPolicy(sandboxAttribute = 'allow-scripts allow-popups') {
  policyMock.useChatComponentSandboxPolicy.mockReturnValue({
    status: 'ready',
    sandboxAttribute,
    blockedReason: null,
  });
}

function getFrame(): HTMLIFrameElement {
  const frame = document.querySelector('iframe.html-component-frame');
  expect(frame).not.toBeNull();
  return frame as HTMLIFrameElement;
}

function dispatchBridgeMessage(
  frame: HTMLIFrameElement,
  data: Record<string, unknown>,
  options: { origin?: string; source?: Window | null } = {},
) {
  const source = 'source' in options ? options.source : frame.contentWindow;
  act(() => {
    window.dispatchEvent(
      new MessageEvent('message', {
        origin: options.origin ?? 'null',
        source: source ?? undefined,
        data,
      }),
    );
  });
}

function bridgeMessage(type: string, payload: Record<string, unknown> = {}) {
  return { bridge: CHAT_HTML_COMPONENT_BRIDGE, type, ...payload };
}

function spyOnFramePosts(frame: HTMLIFrameElement) {
  const frameWindow = frame.contentWindow;
  expect(frameWindow).not.toBeNull();
  return vi.spyOn(frameWindow as Window, 'postMessage').mockImplementation(() => undefined);
}

function postedTypes(spy: ReturnType<typeof spyOnFramePosts>): string[] {
  return spy.mock.calls.map((call) => (call[0] as { type: string }).type);
}

describe('HtmlComponentDisplay', () => {
  beforeEach(() => {
    apiMock.getUserSpacePreviewSettings.mockResolvedValue({
      userspace_preview_sandbox_flags: ['allow-scripts'],
    });
    readyPolicy();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.clearAllMocks();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-theme-pack');
  });

  it('renders a sandboxed srcdoc frame that never carries same-origin or top-navigation flags', () => {
    readyPolicy(
      'allow-scripts allow-same-origin allow-top-navigation allow-top-navigation-by-user-activation allow-popups',
    );
    const component = makeComponent({ height: 480 });

    render(<HtmlComponentDisplay component={component} />);

    const frame = getFrame();
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts allow-popups');
    expect(frame.getAttribute('sandbox')).not.toContain('allow-same-origin');
    expect(frame.getAttribute('sandbox')).not.toMatch(/allow-top-navigation/);
    expect(frame.getAttribute('referrerpolicy')).toBe('no-referrer');
    expect(frame.hasAttribute('loading')).toBe(false);
    expect(frame.getAttribute('title')).toBe('Shipments by origin');
    expect(frame.style.height).toBe('480px');
    expect(frame.getAttribute('srcdoc')).toContain('<div id="map"></div>');
    expect(frame.getAttribute('srcdoc')).toContain(CHAT_HTML_COMPONENT_BRIDGE);
    // The agent HTML only ever reaches srcDoc, never the chat DOM.
    expect(document.querySelector('#map')).toBeNull();
    expect(document.querySelector('.html-component-status-overlay')).not.toBeNull();
    expect(screen.getByText('Shipments by origin')).toBeTruthy();
  });

  it('renders a blocked card instead of the frame when the policy fetch rejects', () => {
    policyMock.useChatComponentSandboxPolicy.mockReturnValue({
      status: 'error',
      sandboxAttribute: '',
      blockedReason: null,
    });

    render(<HtmlComponentDisplay component={makeComponent()} />);

    expect(document.querySelector('iframe')).toBeNull();
    const blocked = document.querySelector('.html-component-blocked');
    expect(blocked).not.toBeNull();
    expect(blocked?.textContent).toMatch(/sandbox policy could not be loaded/i);
  });

  it('renders the policy blocked reason instead of the frame when scripts are disabled', () => {
    policyMock.useChatComponentSandboxPolicy.mockReturnValue({
      status: 'ready',
      sandboxAttribute: '',
      blockedReason: 'Interactive components are disabled by the preview sandbox policy.',
    });

    render(<HtmlComponentDisplay component={makeComponent()} />);

    expect(document.querySelector('iframe')).toBeNull();
    expect(document.querySelector('.html-component-blocked')?.textContent).toBe(
      'Interactive components are disabled by the preview sandbox policy.',
    );
  });

  it('shows a loading status and no frame while the policy is loading', () => {
    policyMock.useChatComponentSandboxPolicy.mockReturnValue({
      status: 'loading',
      sandboxAttribute: '',
      blockedReason: null,
    });

    render(<HtmlComponentDisplay component={makeComponent()} />);

    expect(document.querySelector('iframe')).toBeNull();
    expect(document.querySelector('.html-component-blocked')).toBeNull();
    expect(document.querySelector('.html-component-status')).not.toBeNull();
  });

  it('flips to ready on READY and posts THEME then DATA to the frame', () => {
    const component = makeComponent();
    render(<HtmlComponentDisplay component={component} />);

    const frame = getFrame();
    const postSpy = spyOnFramePosts(frame);

    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));

    expect(document.querySelector('.html-component-status-overlay')).toBeNull();
    expect(postedTypes(postSpy)).toEqual([
      CHAT_HTML_COMPONENT_MESSAGE_TYPES.THEME,
      CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA,
    ]);

    const [themeCall, dataCall] = postSpy.mock.calls;
    expect(themeCall[1]).toBe('*');
    expect(dataCall[1]).toBe('*');
    expect(themeCall[0]).toMatchObject({
      bridge: CHAT_HTML_COMPONENT_BRIDGE,
      theme: expect.objectContaining({ pack: expect.any(String), mode: expect.any(String) }),
    });
    expect(dataCall[0]).toEqual({
      bridge: CHAT_HTML_COMPONENT_BRIDGE,
      type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA,
      data: component.data,
      version: 1,
    });
  });

  it('clamps RESIZE heights to the minimum and the viewport ratio', () => {
    const originalInnerHeight = window.innerHeight;
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 1000 });
    try {
      render(<HtmlComponentDisplay component={makeComponent()} />);
      const frame = getFrame();

      dispatchBridgeMessage(
        frame,
        bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.RESIZE, { height: 12 }),
      );
      expect(frame.style.height).toBe(`${HTML_COMPONENT_MIN_HEIGHT}px`);

      dispatchBridgeMessage(
        frame,
        bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.RESIZE, { height: 9000 }),
      );
      expect(frame.style.height).toBe(`${Math.floor(1000 * HTML_COMPONENT_MAX_HEIGHT_RATIO)}px`);

      dispatchBridgeMessage(
        frame,
        bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.RESIZE, { height: 455.4 }),
      );
      expect(frame.style.height).toBe('455px');

      dispatchBridgeMessage(
        frame,
        bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.RESIZE, { height: 'tall' }),
      );
      expect(frame.style.height).toBe('455px');
    } finally {
      Object.defineProperty(window, 'innerHeight', {
        configurable: true,
        value: originalInnerHeight,
      });
    }
  });

  it('shows an error banner, reports each distinct error once, and reloads the frame on demand', () => {
    const onDisplayError = vi.fn();
    render(<HtmlComponentDisplay component={makeComponent()} onDisplayError={onDisplayError} />);

    const frame = getFrame();
    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));
    dispatchBridgeMessage(
      frame,
      bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR, { message: ' boom ' }),
    );
    dispatchBridgeMessage(
      frame,
      bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR, { message: 'boom' }),
    );

    const banner = document.querySelector('.html-component-error');
    expect(banner).not.toBeNull();
    expect(banner?.textContent).toContain('boom');
    expect(onDisplayError).toHaveBeenCalledTimes(1);
    expect(onDisplayError).toHaveBeenCalledWith('boom');

    dispatchBridgeMessage(
      frame,
      bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR, { message: 'second failure' }),
    );
    expect(onDisplayError).toHaveBeenCalledTimes(2);
    expect(onDisplayError).toHaveBeenLastCalledWith('second failure');

    const initialSrcdoc = frame.getAttribute('srcdoc');
    fireEvent.click(screen.getByRole('button', { name: 'Reload component' }));

    const reloadedFrame = getFrame();
    expect(reloadedFrame).not.toBe(frame);
    expect(reloadedFrame.getAttribute('srcdoc')).toBe(initialSrcdoc);
    expect(document.querySelector('.html-component-error')).toBeNull();
    expect(document.querySelector('.html-component-status-overlay')).not.toBeNull();

    // After a reload the same message is reported again (state was reset).
    dispatchBridgeMessage(
      reloadedFrame,
      bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR, { message: 'boom' }),
    );
    expect(onDisplayError).toHaveBeenCalledTimes(3);
  });

  it('ignores messages with a foreign source, a non-opaque origin, or a missing bridge id', () => {
    const onDisplayError = vi.fn();
    render(<HtmlComponentDisplay component={makeComponent()} onDisplayError={onDisplayError} />);

    const frame = getFrame();
    const postSpy = spyOnFramePosts(frame);

    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY), {
      source: window,
    });
    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY), {
      origin: 'https://evil.example',
    });
    dispatchBridgeMessage(frame, {
      type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR,
      message: 'spoofed',
    });
    dispatchBridgeMessage(frame, {
      bridge: 'some-other-bridge',
      type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR,
      message: 'spoofed',
    });

    expect(postSpy).not.toHaveBeenCalled();
    expect(onDisplayError).not.toHaveBeenCalled();
    expect(document.querySelector('.html-component-status-overlay')).not.toBeNull();
    expect(document.querySelector('.html-component-error')).toBeNull();
  });

  it('posts DATA when component.data changes without rebuilding the srcdoc', () => {
    const component = makeComponent();
    const { rerender } = render(<HtmlComponentDisplay component={component} />);

    const frame = getFrame();
    const postSpy = spyOnFramePosts(frame);
    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));
    const initialSrcdoc = frame.getAttribute('srcdoc');
    postSpy.mockClear();

    const refreshedData = {
      columns: ['lat', 'lng'],
      rows: [{ lat: 40.7, lng: -74 }],
      row_count: 1,
    };
    rerender(<HtmlComponentDisplay component={{ ...component, data: refreshedData }} />);

    expect(getFrame()).toBe(frame);
    expect(frame.getAttribute('srcdoc')).toBe(initialSrcdoc);
    expect(postedTypes(postSpy)).toEqual([CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA]);
    expect(postSpy.mock.calls[0][0]).toEqual({
      bridge: CHAT_HTML_COMPONENT_BRIDGE,
      type: CHAT_HTML_COMPONENT_MESSAGE_TYPES.DATA,
      data: refreshedData,
      version: 2,
    });

    // Re-rendering with the same data identity does not repost.
    rerender(<HtmlComponentDisplay component={{ ...component, data: refreshedData }} />);
    expect(postSpy).toHaveBeenCalledTimes(1);
  });

  it('does not post DATA on data changes before the frame is ready', () => {
    const component = makeComponent();
    const { rerender } = render(<HtmlComponentDisplay component={component} />);

    const frame = getFrame();
    const postSpy = spyOnFramePosts(frame);
    rerender(<HtmlComponentDisplay component={{ ...component, data: { rows: [] } }} />);

    expect(postSpy).not.toHaveBeenCalled();
  });

  it('re-posts THEME when the document theme changes', async () => {
    render(<HtmlComponentDisplay component={makeComponent()} />);

    const frame = getFrame();
    const postSpy = spyOnFramePosts(frame);
    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));
    postSpy.mockClear();

    act(() => {
      document.documentElement.setAttribute('data-theme', 'light');
    });

    await waitFor(() => {
      expect(postedTypes(postSpy)).toContain(CHAT_HTML_COMPONENT_MESSAGE_TYPES.THEME);
    });
    const themeCall = postSpy.mock.calls.find(
      (call) => (call[0] as { type: string }).type === CHAT_HTML_COMPONENT_MESSAGE_TYPES.THEME,
    );
    expect(themeCall?.[0]).toMatchObject({ theme: expect.objectContaining({ mode: 'light' }) });
    expect(getFrame()).toBe(frame);
  });

  it('toggles the source view and the expanded layout from the header actions', () => {
    render(<HtmlComponentDisplay component={makeComponent()} />);

    const frameBefore = getFrame();
    const canvas = document.querySelector('.html-component-canvas') as HTMLElement;
    expect(document.querySelector('.html-component-source')).toBeNull();
    expect(canvas.hidden).toBe(false);

    // Code mode: CodeMirror shows the raw markup, the canvas is hidden but stays mounted.
    fireEvent.click(screen.getByRole('button', { name: 'View source' }));
    const source = document.querySelector('.html-component-source');
    expect(source).not.toBeNull();
    expect(source?.querySelector('.cm-editor')).not.toBeNull();
    expect(source?.querySelector('.cm-content')?.textContent).toContain('id="map"');
    expect(source?.querySelector('#map')).toBeNull();
    expect(canvas.hidden).toBe(true);
    expect(getFrame()).toBe(frameBefore);

    // Back to canvas mode without remounting the iframe.
    fireEvent.click(screen.getByRole('button', { name: 'Show component' }));
    expect(document.querySelector('.html-component-source')).toBeNull();
    expect(canvas.hidden).toBe(false);
    expect(getFrame()).toBe(frameBefore);

    const wrapper = document.querySelector('.html-component-with-anchor');
    const container = document.querySelector('.html-component-container');
    expect(wrapper?.classList.contains('html-component-with-anchor-expanded')).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Expand component' }));
    expect(wrapper?.classList.contains('html-component-with-anchor-expanded')).toBe(true);
    expect(container?.classList.contains('html-component-container-expanded')).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse component' }));
    expect(wrapper?.classList.contains('html-component-with-anchor-expanded')).toBe(false);
  });

  it('renders the injected anchor and description nodes in the documented positions', () => {
    render(
      <HtmlComponentDisplay
        component={makeComponent()}
        descriptionNode={<span>Shipments, last 30 days</span>}
        anchor={<div className="viz-version-anchor">anchor</div>}
      />,
    );

    const container = document.querySelector('.html-component-container');
    // The description sits in the header, directly beneath the title.
    const heading = container?.querySelector('.html-component-header .html-component-heading');
    expect(heading).not.toBeNull();
    const title = heading?.querySelector('.html-component-title');
    const description = heading?.querySelector('.html-component-description');
    expect(description?.textContent).toBe('Shipments, last 30 days');
    expect(title?.nextElementSibling).toBe(description);
    expect(
      container?.querySelector('.html-component-source ~ .html-component-description'),
    ).toBeNull();
    const anchor = document.querySelector('.html-component-with-anchor > .viz-version-anchor');
    expect(anchor).not.toBeNull();
    expect(container?.contains(anchor)).toBe(false);
  });

  it('surfaces a soft warning when the frame loads but never signals ready', () => {
    vi.useFakeTimers();
    render(<HtmlComponentDisplay component={makeComponent()} />);

    const frame = getFrame();
    fireEvent.load(frame);
    expect(document.querySelector('.html-component-ready-warning')).toBeNull();

    act(() => {
      vi.advanceTimersByTime(HTML_COMPONENT_READY_TIMEOUT_MS);
    });
    expect(document.querySelector('.html-component-ready-warning')).not.toBeNull();
    expect(document.querySelector('.html-component-status-overlay')).not.toBeNull();
    expect(document.querySelector('.html-component-error')).toBeNull();

    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));
    expect(document.querySelector('.html-component-ready-warning')).toBeNull();
    expect(document.querySelector('.html-component-status-overlay')).toBeNull();
  });

  it('rebuilds the frame when the html changes', () => {
    const component = makeComponent();
    const { rerender } = render(<HtmlComponentDisplay component={component} />);
    const frame = getFrame();
    dispatchBridgeMessage(frame, bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.READY));
    expect(document.querySelector('.html-component-status-overlay')).toBeNull();

    rerender(
      <HtmlComponentDisplay
        component={{ ...component, html: '<html><body><p>v2</p></body></html>' }}
      />,
    );

    const nextFrame = getFrame();
    expect(nextFrame).not.toBe(frame);
    expect(nextFrame.getAttribute('srcdoc')).toContain('<p>v2</p>');
    expect(document.querySelector('.html-component-status-overlay')).not.toBeNull();
  });

  it('hides the runtime error banner while the source view is open', () => {
    render(<HtmlComponentDisplay component={makeComponent()} />);
    dispatchBridgeMessage(
      getFrame(),
      bridgeMessage(CHAT_HTML_COMPONENT_MESSAGE_TYPES.ERROR, { message: 'boom' }),
    );
    expect(document.querySelector('.html-component-error')).not.toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'View source' }));
    expect(document.querySelector('.html-component-error')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Show component' }));
    expect(document.querySelector('.html-component-error')).not.toBeNull();
  });

  it('keeps the source read-only without onSaveHtml', () => {
    render(<HtmlComponentDisplay component={makeComponent()} />);
    fireEvent.click(screen.getByRole('button', { name: 'View source' }));

    expect(screen.queryByRole('button', { name: 'Undo' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Redo' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Save changes' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Discard changes' })).toBeNull();
    const content = document.querySelector('.html-component-source .cm-content') as HTMLElement;
    expect(content.getAttribute('contenteditable')).toBe('false');
  });

  it('enables Save/Discard once the markup differs and undo/redo as history allows', async () => {
    const onSaveHtml = vi.fn().mockResolvedValue(undefined);
    render(<HtmlComponentDisplay component={makeComponent()} onSaveHtml={onSaveHtml} />);
    fireEvent.click(screen.getByRole('button', { name: 'View source' }));

    const editorDom = document.querySelector('.html-component-source .cm-editor') as HTMLElement;
    const view = EditorView.findFromDOM(editorDom) as EditorView;
    expect(view).not.toBeNull();
    const undoButton = () => screen.getByRole('button', { name: 'Undo' }) as HTMLButtonElement;
    const redoButton = () => screen.getByRole('button', { name: 'Redo' }) as HTMLButtonElement;
    const saveButton = () =>
      screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement;
    const discardButton = () =>
      screen.getByRole('button', { name: 'Discard changes' }) as HTMLButtonElement;
    const replaceAll = (insert: string) => {
      act(() => {
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert } });
      });
    };

    // Pristine: every control is visible but nothing is actionable yet.
    expect(saveButton().disabled).toBe(true);
    expect(discardButton().disabled).toBe(true);
    expect(undoButton().disabled).toBe(true);
    expect(redoButton().disabled).toBe(true);

    replaceAll('<div id="draft">draft</div>');
    await waitFor(() => expect(saveButton().disabled).toBe(false));
    expect(discardButton().disabled).toBe(false);
    await waitFor(() => expect(undoButton().disabled).toBe(false));
    expect(redoButton().disabled).toBe(true);

    // Undo restores the persisted markup and disables save/discard; redo brings the draft back.
    fireEvent.click(undoButton());
    await waitFor(() => expect(view.state.doc.toString()).toBe(SAMPLE_HTML));
    await waitFor(() => expect(saveButton().disabled).toBe(true));
    await waitFor(() => expect(redoButton().disabled).toBe(false));
    expect(undoButton().disabled).toBe(true);
    fireEvent.click(redoButton());
    await waitFor(() => expect(view.state.doc.toString()).toBe('<div id="draft">draft</div>'));
    await waitFor(() => expect(saveButton().disabled).toBe(false));
    expect(redoButton().disabled).toBe(true);

    // Discard returns to the persisted markup.
    fireEvent.click(discardButton());
    await waitFor(() => expect(saveButton().disabled).toBe(true));
    await waitFor(() => expect(view.state.doc.toString()).toBe(SAMPLE_HTML));

    // Save hands the draft to the callback and returns to the canvas.
    replaceAll('<div id="edited">edited</div>');
    await waitFor(() => expect(saveButton().disabled).toBe(false));
    fireEvent.click(saveButton());
    await waitFor(() => expect(onSaveHtml).toHaveBeenCalledWith('<div id="edited">edited</div>'));
    await waitFor(() => expect(document.querySelector('.html-component-source')).toBeNull());
    expect((document.querySelector('.html-component-canvas') as HTMLElement).hidden).toBe(false);
  });

  it('shows the save error and stays in code mode when saving fails', async () => {
    const onSaveHtml = vi.fn().mockRejectedValue(new Error('nope'));
    render(<HtmlComponentDisplay component={makeComponent()} onSaveHtml={onSaveHtml} />);
    fireEvent.click(screen.getByRole('button', { name: 'View source' }));

    const view = EditorView.findFromDOM(
      document.querySelector('.html-component-source .cm-editor') as HTMLElement,
    ) as EditorView;
    act(() => {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: '<p>x</p>' } });
    });
    const saveButton = screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement;
    await waitFor(() => expect(saveButton.disabled).toBe(false));
    fireEvent.click(saveButton);

    await waitFor(() =>
      expect(document.querySelector('.html-component-save-error')?.textContent).toBe('nope'),
    );
    expect(document.querySelector('.html-component-source')).not.toBeNull();
  });
});
