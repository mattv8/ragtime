const readline = require('readline');
const dns = require('dns');
const { promisify } = require('util');
const lookupAsync = promisify(dns.lookup);

let playwright;
let playwrightLoadError = '';
try {
  playwright = require('playwright');
} catch (error) {
  playwright = null;
  playwrightLoadError = error && error.message
    ? String(error.message)
    : 'Playwright package is not installed in runtime container.';
}

let browser = null;

function writeMessage(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function makeError(message, code = 'operation_failed') {
  const error = new Error(message);
  error.code = code;
  return error;
}

async function ensureBrowser() {
  if (!playwright) {
    throw makeError(playwrightLoadError, 'playwright_missing');
  }
  if (browser && browser.isConnected()) {
    return browser;
  }
  if (browser) {
    try {
      await browser.close();
    } catch (_) {
      // ignore stale browser close failures
    }
  }
  browser = await playwright.chromium.launch({
    headless: true,
    args: ['--disable-dev-shm-usage'],
  });
  return browser;
}

async function waitForStableSignature(page, minimumWaitMs, readSignature) {
  const baseline = Math.max(0, Number(minimumWaitMs) || 0);
  const maxExtraWaitMs = 2500;
  const stableWindowMs = 650;
  const pollEveryMs = 250;
  const startedAt = Date.now();
  let stableSince = 0;
  let previousSignature = '';

  while (Date.now() - startedAt <= baseline + maxExtraWaitMs) {
    const elapsed = Date.now() - startedAt;
    if (elapsed < baseline) {
      await page.waitForTimeout(Math.min(pollEveryMs, baseline - elapsed));
      continue;
    }

    const signature = await readSignature().catch(() => '');
    if (!signature) {
      await page.waitForTimeout(pollEveryMs);
      continue;
    }

    if (signature === previousSignature) {
      if (!stableSince) {
        stableSince = Date.now();
      }
      if (Date.now() - stableSince >= stableWindowMs) {
        return;
      }
    } else {
      previousSignature = signature;
      stableSince = 0;
    }

    await page.waitForTimeout(pollEveryMs);
  }
}

async function waitForHmrSettle(page, minimumWaitMs) {
  await waitForStableSignature(page, minimumWaitMs, async () => {
    return await page.evaluate(() => {
      const title = document.title || '';
      const ready = document.readyState || '';
      const href = String(location.href || '');
      const body = document.body;
      const root = document.documentElement;
      const bodyChars = body && body.innerText ? body.innerText.length : 0;
      const bodyHtml = body && body.innerHTML ? body.innerHTML.length : 0;
      const rootHtml = root && root.innerHTML ? root.innerHTML.length : 0;
      return [title, ready, href, bodyChars, bodyHtml, rootHtml].join('|');
    });
  });
}

async function waitForSettle(page, minimumWaitMs) {
  await waitForStableSignature(page, minimumWaitMs, async () => {
    return await page.evaluate(() => {
      const body = document.body;
      const bodyChars = body && body.innerText ? body.innerText.length : 0;
      const bodyHtml = body && body.innerHTML ? body.innerHTML.length : 0;
      return [bodyChars, bodyHtml].join('|');
    });
  });
}

function attachConsoleErrorCapture(page) {
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text().slice(0, 300));
    }
  });
  page.on('pageerror', (err) => {
    consoleErrors.push(String(err).slice(0, 300));
  });
  return consoleErrors;
}

function isPrivateOrLocalHostname(hostname) {
  const host = String(hostname || '').trim().toLowerCase().replace(/^\[|\]$/g, '');
  if (!host) return true;
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return true;
  if (host === '::1' || host.startsWith('fe80:') || host.startsWith('fc') || host.startsWith('fd')) return true;
  const parts = host.split('.').map((part) => Number(part));
  if (parts.length === 4 && parts.every((part) => Number.isInteger(part) && part >= 0 && part <= 255)) {
    const [a, b] = parts;
    return (
      a === 0 ||
      a === 10 ||
      a === 127 ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168)
    );
  }
  return false;
}

async function isPrivateOrLocalIp(hostname) {
  if (isPrivateOrLocalHostname(hostname)) return true;
  try {
    const { address } = await lookupAsync(hostname);
    return isPrivateOrLocalHostname(address);
  } catch (err) {
    return true; // fail closed if DNS lookup fails
  }
}

function parseUrl(value) {
  try {
    return new URL(String(value || ''));
  } catch (_) {
    return null;
  }
}

function isSameOrigin(urlValue, originValue) {
  const url = parseUrl(urlValue);
  const origin = parseUrl(originValue);
  return Boolean(url && origin && url.origin === origin.origin);
}

async function resolveDebugTargetUrl(value, baseUrl, allowExternalNavigation) {
  const raw = String(value || '').trim();
  if (!raw) return baseUrl;
  let resolved;
  try {
    resolved = new URL(raw, baseUrl).toString();
  } catch (_) {
    throw makeError(`Invalid navigation target: ${raw}`, 'bad_request');
  }
  const parsed = parseUrl(resolved);
  if (!parsed || !/^https?:$/i.test(parsed.protocol)) {
    throw makeError('Only http/https navigation targets are allowed.', 'bad_request');
  }
  if (!isSameOrigin(resolved, baseUrl)) {
    const isPrivate = await isPrivateOrLocalIp(parsed.hostname);
    if (isPrivate) {
      throw makeError(`Blocked private/local navigation target: ${parsed.hostname}`, 'blocked_url');
    }
    if (!allowExternalNavigation) {
      throw makeError('External navigation is disabled for this debug run.', 'blocked_url');
    }
  }
  return resolved;
}

function normalizeDebugSteps(value) {
  if (!Array.isArray(value)) {
    throw makeError('Debug steps must be an array.', 'bad_request');
  }
  if (value.length > 25) {
    throw makeError('Debug runs are limited to 25 steps.', 'bad_request');
  }
  return value.map((step, index) => {
    if (!step || typeof step !== 'object' || Array.isArray(step)) {
      throw makeError(`Debug step ${index + 1} must be an object.`, 'bad_request');
    }
    return step;
  });
}

async function installDebugRequestGuards(page, baseUrl, allowExternalNavigation) {
  await page.route('**/*', async (route) => {
    const request = route.request();
    const requestUrl = request.url();
    const parsed = parseUrl(requestUrl);
    if (parsed && !isSameOrigin(requestUrl, baseUrl)) {
      const isPrivate = await isPrivateOrLocalIp(parsed.hostname);
      if (isPrivate) {
        await route.abort('blockedbyclient').catch(() => null);
        return;
      }
      if (!allowExternalNavigation && request.isNavigationRequest()) {
        await route.abort('blockedbyclient').catch(() => null);
        return;
      }
    }
    await route.continue().catch(() => null);
  });
}

async function runDebugSteps(request) {
  const activeBrowser = await ensureBrowser();
  const baseUrl = String(request.url || '');
  if (!baseUrl) {
    throw makeError('Debug run request is missing preview URL.', 'bad_request');
  }
  const allowExternalNavigation = Boolean(request.allow_external_navigation);
  const steps = normalizeDebugSteps(request.steps || []);
  const timeoutMs = Math.max(1000, Math.min(60000, Number(request.timeout_ms || 25000)));
  const context = await activeBrowser.newContext({
    viewport: {
      width: Math.max(320, Math.min(1920, Number(request.viewport_width || 1280))),
      height: Math.max(240, Math.min(1600, Number(request.viewport_height || 900))),
    },
    deviceScaleFactor: 1,
  });

  const stepResults = [];
  const consoleErrors = [];

  try {
    const page = await context.newPage();
    page.setDefaultTimeout(timeoutMs);
    page.on('console', (msg) => {
      if (['error', 'warning'].includes(msg.type())) {
        consoleErrors.push(`${msg.type()}: ${msg.text()}`.slice(0, 300));
      }
    });
    page.on('pageerror', (err) => consoleErrors.push(String(err).slice(0, 300)));
    await installDebugRequestGuards(page, baseUrl, allowExternalNavigation);

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await page.waitForLoadState('networkidle', { timeout: Math.min(timeoutMs, 8000) }).catch(() => null);

    for (let i = 0; i < steps.length; i += 1) {
      const step = steps[i];
      const action = String(step.action || '').trim().toLowerCase();
      const startedAt = Date.now();
      const result = { index: i, action, ok: true };
      try {
        if (action === 'goto') {
          const target = await resolveDebugTargetUrl(step.url || step.path || '', baseUrl, allowExternalNavigation);
          const response = await page.goto(target, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
          await page.waitForLoadState('networkidle', { timeout: Math.min(timeoutMs, 8000) }).catch(() => null);
          result.url = page.url();
          result.status_code = response ? response.status() : null;
        } else if (action === 'click') {
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('click requires selector.', 'bad_request');
          await page.locator(selector).first().click({ timeout: Math.min(timeoutMs, Number(step.timeout_ms || timeoutMs)) });
        } else if (action === 'fill') {
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('fill requires selector.', 'bad_request');
          await page.locator(selector).first().fill(String(step.value || '').slice(0, 5000));
        } else if (action === 'type') {
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('type requires selector.', 'bad_request');
          await page.locator(selector).first().pressSequentially(String(step.text || step.value || '').slice(0, 5000), { delay: Math.max(0, Math.min(200, Number(step.delay_ms || 0))) });
        } else if (action === 'press') {
          const key = String(step.key || '').slice(0, 80);
          if (!key) throw makeError('press requires key.', 'bad_request');
          const selector = String(step.selector || '').slice(0, 500);
          if (selector) await page.locator(selector).first().press(key);
          else await page.keyboard.press(key);
        } else if (action === 'select_option') {
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('select_option requires selector.', 'bad_request');
          await page.locator(selector).first().selectOption(String(step.value || '').slice(0, 500));
        } else if (action === 'wait_for_selector') {
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('wait_for_selector requires selector.', 'bad_request');
          await page.waitForSelector(selector, { state: String(step.state || 'visible'), timeout: Math.min(timeoutMs, Number(step.timeout_ms || timeoutMs)) });
        } else if (action === 'wait_for_timeout') {
          await page.waitForTimeout(Math.max(0, Math.min(5000, Number(step.ms || step.timeout_ms || 1000))));
        } else if (action === 'query') {
          // Safe, structured DOM inspection. Intentionally does NOT evaluate
          // agent-supplied JavaScript; it only reads element state through the
          // Playwright locator API so the agent cannot run arbitrary scripts
          // inside the workspace devserver origin.
          const selector = String(step.selector || '').slice(0, 500);
          if (!selector) throw makeError('query requires selector.', 'bad_request');
          const locator = page.locator(selector);
          const count = await locator.count();
          result.count = count;
          if (count > 0) {
            const first = locator.first();
            result.visible = await first.isVisible().catch(() => false);
            const textContent = await first.textContent().catch(() => null);
            result.text = textContent ? String(textContent).trim().slice(0, 2000) : '';
            const requestedAttributes = Array.isArray(step.attributes)
              ? step.attributes.slice(0, 12)
              : [];
            if (requestedAttributes.length > 0) {
              const attributes = {};
              for (const attrName of requestedAttributes) {
                const name = String(attrName || '').slice(0, 80);
                if (!name) continue;
                attributes[name] = await first.getAttribute(name).catch(() => null);
              }
              result.attributes = attributes;
            }
          } else {
            result.visible = false;
            result.text = '';
          }
        } else if (action === 'content') {
          const metrics = await page.evaluate(() => {
            const bodyText = document.body && document.body.innerText ? document.body.innerText.trim() : '';
            return {
              title: document.title || '',
              url: location.href,
              body_text_preview: bodyText.slice(0, 1000),
              body_text_length: bodyText.length,
              body_html_length: document.body && document.body.innerHTML ? document.body.innerHTML.length : 0,
            };
          });
          Object.assign(result, metrics);
        } else if (action === 'screenshot') {
          const outputPath = String(step.output_path || '');
          if (!outputPath) throw makeError('screenshot step requires output_path.', 'bad_request');
          await page.screenshot({ path: outputPath, fullPage: Boolean(step.full_page ?? true), animations: 'disabled' });
          result.output_path = outputPath;
        } else {
          throw makeError(`Unsupported debug action: ${action}`, 'bad_request');
        }
      } catch (error) {
        result.ok = false;
        result.error = error && error.message ? String(error.message) : String(error);
        stepResults.push(result);
        if (step.stop_on_error !== false) break;
        continue;
      } finally {
        result.elapsed_ms = Date.now() - startedAt;
      }
      stepResults.push(result);
    }

    return {
      ok: stepResults.every((step) => step.ok),
      url: page.url(),
      title: await page.title().catch(() => ''),
      steps: stepResults,
      console_errors: consoleErrors.slice(0, 20),
    };
  } finally {
    await context.close().catch(() => null);
  }
}

async function runScreenshot(request) {
  const activeBrowser = await ensureBrowser();
  const context = await activeBrowser.newContext({
    viewport: {
      width: Number(request.viewport_width || 1440),
      height: Number(request.viewport_height || 900),
    },
    deviceScaleFactor: 1,
  });

  try {
    const page = await context.newPage();
    const timeoutMs = Number(request.timeout_ms || 25000);
    const waitAfterLoadMs = Number(request.wait_after_load_ms || 1800);
    const captureElement = Boolean(request.capture_element);
    const waitForSelector = String(request.wait_for_selector || '');
    const clipPaddingPx = Number(request.clip_padding_px || 16);
    const refreshBeforeCapture = Boolean(request.refresh_before_capture);
    const maxPixels = Number(request.max_pixels || 1440000);
    const targetUrl = String(request.url || '');
    const outputPath = String(request.output_path || '');
    const viewportWidth = Number(request.viewport_width || 1440);
    const viewportHeight = Number(request.viewport_height || 900);
    const captureFullPage = Boolean(request.capture_full_page);

    if (!targetUrl || !outputPath) {
      throw makeError('Screenshot request is missing target URL or output path.');
    }

    page.setDefaultTimeout(timeoutMs);

    const initialResponse = await page.goto(targetUrl, {
      waitUntil: 'domcontentloaded',
      timeout: timeoutMs,
    });

    await page.waitForLoadState('networkidle', {
      timeout: Math.min(timeoutMs, 8000),
    }).catch(() => null);

    if (refreshBeforeCapture) {
      await page.reload({
        waitUntil: 'domcontentloaded',
        timeout: timeoutMs,
      });
      await page.waitForLoadState('networkidle', {
        timeout: Math.min(timeoutMs, 8000),
      }).catch(() => null);
    }

    if (waitForSelector) {
      await page.waitForSelector(waitForSelector, {
        state: 'visible',
        timeout: Math.min(timeoutMs, 10000),
      }).catch(() => null);
    }

    const settleFloorMs = refreshBeforeCapture ? 1800 : 900;
    const settleWaitMs = Math.max(waitAfterLoadMs, settleFloorMs);
    await waitForHmrSettle(page, settleWaitMs);

    const screenshotOptions = {
      path: outputPath,
      animations: 'disabled',
    };

    let effectiveWidth = viewportWidth;
    let effectiveHeight = viewportHeight;
    let effectiveFullPage = captureFullPage;
    let elementMatchCount = 0;
    let elementVisibleCount = 0;
    let elementClipUsed = false;
    let elementBounds = null;

    if (captureElement) {
      if (!waitForSelector) {
        throw makeError('capture_element=true requires wait_for_selector');
      }

      const matches = page.locator(waitForSelector);
      elementMatchCount = await matches.count();
      if (elementMatchCount < 1) {
        throw makeError(
          `Element capture selector matched no elements: ${waitForSelector}`
        );
      }

      const visibleIndexes = [];
      for (let i = 0; i < elementMatchCount; i += 1) {
        const candidate = matches.nth(i);
        const isVisible = await candidate.isVisible().catch(() => false);
        if (isVisible) {
          visibleIndexes.push(i);
        }
      }

      elementVisibleCount = visibleIndexes.length;
      if (elementVisibleCount !== 1) {
        throw makeError(
          `Element capture selector must match exactly one visible element: ${waitForSelector} (visible=${elementVisibleCount}, total=${elementMatchCount})`
        );
      }

      const target = matches.nth(visibleIndexes[0]);
      await target.scrollIntoViewIfNeeded().catch(() => null);
      const rawBounds = await target.boundingBox();
      if (!rawBounds || rawBounds.width <= 0 || rawBounds.height <= 0) {
        throw makeError(
          `Element capture target has invalid bounds: ${waitForSelector}`
        );
      }

      const pad = Math.max(0, Math.floor(Number(clipPaddingPx) || 0));
      const pageMetrics = await page.evaluate(() => {
        const body = document.body;
        const doc = document.documentElement;
        const pageWidth = Math.max(
          body ? body.scrollWidth : 0,
          doc ? doc.scrollWidth : 0,
          window.innerWidth || 0
        );
        const pageHeight = Math.max(
          body ? body.scrollHeight : 0,
          doc ? doc.scrollHeight : 0,
          window.innerHeight || 0
        );
        return { pageWidth, pageHeight };
      });

      const maxPageWidth = Math.max(1, Number(pageMetrics.pageWidth) || viewportWidth);
      const maxPageHeight = Math.max(1, Number(pageMetrics.pageHeight) || viewportHeight);

      let clipX = Math.max(0, Math.floor(rawBounds.x - pad));
      let clipY = Math.max(0, Math.floor(rawBounds.y - pad));
      let clipRight = Math.min(maxPageWidth, Math.ceil(rawBounds.x + rawBounds.width + pad));
      let clipBottom = Math.min(maxPageHeight, Math.ceil(rawBounds.y + rawBounds.height + pad));

      if (clipRight <= clipX) {
        clipRight = Math.min(maxPageWidth, clipX + 1);
      }
      if (clipBottom <= clipY) {
        clipBottom = Math.min(maxPageHeight, clipY + 1);
      }

      let clipWidth = Math.max(1, clipRight - clipX);
      let clipHeight = Math.max(1, clipBottom - clipY);

      if (clipWidth * clipHeight > maxPixels) {
        const scale = Math.sqrt(maxPixels / (clipWidth * clipHeight));
        clipWidth = Math.max(1, Math.floor(clipWidth * scale));
        clipHeight = Math.max(1, Math.floor(clipHeight * scale));
      }

      screenshotOptions.clip = {
        x: clipX,
        y: clipY,
        width: clipWidth,
        height: clipHeight,
      };
      elementClipUsed = true;
      effectiveWidth = clipWidth;
      effectiveHeight = clipHeight;
      effectiveFullPage = false;
      elementBounds = {
        x: rawBounds.x,
        y: rawBounds.y,
        width: rawBounds.width,
        height: rawBounds.height,
        clip: screenshotOptions.clip,
        padding_px: Math.max(0, Math.floor(Number(clipPaddingPx) || 0)),
      };
    }

    if (!captureElement && captureFullPage) {
      const fullHeight = await page.evaluate(() => {
        const bodyHeight = document.body ? document.body.scrollHeight : 0;
        const docHeight = document.documentElement
          ? document.documentElement.scrollHeight
          : 0;
        return Math.max(bodyHeight, docHeight, window.innerHeight || 0);
      });

      if (viewportWidth * fullHeight <= maxPixels) {
        screenshotOptions.fullPage = true;
        effectiveHeight = fullHeight;
      } else {
        effectiveFullPage = false;
        const clipHeight = Math.max(240, Math.floor(maxPixels / Math.max(1, viewportWidth)));
        screenshotOptions.clip = {
          x: 0,
          y: 0,
          width: viewportWidth,
          height: clipHeight,
        };
        effectiveHeight = clipHeight;
      }
    } else if (!captureElement && viewportWidth * viewportHeight > maxPixels) {
      const scale = Math.sqrt(maxPixels / (viewportWidth * viewportHeight));
      const clipWidth = Math.max(320, Math.floor(viewportWidth * scale));
      const clipHeight = Math.max(240, Math.floor(viewportHeight * scale));
      screenshotOptions.clip = {
        x: 0,
        y: 0,
        width: clipWidth,
        height: clipHeight,
      };
      effectiveWidth = clipWidth;
      effectiveHeight = clipHeight;
    }

    await page.screenshot(screenshotOptions);

    const title = await page.title().catch(() => '');
    const htmlLength = await page
      .content()
      .then((html) => html.length)
      .catch(() => null);

    return {
      ok: true,
      status_code: initialResponse ? initialResponse.status() : null,
      title,
      html_length: htmlLength,
      output_path: outputPath,
      screenshot_url: targetUrl,
      effective_width: effectiveWidth,
      effective_height: effectiveHeight,
      effective_full_page: effectiveFullPage,
      wait_after_load_ms: waitAfterLoadMs,
      effective_wait_after_load_ms: settleWaitMs,
      capture_element: captureElement,
      element_selector: waitForSelector || null,
      element_match_count: elementMatchCount,
      element_visible_count: elementVisibleCount,
      element_clip_used: elementClipUsed,
      element_bounds: elementBounds,
    };
  } finally {
    await context.close().catch(() => null);
  }
}

async function runContentProbe(request) {
  const activeBrowser = await ensureBrowser();
  const context = await activeBrowser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1,
  });

  try {
    const page = await context.newPage();
    const targetUrl = String(request.url || '');
    const timeoutMs = Number(request.timeout_ms || 15000);
    const waitAfterLoadMs = Number(request.wait_after_load_ms || 2000);
    const injectMockContext = Boolean(request.inject_mock_context);
    const consoleErrors = attachConsoleErrorCapture(page);

    if (!targetUrl) {
      throw makeError('Content probe request is missing target URL.');
    }

    page.setDefaultTimeout(timeoutMs);

    if (injectMockContext) {
      await page.addInitScript(() => {
        window.__ragtime_context = Object.freeze({
          components: Object.freeze(new Proxy({}, {
            get: function (_, prop) {
              if (typeof prop !== 'string') return undefined;
              return Object.freeze({
                component_id: prop,
                execute: function () {
                  return Promise.resolve({
                    rows: [],
                    columns: [],
                    row_count: 0,
                  });
                },
              });
            },
            has: function () { return true; },
          })),
        });
        if (!window.context) window.context = window.__ragtime_context;
      });
    }

    const initialResponse = await page.goto(targetUrl, {
      waitUntil: 'domcontentloaded',
      timeout: timeoutMs,
    });

    await page.waitForLoadState('networkidle', {
      timeout: Math.min(timeoutMs, 8000),
    }).catch(() => null);

    await waitForSettle(page, waitAfterLoadMs);

    const metrics = await page.evaluate(() => {
      const body = document.body;
      const bodyText = body && body.innerText ? body.innerText.trim() : '';
      const bodyTextLength = bodyText.length;
      const bodyHtmlLength = body && body.innerHTML ? body.innerHTML.length : 0;
      const title = document.title || '';
      const lowerText = bodyText.toLowerCase();
      const hasErrorIndicator =
        lowerText.includes('cannot get') ||
        lowerText.includes('not found') ||
        lowerText.includes('module not found') ||
        lowerText.includes('failed to resolve') ||
        lowerText.includes('syntax error') ||
        lowerText.includes('unexpected token');

      return {
        body_text_length: bodyTextLength,
        body_text_preview: bodyText.slice(0, 200),
        body_html_length: bodyHtmlLength,
        title,
        has_error_indicator: hasErrorIndicator,
      };
    });

    return {
      ok: true,
      status_code: initialResponse ? initialResponse.status() : null,
      body_text_length: metrics.body_text_length,
      body_text_preview: metrics.body_text_preview,
      body_html_length: metrics.body_html_length,
      title: metrics.title,
      has_error_indicator: metrics.has_error_indicator,
      console_errors: consoleErrors.slice(0, 5),
    };
  } finally {
    await context.close().catch(() => null);
  }
}

async function runExternalBrowse(request) {
  const activeBrowser = await ensureBrowser();
  const contextOptions = {
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
  };
  const userAgent = String(request.user_agent || '').trim();
  if (userAgent) {
    contextOptions.userAgent = userAgent;
  }
  const context = await activeBrowser.newContext(contextOptions);

  try {
    const page = await context.newPage();
    const targetUrl = String(request.url || '');
    const timeoutMs = Number(request.timeout_ms || 20000);
    const waitAfterLoadMs = Number(request.wait_after_load_ms || 1500);
    const extractLinks = Boolean(request.extract_links);
    const maxTextChars = Math.max(200, Math.min(20000, Number(request.max_text_chars || 4000)));
    const maxLinks = Math.max(0, Math.min(100, Number(request.max_links || 20)));
    const consoleErrors = attachConsoleErrorCapture(page);

    if (!targetUrl) {
      throw makeError('External browse request is missing target URL.');
    }

    page.setDefaultTimeout(timeoutMs);

    let response = null;
    try {
      response = await page.goto(targetUrl, {
        waitUntil: 'domcontentloaded',
        timeout: timeoutMs,
      });
    } catch (error) {
      throw makeError(
        `Navigation failed: ${error && error.message ? error.message : String(error)}`,
        'navigation_failed'
      );
    }

    await page.waitForLoadState('networkidle', {
      timeout: Math.min(timeoutMs, 8000),
    }).catch(() => null);

    await waitForSettle(page, waitAfterLoadMs);

    const extracted = await page.evaluate(({ withLinks, linkCap }) => {
      const body = document.body;
      const fullText = body && body.innerText ? body.innerText.trim() : '';
      const title = document.title || '';
      let links = [];
      if (withLinks) {
        const seen = new Set();
        const anchors = Array.from(document.querySelectorAll('a[href]'));
        for (const anchor of anchors) {
          if (links.length >= linkCap) break;
          const href = anchor.getAttribute('href') || '';
          let absolute = '';
          try {
            absolute = new URL(href, document.baseURI).toString();
          } catch (_) {
            continue;
          }
          if (!/^https?:/i.test(absolute)) continue;
          if (seen.has(absolute)) continue;
          seen.add(absolute);
          const text = (anchor.innerText || anchor.textContent || '').trim().slice(0, 200);
          links.push({ url: absolute, text });
        }
      }
      return {
        title,
        full_text: fullText,
        links,
      };
    }, { withLinks: extractLinks, linkCap: maxLinks });

    const fullText = String(extracted.full_text || '');
    const truncated = fullText.length > maxTextChars;
    const text = truncated ? fullText.slice(0, maxTextChars) : fullText;

    return {
      ok: true,
      requested_url: targetUrl,
      url: page.url(),
      status_code: response ? response.status() : null,
      title: extracted.title || '',
      text,
      text_length: fullText.length,
      truncated,
      links: Array.isArray(extracted.links) ? extracted.links : [],
      console_errors: consoleErrors.slice(0, 5),
    };
  } finally {
    await context.close().catch(() => null);
  }
}

async function handleRequest(request) {
  if (!request || typeof request !== 'object') {
    throw makeError('Playwright broker received an invalid request.', 'bad_request');
  }
  if (request.type === 'screenshot') {
    return await runScreenshot(request);
  }
  if (request.type === 'content_probe') {
    return await runContentProbe(request);
  }
  if (request.type === 'external_browse') {
    return await runExternalBrowse(request);
  }
  if (request.type === 'debug_steps') {
    return await runDebugSteps(request);
  }
  throw makeError(`Unsupported Playwright broker request type: ${String(request.type || '')}`, 'bad_request');
}

async function cleanup() {
  if (browser) {
    try {
      await browser.close();
    } catch (_) {
      // ignore close failures
    }
    browser = null;
  }
}

process.on('SIGTERM', () => {
  cleanup().finally(() => process.exit(0));
});
process.on('SIGINT', () => {
  cleanup().finally(() => process.exit(0));
});

async function main() {
  writeMessage({ type: 'ready' });
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = String(line || '').trim();
    if (!trimmed) {
      continue;
    }

    let request;
    try {
      request = JSON.parse(trimmed);
    } catch (error) {
      writeMessage({
        id: null,
        ok: false,
        code: 'bad_request',
        error: error && error.message ? String(error.message) : 'Invalid JSON request',
      });
      continue;
    }

    try {
      const result = await handleRequest(request);
      writeMessage({ id: request.id ?? null, ok: true, result });
    } catch (error) {
      const message = error && error.message ? String(error.message) : String(error);
      const code = error && error.code ? String(error.code) : 'operation_failed';
      writeMessage({ id: request.id ?? null, ok: false, code, error: message });
    }
  }

  await cleanup();
}

main().catch((error) => {
  const message = error && error.message ? String(error.message) : String(error);
  process.stderr.write(`${message}\n`);
  cleanup().finally(() => process.exit(1));
});
