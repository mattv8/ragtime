const readline = require('readline');

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

async function waitForHmrSettle(page, minimumWaitMs) {
  const baseline = Math.max(0, Number(minimumWaitMs) || 0);
  const maxExtraWaitMs = 2500;
  const stableWindowMs = 650;
  const pollEveryMs = 250;
  const startedAt = Date.now();
  let stableSince = 0;
  let previousSignature = '';

  const readSignature = async () => {
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
  };

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

async function waitForSettle(page, minimumWaitMs) {
  const baseline = Math.max(0, Number(minimumWaitMs) || 0);
  const maxExtraWaitMs = 2500;
  const stableWindowMs = 650;
  const pollEveryMs = 250;
  const startedAt = Date.now();
  let stableSince = 0;
  let previousSignature = '';

  const readSignature = async () => {
    return await page.evaluate(() => {
      const body = document.body;
      const bodyChars = body && body.innerText ? body.innerText.length : 0;
      const bodyHtml = body && body.innerHTML ? body.innerHTML.length : 0;
      return [bodyChars, bodyHtml].join('|');
    });
  };

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
      if (!stableSince) stableSince = Date.now();
      if (Date.now() - stableSince >= stableWindowMs) return;
    } else {
      previousSignature = signature;
      stableSince = 0;
    }

    await page.waitForTimeout(pollEveryMs);
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
    const consoleErrors = [];

    if (!targetUrl) {
      throw makeError('Content probe request is missing target URL.');
    }

    page.setDefaultTimeout(timeoutMs);
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text().slice(0, 300));
      }
    });
    page.on('pageerror', (err) => {
      consoleErrors.push(String(err).slice(0, 300));
    });

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