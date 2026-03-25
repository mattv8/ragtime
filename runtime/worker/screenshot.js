// Playwright screenshot capture script for ragtime runtime worker.
// Invoked by worker/service.py via: node -e <this_script> <args...>

const targetUrl = process.argv[1];
const outputPath = process.argv[2];
const viewportWidth = Number(process.argv[3] || 1440);
const viewportHeight = Number(process.argv[4] || 900);
const captureFullPage = (process.argv[5] || 'true') === 'true';
const timeoutMs = Number(process.argv[6] || 25000);
const waitForSelector = process.argv[7] || '';
const captureElement = (process.argv[8] || 'false') === 'true';
const clipPaddingPx = Number(process.argv[9] || 16);
const waitAfterLoadMs = Number(process.argv[10] || 1800);
const refreshBeforeCapture = (process.argv[11] || 'true') === 'true';
const maxPixels = Number(process.argv[12] || 1440000);

let playwright;
try {
    playwright = require('playwright');
} catch (_) {
    process.stderr.write('Playwright package is not installed in runtime container.');
    process.exit(1);
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

async function run() {
    const browser = await playwright.chromium.launch({
        headless: true,
        args: ['--disable-dev-shm-usage'],
    });

    try {
        const context = await browser.newContext({
            viewport: { width: viewportWidth, height: viewportHeight },
            deviceScaleFactor: 1,
        });
        const page = await context.newPage();
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
                throw new Error('capture_element=true requires wait_for_selector');
            }

            const matches = page.locator(waitForSelector);
            elementMatchCount = await matches.count();
            if (elementMatchCount < 1) {
                throw new Error(
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
                throw new Error(
                    `Element capture selector must match exactly one visible element: ${waitForSelector} (visible=${elementVisibleCount}, total=${elementMatchCount})`
                );
            }

            const target = matches.nth(visibleIndexes[0]);
            await target.scrollIntoViewIfNeeded().catch(() => null);
            const rawBounds = await target.boundingBox();
            if (!rawBounds || rawBounds.width <= 0 || rawBounds.height <= 0) {
                throw new Error(
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

        const output = {
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
        process.stdout.write(JSON.stringify(output));
    } finally {
        await browser.close();
    }
}

run().catch((error) => {
    const message = error && error.message ? error.message : String(error);
    process.stderr.write(message);
    process.exit(1);
});
