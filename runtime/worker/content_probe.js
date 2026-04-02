// Playwright content probe script for ragtime runtime worker.
// Lightweight alternative to screenshot.js -- loads the page in a headless
// browser, waits for JS to settle, and returns content metrics (text length,
// console errors, etc.) WITHOUT capturing an image.
//
// Invoked by worker/service.py via: node -e <this_script> <args...>

const targetUrl = process.argv[1];
const timeoutMs = Number(process.argv[2] || 15000);
const waitAfterLoadMs = Number(process.argv[3] || 2000);
const injectMockContext = process.argv[4] === 'true';

let playwright;
try {
    playwright = require('playwright');
} catch (_) {
    process.stderr.write('Playwright package is not installed in runtime container.');
    process.exit(1);
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

async function run() {
    const consoleErrors = [];

    const browser = await playwright.chromium.launch({
        headless: true,
        args: ['--disable-dev-shm-usage'],
    });

    try {
        const context = await browser.newContext({
            viewport: { width: 1280, height: 720 },
            deviceScaleFactor: 1,
        });
        const page = await context.newPage();
        page.setDefaultTimeout(timeoutMs);

        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                consoleErrors.push(msg.text().slice(0, 300));
            }
        });

        page.on('pageerror', (err) => {
            consoleErrors.push(String(err).slice(0, 300));
        });

        // When injectMockContext is enabled, provide a stub
        // window.__ragtime_context so that dashboards relying on live data
        // (context.components[id].execute()) still exercise their rendering
        // code paths.  Without this, data-dependent code never runs during the
        // probe and runtime errors in render logic go undetected.
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

            // Check for common error page indicators
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
                title: title,
                has_error_indicator: hasErrorIndicator,
            };
        });

        const output = {
            ok: true,
            status_code: initialResponse ? initialResponse.status() : null,
            body_text_length: metrics.body_text_length,
            body_text_preview: metrics.body_text_preview,
            body_html_length: metrics.body_html_length,
            title: metrics.title,
            has_error_indicator: metrics.has_error_indicator,
            console_errors: consoleErrors.slice(0, 5),
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
