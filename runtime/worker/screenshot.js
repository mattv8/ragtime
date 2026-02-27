// Playwright screenshot capture script for ragtime runtime worker.
// Invoked by worker/service.py via: node -e <this_script> <args...>

const targetUrl = process.argv[1];
const outputPath = process.argv[2];
const viewportWidth = Number(process.argv[3] || 1440);
const viewportHeight = Number(process.argv[4] || 900);
const captureFullPage = (process.argv[5] || 'true') === 'true';
const timeoutMs = Number(process.argv[6] || 25000);
const waitForSelector = process.argv[7] || '';
const waitAfterLoadMs = Number(process.argv[8] || 900);
const refreshBeforeCapture = (process.argv[9] || 'true') === 'true';
const maxPixels = Number(process.argv[10] || 1440000);

let playwright;
try {
    playwright = require('playwright');
} catch (_) {
    process.stderr.write('Playwright package is not installed in runtime container.');
    process.exit(1);
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

        if (waitAfterLoadMs > 0) {
            await page.waitForTimeout(waitAfterLoadMs);
        }

        const screenshotOptions = {
            path: outputPath,
            animations: 'disabled',
        };

        let effectiveWidth = viewportWidth;
        let effectiveHeight = viewportHeight;
        let effectiveFullPage = captureFullPage;

        if (captureFullPage) {
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
        } else if (viewportWidth * viewportHeight > maxPixels) {
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
