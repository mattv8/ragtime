export default String.raw`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Userspace preview unreachable</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #222; }
    h1 { font-size: 1.25rem; margin-bottom: .5rem; }
    code { background: #f3f3f3; padding: 2px 6px; border-radius: 3px; }
    .hint { background: #fff7e6; border-left: 4px solid #f0ad4e; padding: .75rem 1rem; margin-top: 1rem; }
    p { line-height: 1.5; }
  </style>
</head>
<body>
  <h1>Preview subdomain is not routed to Ragtime</h1>
  <p>The workspace preview host <code>__RAGTIME_PREVIEW_HOST__</code> is configured but does not reach this Ragtime instance. Ragtime refused to redirect into an upstream that will fail at the reverse proxy.</p>
  <p><strong>Workspace:</strong> <code>__RAGTIME_WORKSPACE_ID__</code><br><strong>Preview origin:</strong> <code>__RAGTIME_PREVIEW_ORIGIN__</code></p>
  <div class="hint">
    <p><strong>Fix:</strong> configure a wildcard vhost <code>*.__RAGTIME_PREVIEW_BASE_DOMAIN__</code> on your reverse proxy (Caddy/Traefik/nginx) that forwards to the Ragtime container, or point <code>USERSPACE_PREVIEW_BASE_DOMAIN</code> at a domain whose wildcard already proxies to Ragtime. Verify with:<br><code>curl -I https://__RAGTIME_PREVIEW_HOST__/__ragtime/preview-host-probe</code></p>
  </div>
</body>
</html>`;
