export default String.raw`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in to Ragtime</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { min-height: 100vh; margin: 0; display: grid; place-items: center; background: radial-gradient(circle at top left, #eef2ff, transparent 34rem), linear-gradient(135deg, #f8fafc, #e0f2fe); color: #0f172a; }
    .card { width: min(420px, calc(100vw - 32px)); padding: 28px; border-radius: 24px; background: rgba(255,255,255,.86); box-shadow: 0 24px 80px rgba(15,23,42,.18); border: 1px solid rgba(148,163,184,.35); backdrop-filter: blur(16px); }
    h1 { margin: 0 0 8px; font-size: 24px; }
    p { margin: 0 0 20px; color: #475569; }
    label { display: block; margin: 14px 0 6px; font-size: 13px; font-weight: 700; color: #334155; }
    input { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 14px; font: inherit; background: white; color: #0f172a; }
    button { width: 100%; margin-top: 18px; border: 0; border-radius: 999px; padding: 12px 16px; font: inherit; font-weight: 800; color: white; background: linear-gradient(135deg, #4f46e5, #06b6d4); cursor: pointer; }
    ul { list-style: none; padding: 0; margin: 20px 0 0; display: grid; gap: 8px; }
    li { display: flex; align-items: center; gap: 8px; color: #334155; font-size: 13px; }
    small { margin-left: auto; color: #64748b; }
    .dot { width: 8px; height: 8px; border-radius: 999px; background: #94a3b8; }
    .status-available { background: #10b981; }
    .status-unavailable, .status-not_configured { background: #f59e0b; }
    .error { margin-bottom: 14px; border: 1px solid #fecaca; background: #fef2f2; color: #991b1b; border-radius: 12px; padding: 10px 12px; font-size: 13px; }
    @media (prefers-color-scheme: dark) {
      body { background: radial-gradient(circle at top left, #1e1b4b, transparent 34rem), linear-gradient(135deg, #020617, #0f172a); color: #f8fafc; }
      .card { background: rgba(15,23,42,.84); border-color: rgba(148,163,184,.24); }
      p, li, label { color: #cbd5e1; } small { color: #94a3b8; }
      input { background: #020617; color: #f8fafc; border-color: #334155; }
    }
  </style>
</head>
<body>
  <main class="card">
    <h1>Sign in to Ragtime</h1>
    <p>__RAGTIME_METHOD_LABEL__ authorization is required to continue.</p>
    __RAGTIME_ERROR_HTML__
    <form method="post" action="/__ragtime/browser-auth/start">
      <input type="hidden" name="surfaces" value="__RAGTIME_SURFACE_VALUE__">
      <input type="hidden" name="auth_method_key" value="__RAGTIME_AUTH_METHOD_KEY__">
      <input type="hidden" name="return_to" value="__RAGTIME_RETURN_TO__">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" value="__RAGTIME_USERNAME_VALUE__" required autofocus>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" value="__RAGTIME_PASSWORD_VALUE__" required>
      <button type="submit">Continue</button>
    </form>
    <ul aria-label="Available auth methods">__RAGTIME_METHOD_ITEMS_HTML__</ul>
  </main>
</body>
</html>`;
