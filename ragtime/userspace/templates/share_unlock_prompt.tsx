export default String.raw`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Shared Workspace</title>
  <style>
    body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; background: #0f172a; color: #e2e8f0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    form { width: min(92vw, 360px); padding: 20px; border: 1px solid #334155; border-radius: 12px; background: #111827; }
    h1 { font-size: 18px; margin: 0 0 10px 0; }
    .subtitle { margin: 0 0 6px 0; color: #e2e8f0; font-size: 15px; font-weight: 600; }
    .owner { margin: 0 0 14px 0; color: #94a3b8; font-size: 13px; }
    .error { color: #fca5a5; margin: 0 0 12px 0; font-size: 14px; }
    label { display: block; margin-bottom: 8px; font-size: 13px; }
    input[type="password"] { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px; border: 1px solid #334155; background: #0b1220; color: #e2e8f0; }
    button { margin-top: 12px; width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #334155; background: #1d4ed8; color: #fff; cursor: pointer; }
  </style>
</head>
<body>
  <form method="post" action="__RAGTIME_FORM_ACTION__">
    <h1>__RAGTIME_TITLE__</h1>
    __RAGTIME_SUBTITLE_BLOCK__
    __RAGTIME_OWNER_BLOCK__
    __RAGTIME_ERROR_BLOCK__
    __RAGTIME_NEXT_BLOCK__
    <label for="share_password">Password</label>
    <input id="share_password" name="share_password" type="password" required autofocus autocomplete="current-password">
    <button type="submit">Continue</button>
  </form>
</body>
</html>`;
