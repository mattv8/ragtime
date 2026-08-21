export default String.raw`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Shared Workspace</title>
  <script>
    (() => {
      const injectedDefaultThemePack = __RAGTIME_DEFAULT_THEME_PACK_JSON__;
      const root = document.documentElement;
      const allowedThemePacks = ['default', 'modern', 'serif'];
      const canonicalizeThemePack = (value) => value === 'vscode' ? 'modern' : value;
      const readValidThemePack = (value) => {
        const canonical = canonicalizeThemePack(value);
        return allowedThemePacks.includes(canonical) ? canonical : null;
      };
      const readValidColorMode = (value) => value === 'light' || value === 'dark' ? value : null;
      let storedThemePack = null;
      let storedColorMode = null;
      try {
        storedThemePack = readValidThemePack(localStorage.getItem("ragtime-theme-pack"));
        if (storedThemePack === 'modern' && localStorage.getItem("ragtime-theme-pack") === 'vscode') {
          localStorage.setItem("ragtime-theme-pack", "modern");
        }
        storedColorMode = readValidColorMode(localStorage.getItem("ragtime-theme"));
      } catch {
        storedThemePack = null;
        storedColorMode = null;
      }
      const resolvedThemePack = storedThemePack || readValidThemePack(injectedDefaultThemePack) || 'default';
      if (resolvedThemePack === 'default') {
        root.removeAttribute('data-theme-pack');
      } else {
        root.setAttribute('data-theme-pack', resolvedThemePack);
      }
      if (storedColorMode) {
        root.setAttribute('data-theme', storedColorMode);
      } else {
        root.removeAttribute('data-theme');
      }
    })();
  </script>
  <link rel="stylesheet" href="/assets/share-theme.css">
</head>
<body id="share-unlock-page">
  <form id="share-unlock-form" method="post" action="__RAGTIME_FORM_ACTION__">
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
