// __RAGTIME_RUNTIME_BRIDGE_VERSION_TAG__ — platform-managed, do not edit
(function () {
  var B = 'userspace-exec-v1';
  var E = 'ragtime-execute';
  var R = 'ragtime-execute-result';
  var S = 'ragtime-sandbox-blocked';
  var H = 'ragtime-handshake';
  var H_ACK = 'ragtime-handshake-ack';
  var T = __RAGTIME_RUNTIME_BRIDGE_TIMEOUT_MS__;
  var T_LABEL = '__RAGTIME_RUNTIME_BRIDGE_TIMEOUT_LABEL__';
  // Parent-reachability probe timeout. Short by design: the parent only has
  // to ack a handshake, no API call involved. If the iframe parent is broken
  // (origin mismatch, handler not registered, parent navigated, sandbox
  // restriction) we fall back to direct execution within this window instead
  // of waiting the full T timeout.
  var H_TIMEOUT_MS = 2500;
  var CHART_URL = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
  var JQUERY_URL = 'https://code.jquery.com/jquery-3.7.1.min.js';
  var DATATABLES_JS_URL = 'https://cdn.datatables.net/1.13.8/js/dataTables.min.js';
  var DATATABLES_CSS_URL = 'https://cdn.datatables.net/1.13.8/css/dataTables.dataTables.min.css';
  var reportedSandboxBlocks = Object.create(null);

  function getBridgeConfig() {
    var config = window.__ragtime_preview_bridge;
    return config && typeof config === 'object' ? config : null;
  }

  function getParentOrigin() {
    var config = getBridgeConfig();
    var origin = config && typeof config.parent_origin === 'string'
      ? config.parent_origin.trim()
      : '';
    return origin || '*';
  }

  function normalizeOrigin(value) {
    try {
      var parsed = new URL(String(value || ''));
      var protocol = parsed.protocol || '';
      var hostname = parsed.hostname || '';
      var port = parsed.port || '';
      if ((protocol === 'https:' && port === '443') || (protocol === 'http:' && port === '80')) {
        port = '';
      }
      if (!protocol || !hostname) return '';
      return protocol + '//' + hostname + (port ? ':' + port : '');
    } catch (error) {
      return String(value || '').trim();
    }
  }

  function hasDataTables() {
    return !!(window.jQuery && window.jQuery.fn && window.jQuery.fn.DataTable);
  }

  function ensureStylesheet(href) {
    if (document.querySelector('link[href="' + href + '"]')) return;
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var existing = document.querySelector('script[src="' + src + '"]');
      if (existing) {
        if (existing.getAttribute('data-ragtime-loaded') === '1') {
          resolve();
          return;
        }
        existing.addEventListener('load', function () { resolve(); }, { once: true });
        existing.addEventListener('error', function () { reject(new Error('Failed to load ' + src)); }, { once: true });
        return;
      }
      var script = document.createElement('script');
      script.src = src;
      script.async = false;
      script.onload = function () {
        script.setAttribute('data-ragtime-loaded', '1');
        resolve();
      };
      script.onerror = function () { reject(new Error('Failed to load ' + src)); };
      document.head.appendChild(script);
    });
  }

  function bootstrapVizLibs() {
    if (window.__ragtime_viz_bootstrap_promise) return window.__ragtime_viz_bootstrap_promise;

    ensureStylesheet(DATATABLES_CSS_URL);

    var canUseDocumentWrite = document.readyState === 'loading' && !!document.currentScript;
    if (canUseDocumentWrite) {
      if (!window.Chart) {
        document.write('<script src="' + CHART_URL + '"><\/script>');
      }
      if (!window.jQuery) {
        document.write('<script src="' + JQUERY_URL + '"><\/script>');
      }
      if (!hasDataTables()) {
        document.write('<script src="' + DATATABLES_JS_URL + '"><\/script>');
      }
      window.__ragtime_viz_bootstrap_promise = Promise.resolve();
      return window.__ragtime_viz_bootstrap_promise;
    }

    var chain = Promise.resolve();
    if (!window.Chart) {
      chain = chain.then(function () { return loadScript(CHART_URL); });
    }
    if (!window.jQuery) {
      chain = chain.then(function () { return loadScript(JQUERY_URL); });
    }
    if (!hasDataTables()) {
      chain = chain.then(function () { return loadScript(DATATABLES_JS_URL); });
    }

    window.__ragtime_viz_bootstrap_promise = chain.catch(function (error) {
      console.warn('[ragtime bridge] visualization bootstrap failed:', error);
    });
    return window.__ragtime_viz_bootstrap_promise;
  }

  function getSandboxFlags() {
    if (Array.isArray(window.__ragtime_preview_sandbox_flags)) {
      return window.__ragtime_preview_sandbox_flags;
    }
    try {
      var frame = window.frameElement;
      if (frame && frame.sandbox) {
        if (typeof frame.sandbox.contains === 'function' && typeof frame.sandbox.length === 'number') {
          return Array.prototype.slice.call(frame.sandbox);
        }
        if (typeof frame.getAttribute === 'function') {
          var attr = frame.getAttribute('sandbox');
          if (attr) {
            return attr.split(/\s+/).filter(Boolean);
          }
        }
      }
    } catch (error) {
      console.warn('[ragtime bridge] failed to inspect sandbox flags:', error);
    }
    return null;
  }

  function hasSandboxFlag(flag) {
    var flags = getSandboxFlags();
    if (!flags) {
      return true;
    }
    return flags.indexOf(flag) >= 0;
  }

  function reportSandboxBlocked(action, flag) {
    var message = 'Preview sandbox blocked ' + action + '. Enable ' + flag + ' in Settings > User Space Preview Sandbox if this workspace needs it.';
    if (reportedSandboxBlocks[message]) {
      return;
    }
    reportedSandboxBlocks[message] = true;
    console.warn('[ragtime bridge] ' + message);
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage(
          { bridge: B, type: S, action: action, flag: flag, message: message },
          getParentOrigin()
        );
      }
    } catch (error) {
      console.warn('[ragtime bridge] failed to report sandbox block:', error);
    }
  }

  function overrideMethod(target, key, buildWrapper) {
    if (!target || typeof buildWrapper !== 'function') {
      return;
    }
    var original = target[key];
    if (typeof original !== 'function' || original.__ragtimeSandboxWrapped) {
      return;
    }
    var wrapped = buildWrapper(original);
    if (typeof wrapped !== 'function') {
      return;
    }
    wrapped.__ragtimeSandboxWrapped = true;
    try {
      target[key] = wrapped;
    } catch (error) {
      console.warn('[ragtime bridge] failed to override ' + key + ':', error);
    }
  }

  function installSandboxGuards() {
    if (!hasSandboxFlag('allow-modals')) {
      overrideMethod(window, 'alert', function () {
        return function () {
          reportSandboxBlocked('window.alert()', 'allow-modals');
        };
      });
      overrideMethod(window, 'confirm', function () {
        return function () {
          reportSandboxBlocked('window.confirm()', 'allow-modals');
          return false;
        };
      });
      overrideMethod(window, 'prompt', function () {
        return function () {
          reportSandboxBlocked('window.prompt()', 'allow-modals');
          return null;
        };
      });
    }

    if (!hasSandboxFlag('allow-popups')) {
      overrideMethod(window, 'open', function () {
        return function () {
          reportSandboxBlocked('window.open()', 'allow-popups');
          return null;
        };
      });
    }

    if (window.HTMLAnchorElement && window.HTMLAnchorElement.prototype) {
      overrideMethod(window.HTMLAnchorElement.prototype, 'click', function (original) {
        return function () {
          if (this && this.hasAttribute && this.hasAttribute('download') && !hasSandboxFlag('allow-downloads')) {
            reportSandboxBlocked('download link activation', 'allow-downloads');
            return;
          }
          return original.apply(this, arguments);
        };
      });
    }

    if (!hasSandboxFlag('allow-forms') && window.HTMLFormElement && window.HTMLFormElement.prototype) {
      overrideMethod(window.HTMLFormElement.prototype, 'submit', function () {
        return function () {
          reportSandboxBlocked('HTMLFormElement.submit()', 'allow-forms');
        };
      });
      overrideMethod(window.HTMLFormElement.prototype, 'requestSubmit', function () {
        return function () {
          reportSandboxBlocked('HTMLFormElement.requestSubmit()', 'allow-forms');
        };
      });
    }

    if (!hasSandboxFlag('allow-pointer-lock') && window.Element && window.Element.prototype) {
      overrideMethod(window.Element.prototype, 'requestPointerLock', function () {
        return function () {
          reportSandboxBlocked('Element.requestPointerLock()', 'allow-pointer-lock');
        };
      });
    }

    if (!hasSandboxFlag('allow-orientation-lock') && window.screen && window.screen.orientation) {
      overrideMethod(window.screen.orientation, 'lock', function () {
        return function () {
          reportSandboxBlocked('screen.orientation.lock()', 'allow-orientation-lock');
          return Promise.reject(new Error('Preview sandbox blocked screen.orientation.lock()'));
        };
      });
    }

    if (!hasSandboxFlag('allow-storage-access-by-user-activation') && document) {
      overrideMethod(document, 'requestStorageAccess', function () {
        return function () {
          reportSandboxBlocked('document.requestStorageAccess()', 'allow-storage-access-by-user-activation');
          return Promise.reject(new Error('Preview sandbox blocked document.requestStorageAccess()'));
        };
      });
    }

    if (!hasSandboxFlag('allow-presentation') && window.PresentationRequest && window.PresentationRequest.prototype) {
      overrideMethod(window.PresentationRequest.prototype, 'start', function () {
        return function () {
          reportSandboxBlocked('PresentationRequest.start()', 'allow-presentation');
          return Promise.reject(new Error('Preview sandbox blocked PresentationRequest.start()'));
        };
      });
    }
  }

  installSandboxGuards();

  // Viz libs are loaded lazily on first execute() call, not eagerly on page
  // load — avoids parser-blocking document.write warnings for cross-site CDN
  // scripts in proxy/shared contexts where the libs are never needed.

  function getDirectExecuteUrl() {
    var config = getBridgeConfig();
    return config && typeof config.execute_url === 'string'
      ? config.execute_url
      : null;
  }

  function executeDirect(componentId, request, resolve) {
    var executeUrl = getDirectExecuteUrl();
    if (!executeUrl) {
      resolve({ rows: [], columns: [], row_count: 0, error: 'Live data host unavailable in this context' });
      return;
    }

    var controller = null;
    var abortTimer = null;
    try {
      if (typeof AbortController === 'function') {
        controller = new AbortController();
        abortTimer = setTimeout(function () {
          try { controller.abort(); } catch (_e) { /* noop */ }
        }, T);
      }
    } catch (_e) {
      controller = null;
    }

    var fetchOpts = {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ component_id: componentId, request: request || {} }),
    };
    if (controller) {
      fetchOpts.signal = controller.signal;
    }

    fetch(executeUrl, fetchOpts)
      .then(function (response) {
        return response
          .json()
          .catch(function () { return {}; })
          .then(function (payload) { return { ok: response.ok, payload: payload }; });
      })
      .then(function (result) {
        if (abortTimer) { clearTimeout(abortTimer); abortTimer = null; }
        if (result.ok) {
          resolve(result.payload || { rows: [], columns: [], row_count: 0 });
          return;
        }
        var detail = result.payload && (result.payload.detail || result.payload.error);
        resolve({
          rows: [],
          columns: [],
          row_count: 0,
          error: detail || 'Failed to execute live data component',
        });
      })
      .catch(function (error) {
        if (abortTimer) { clearTimeout(abortTimer); abortTimer = null; }
        var isAbort = error && (error.name === 'AbortError' || error.code === 20);
        resolve({
          rows: [],
          columns: [],
          row_count: 0,
          error: isAbort
            ? ('Live data execution timed out after ' + T_LABEL)
            : (error && error.message ? error.message : String(error)),
        });
      });
  }

  // Parent-reachability state machine. The first execute() call kicks off a
  // handshake postMessage to the parent host. Until the handshake completes
  // (ack received OR H_TIMEOUT_MS elapses), execute() calls are queued. After
  // resolution, calls route through postMessage when reachable, or fall back
  // to executeDirect when not — without ever waiting the full T timeout for
  // every call against a broken parent.
  var parentReachableState = 'unknown'; // 'unknown' | 'reachable' | 'unreachable'
  var pendingHandshakeWaiters = [];
  var handshakeStarted = false;

  function resolveHandshake(state) {
    if (parentReachableState !== 'unknown') return;
    parentReachableState = state;
    var waiters = pendingHandshakeWaiters;
    pendingHandshakeWaiters = [];
    for (var i = 0; i < waiters.length; i++) {
      try { waiters[i](state); } catch (_e) { /* noop */ }
    }
  }

  function startParentHandshake() {
    if (handshakeStarted) return;
    handshakeStarted = true;

    var hasParentHost = !!(window.parent && window.parent !== window);
    if (!hasParentHost) {
      resolveHandshake('unreachable');
      return;
    }

    var parentOrigin = getParentOrigin();
    var expectedParentOrigin = parentOrigin === '*' ? '' : normalizeOrigin(parentOrigin);
    var callId = '__handshake_' + Math.random().toString(36).slice(2) + '_' + Date.now();

    function ackHandler(event) {
      if (parentReachableState !== 'unknown') return;
      if (event.source !== window.parent) return;
      if (expectedParentOrigin && normalizeOrigin(event.origin) !== expectedParentOrigin) return;
      if (
        event.data &&
        event.data.bridge === B &&
        event.data.type === H_ACK
      ) {
        window.removeEventListener('message', ackHandler);
        clearTimeout(probeTimer);
        resolveHandshake('reachable');
      }
    }
    window.addEventListener('message', ackHandler);

    var probeTimer = setTimeout(function () {
      window.removeEventListener('message', ackHandler);
      resolveHandshake('unreachable');
    }, H_TIMEOUT_MS);

    try {
      window.parent.postMessage(
        { bridge: B, type: H, callId: callId },
        parentOrigin
      );
    } catch (_error) {
      window.removeEventListener('message', ackHandler);
      clearTimeout(probeTimer);
      resolveHandshake('unreachable');
    }
  }

  function whenParentResolved(callback) {
    if (parentReachableState !== 'unknown') {
      callback(parentReachableState);
      return;
    }
    pendingHandshakeWaiters.push(callback);
    startParentHandshake();
  }

  function executeViaPostMessage(componentId, request, resolve) {
    var parentOrigin = getParentOrigin();
    var expectedParentOrigin = parentOrigin === '*' ? '' : normalizeOrigin(parentOrigin);
    var callId = '__exec_' + Math.random().toString(36).slice(2) + '_' + Date.now();
    var completed = false;

    var timer = setTimeout(function () {
      if (completed) return;
      completed = true;
      window.removeEventListener('message', handler);
      // Safety net: even after a successful handshake the data execution may
      // hang server-side. Fall back to direct execution rather than reporting
      // an error so the workspace UI eventually receives a result.
      executeDirect(componentId, request, resolve);
    }, T);
    function handler(event) {
      if (completed) return;
      if (event.source !== window.parent) return;
      if (expectedParentOrigin && normalizeOrigin(event.origin) !== expectedParentOrigin) return;
      if (
        event.data &&
        event.data.bridge === B &&
        event.data.type === R &&
        event.data.callId === callId
      ) {
        completed = true;
        window.removeEventListener('message', handler);
        clearTimeout(timer);
        resolve(event.data.result || { rows: [], columns: [], row_count: 0, error: 'Empty response' });
      }
    }
    window.addEventListener('message', handler);
    try {
      window.parent.postMessage(
        { bridge: B, type: E, callId: callId, component_id: componentId, request: request || {} },
        parentOrigin
      );
    } catch (error) {
      if (completed) return;
      completed = true;
      window.removeEventListener('message', handler);
      clearTimeout(timer);
      executeDirect(componentId, request, resolve);
    }
  }

  function makeExecute(componentId) {
    return function execute(request) {
      // Lazily bootstrap viz libs on first execute call.
      bootstrapVizLibs();
      var hasParentHost = !!(window.parent && window.parent !== window);
      return new Promise(function (resolve) {
        if (!hasParentHost) {
          executeDirect(componentId, request, resolve);
          return;
        }
        whenParentResolved(function (state) {
          if (state === 'reachable') {
            executeViaPostMessage(componentId, request, resolve);
          } else {
            executeDirect(componentId, request, resolve);
          }
        });
      });
    };
  }

  // Kick off the handshake eagerly so the very first execute() call doesn't
  // pay the H_TIMEOUT_MS detection cost on top of its own latency.
  if (window.parent && window.parent !== window) {
    startParentHandshake();
  }


  var componentsProxy = new Proxy({}, {
    get: function (_, prop) {
      if (typeof prop !== 'string') return undefined;
      return Object.freeze({ component_id: prop, execute: makeExecute(prop) });
    },
    has: function () { return true; },
  });

  window.__ragtime_context = Object.freeze({
    components: Object.freeze(componentsProxy),
  });
  if (!window.context) { window.context = window.__ragtime_context; }
})();