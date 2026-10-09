import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ContentProtectionSettingsSection } from './ContentProtectionSettingsSection';
vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({ models: [], loading: false, error: null, refresh: vi.fn() }),
}));
const config = {
  schema_version: 2 as const,
  revision: 3,
  enabled: true,
  share_with_assistant: false,
  classifier: {
    backend: 'jev' as const,
    jev: { transport: 'auto' as const, model: 'jev-latest' },
    llm_model: null,
  },
  strictness: 'strict' as const,
  categories: [
    {
      id: 'credentials',
      name: 'Credentials',
      description: 'Credentials.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Restricted.',
      threshold_override: null,
      system: false,
    },
    {
      id: 'rule_override',
      name: 'Rule override',
      description: 'Overrides.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Cannot override.',
      threshold_override: null,
      system: true,
    },
  ],
  access_levels: [{ id: 'standard', name: 'Standard', granted_category_ids: [], guidance: '' }],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'all_supported_traffic' as const,
  requirements: [],
  user_overrides: [],
};
const catalog = {
  users: [{ id: 'u1', name: 'Ada' }],
  groups: [],
  tools: [],
  mcp_routes: [],
  surfaces: [{ id: 'chat', name: 'Chat' }],
  classifier_status: { typesafe_key_configured: false, openrouter_key_configured: false },
};
function response(body: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } }),
  );
}
function installFetch(overrides: Record<string, unknown> = {}) {
  const fetch = vi.fn((url: string, options?: RequestInit) => {
    if (url.endsWith('/config') && options?.method === 'PUT') return response(config);
    if (url.endsWith('/config')) return response(config);
    if (url.endsWith('/catalog')) return response(catalog);
    if (url.endsWith('/preview'))
      return response(
        overrides.preview || {
          required: true,
          provenance: 'test',
          access_levels: [config.access_levels],
          granted_category_ids: [],
          categories: [],
          guidance: [],
          policy_revision: 3,
          guidance_revision: 'x',
          share_with_assistant: false,
        },
      );
    if (url.endsWith('/test'))
      return response(overrides.test || { code: 'ready', verdict: 'allow' });
    if (url.endsWith('/readiness'))
      return response(overrides.readiness || { code: 'ready', verdict: 'allow' });
    return response({ items: [] });
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
async function openInspect() {
  const user = userEvent.setup();
  render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
  await user.click(await screen.findByRole('button', { name: 'Test & inspect' }));
  return user;
}
async function openSetup() {
  const user = userEvent.setup();
  render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
  await user.click(await screen.findByRole('button', { name: 'Configure classifier' }));
  return user;
}
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
describe('ContentProtectionSettingsSection', () => {
  it('cancels a category edit with Escape without closing the setup draft', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    const card = document.getElementById('content-protection-category-credentials')!;
    const name = within(card).getByRole('textbox', { name: 'Name' }) as HTMLInputElement;
    await user.clear(name);
    await user.type(name, 'Uncommitted category name');
    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(name.value).toBe('Credentials');
  });
  it('saves the exact draft revision and prevents edits or dismissal during the save', async () => {
    let finishSave!: (response: Response) => void;
    const pendingSave = new Promise<Response>((resolve) => {
      finishSave = resolve;
    });
    const fetch = vi.fn((url: string, options?: RequestInit) => {
      if (options?.method === 'PUT') return pendingSave;
      return response(url.endsWith('/catalog') ? catalog : config);
    });
    vi.stubGlobal('fetch', fetch);
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.selectOptions(screen.getByLabelText('Strictness'), 'balanced');
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    const savedBody = JSON.parse(
      fetch.mock.calls.find(([, options]) => options?.method === 'PUT')?.[1]?.body as string,
    );
    expect(savedBody).toMatchObject({
      expected_revision: 3,
      config: { strictness: 'balanced', revision: 3 },
    });
    expect(screen.getByLabelText('Strictness').matches(':disabled')).toBe(true);
    expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect((screen.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(true);
    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog')).toBeTruthy();
    await act(async () =>
      finishSave(new Response(JSON.stringify({ ...savedBody.config, revision: 4 }))),
    );
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('preserves a conflicted draft when reload fails and then adopts the server revision', async () => {
    let failReload = false;
    let serverConfig = config;
    const fetch = vi.fn((url: string, options?: RequestInit) => {
      if (options?.method === 'PUT') {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Conflict' }), { status: 409 }),
        );
      }
      if (url.endsWith('/config') && failReload) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Reload unavailable' }), { status: 503 }),
        );
      }
      return response(url.endsWith('/catalog') ? catalog : serverConfig);
    });
    vi.stubGlobal('fetch', fetch);
    const user = await openSetup();
    await user.clear(screen.getByLabelText('Model'));
    await user.type(screen.getByLabelText('Model'), 'jev-1.13.0');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await screen.findByRole('button', { name: 'Reload and discard draft' });
    failReload = true;
    await user.click(screen.getByRole('button', { name: 'Reload and discard draft' }));
    expect(await screen.findByText('Reload unavailable')).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Classifier' }));
    expect((screen.getByLabelText('Model') as HTMLInputElement).value).toBe('jev-1.13.0');
    failReload = false;
    serverConfig = { ...config, revision: 5 };
    await user.click(screen.getByRole('button', { name: 'Reload and discard draft' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    await user.click(screen.getByRole('button', { name: 'Configure classifier' }));
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    const writes = fetch.mock.calls.filter(([, options]) => options?.method === 'PUT');
    expect(JSON.parse(writes[1][1]?.body as string).expected_revision).toBe(5);
  });

  it('retains dormant requirements and user overrides when coverage and enforcement change', async () => {
    const original = {
      ...config,
      requirements: [{ scope_kind: 'tool', scope_key: 'preserved-tool', mode: 'require' }],
      user_overrides: [{ user_id: 'other-user', mode: 'never_classify' }],
    };
    const fetch = vi.fn((url: string, options?: RequestInit) => {
      if (options?.method === 'PUT') return response(JSON.parse(options.body as string).config);
      return response(url.endsWith('/catalog') ? catalog : original);
    });
    vi.stubGlobal('fetch', fetch);
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByLabelText('Enable content protection'));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    const write = fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
    expect(JSON.parse(write?.[1]?.body as string).config).toMatchObject({
      enabled: false,
      coverage_mode: 'selected_scopes',
      requirements: original.requirements,
      user_overrides: original.user_overrides,
    });
  });

  it('supports keyboard tab navigation and restores focus after Escape', async () => {
    installFetch();
    const user = await openSetup();
    const first = screen.getByRole('tab', { name: 'Classifier' });
    first.focus();
    await user.keyboard('{End}');
    expect(screen.getByRole('tab', { name: 'Review & save' }).getAttribute('aria-selected')).toBe(
      'true',
    );
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.activeElement).toBe(
      screen.getByRole('button', { name: 'Configure classifier' }),
    );
  });
  it('passes a selected real user to the test API', async () => {
    const fetch = installFetch();
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    await user.type(screen.getByLabelText('Sample content'), 'secret');
    await user.selectOptions(screen.getByLabelText('Or test as a real user'), 'u1');
    await user.click(screen.getByRole('button', { name: 'Test content' }));
    await waitFor(() =>
      expect(
        JSON.parse(
          fetch.mock.calls.find(([url]) => String(url).endsWith('/test'))?.[1]?.body as string,
        ),
      ).toMatchObject({ access_level_ids: [], user_id: 'u1' }),
    );
  });
  it('disables level selection while a real user is selected', async () => {
    installFetch();
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    await user.selectOptions(screen.getByLabelText('Or test as a real user'), 'u1');
    expect(
      (screen.getByRole('group', { name: 'Target access levels' }) as HTMLFieldSetElement).disabled,
    ).toBe(true);
  });
  it('renders advisory prompt fragment only when preview enables sharing', async () => {
    installFetch({
      preview: {
        required: true,
        provenance: 'test',
        access_levels: [],
        granted_category_ids: [],
        categories: [],
        guidance: ['different guidance'],
        prompt_fragment: 'audience fragment',
        policy_revision: 3,
        guidance_revision: 'x',
        share_with_assistant: true,
      },
    });
    const user = await openInspect();
    await user.click(screen.getByRole('button', { name: 'Check saved policy' }));
    expect(await screen.findByText('audience fragment')).toBeTruthy();
  });
  it('shows category probabilities with threshold denials', async () => {
    installFetch({
      test: { code: 'content_denied', verdict: 'deny', probabilities: { credentials: 0.85 } },
    });
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    await user.type(screen.getByLabelText('Sample content'), 'secret');
    await user.click(screen.getByRole('button', { name: 'Test content' }));
    expect(await screen.findByText('85.0%')).toBeTruthy();
    expect(screen.getByText('above threshold')).toBeTruthy();
  });
  it('shows a retryable load error instead of a permanent loading state', async () => {
    let fails = true;
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) =>
        url.endsWith('/config') && fails
          ? Promise.resolve(new Response(JSON.stringify({ detail: 'offline' }), { status: 500 }))
          : response(url.endsWith('/catalog') ? catalog : config),
      ),
    );
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    expect(await screen.findByText('Unable to load content protection settings')).toBeTruthy();
    fails = false;
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('button', { name: 'Configure classifier' })).toBeTruthy();
  });
  it('reloads the saved configuration when refreshKey changes', async () => {
    const fetch = installFetch();
    const view = render(
      <ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={0} />,
    );

    await screen.findByRole('button', { name: 'Configure classifier' });
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/config'))).toHaveLength(1);

    view.rerender(<ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={1} />);

    await waitFor(() =>
      expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/config'))).toHaveLength(2),
    );
  });
  it('sends the selected baseline, MCP route, and tool in a saved preview', async () => {
    const fetch = installFetch();
    const user = await openInspect();
    await user.selectOptions(screen.getByLabelText('Who is making the request?'), 'public');
    await user.selectOptions(screen.getByLabelText('MCP route (optional)'), '');
    await user.selectOptions(screen.getByLabelText('Tool (optional)'), '');
    await user.click(screen.getByRole('button', { name: 'Check saved policy' }));
    await waitFor(() => {
      const body = JSON.parse(
        fetch.mock.calls.find(([url]) => String(url).endsWith('/preview'))?.[1]?.body as string,
      );
      expect(body).toMatchObject({ baseline: 'public', public: true, surface: 'chat' });
    });
  });
  it('marks the service preview with the service baseline', async () => {
    const fetch = installFetch();
    const user = await openInspect();
    await user.click(screen.getByRole('button', { name: 'Check saved policy' }));
    await waitFor(() =>
      expect(
        JSON.parse(
          fetch.mock.calls.find(([url]) => String(url).endsWith('/preview'))?.[1]?.body as string,
        ),
      ).toMatchObject({ baseline: 'service', public: false }),
    );
  });
  it('clears a preview when its identity changes', async () => {
    installFetch();
    const user = await openInspect();
    await user.click(screen.getByRole('button', { name: 'Check saved policy' }));
    expect(await screen.findByText('Classification required')).toBeTruthy();
    await user.selectOptions(screen.getByLabelText('Who is making the request?'), 'public');
    expect(screen.queryByText('Classification required')).toBeNull();
  });
  it('clears a sample result when the sample changes', async () => {
    installFetch();
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    const sample = screen.getByLabelText('Sample content');
    await user.type(sample, 'secret');
    await user.click(screen.getByRole('button', { name: 'Test content' }));
    expect(await screen.findByText('allow · ready')).toBeTruthy();
    await user.type(sample, ' changed');
    expect(screen.queryByText('allow · ready')).toBeNull();
  });
  it('invalidates a readiness result when classifier settings change', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));
    expect(await screen.findByText('Ready')).toBeTruthy();
    await user.clear(screen.getByLabelText('Model'));
    await user.type(screen.getByLabelText('Model'), 'jev-pinned');
    expect(screen.queryByText('Ready')).toBeNull();
  });
  it('cancels readiness when the setup dialog is cancelled', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));
    expect(await screen.findByText('Ready')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();
  });
  it('ignores a readiness response that arrives after cancel', async () => {
    let resolveReadiness: ((value: Response) => void) | undefined;
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/readiness'))
          return new Promise<Response>((resolve) => {
            resolveReadiness = resolve;
          });
        return response(url.endsWith('/catalog') ? catalog : config);
      }),
    );
    const user = await openSetup();
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    resolveReadiness?.(new Response(JSON.stringify({ code: 'ready' })));
    await Promise.resolve();
    await user.click(screen.getByRole('button', { name: 'Configure classifier' }));
    expect(screen.queryByText('Ready')).toBeNull();
  });
  it('keeps Jev transport controls disabled when generic LLM is selected', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    expect((screen.getByRole('radio', { name: 'auto' }) as HTMLInputElement).disabled).toBe(true);
  });
  it('uses one named radio group to select the classifier backend', async () => {
    installFetch();
    await openSetup();
    expect(screen.getByRole('radio', { name: 'Jev' }).getAttribute('name')).toBe(
      'content-protection-backend',
    );
    expect(screen.getByLabelText('Use generic LLM').getAttribute('name')).toBe(
      'content-protection-backend',
    );
  });
  it('keeps unknown group mappings visible so they can be removed', async () => {
    const mapped = {
      ...config,
      group_access_levels: [{ group_id: 'gone', access_level_id: 'standard' }],
    };
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => response(url.endsWith('/catalog') ? catalog : mapped)),
    );
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Access levels' }));
    expect(await screen.findByText(/Missing group gone/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Remove mapping' }));
    expect(screen.queryByText(/Missing group gone/)).toBeNull();
  });
  it('shows an explicit decisions retry after a failed request', async () => {
    let fail = true;
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) =>
        url.endsWith('/decisions') && fail
          ? Promise.resolve(
              new Response(JSON.stringify({ detail: 'unavailable' }), { status: 503 }),
            )
          : response(
              url.endsWith('/catalog') ? catalog : url.endsWith('/config') ? config : { items: [] },
            ),
      ),
    );
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Recent decisions' }));
    expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy();
    fail = false;
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('No decision metadata available.')).toBeTruthy();
  });
  it('links OpenRouter key management to the owning settings field and closes setup', async () => {
    installFetch();
    const user = await openSetup();
    const link = screen.getByRole('link', { name: /Manage the OpenRouter key/ });
    expect(link.getAttribute('href')).toBe('#setting-openrouter-api-key');
    await user.click(link);
    expect(screen.queryByRole('dialog')).toBeNull();
  });
  it('renders semantic hooks for preview and advisory results', async () => {
    installFetch({
      preview: {
        required: false,
        provenance: 'test',
        access_levels: [],
        granted_category_ids: [],
        categories: [],
        guidance: [],
        prompt_fragment: 'fragment',
        policy_revision: 3,
        guidance_revision: 'x',
        share_with_assistant: true,
      },
    });
    const user = await openInspect();
    await user.click(screen.getByRole('button', { name: 'Check saved policy' }));
    await screen.findByText('Not required');
    expect(document.querySelector('[data-content-protection-preview-result]')).toBeTruthy();
    expect(document.querySelector('[data-content-protection-advisory-preview]')).toBeTruthy();
  });
  it('keeps policy navigation links in new tabs', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    expect(screen.getByRole('link', { name: 'User policies' }).getAttribute('target')).toBe(
      '_blank',
    );
    expect(screen.getByRole('link', { name: 'MCP routes' }).getAttribute('target')).toBe('_blank');
  });
});
