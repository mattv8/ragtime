import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ContentProtectionSettingsSection } from './ContentProtectionSettingsSection';
const refreshModels = vi.hoisted(() => vi.fn());
const availableModels = vi.hoisted(() => ({
  models: [] as { id: string; name: string; provider: string }[],
}));
vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({
    ...availableModels,
    loading: false,
    error: null,
    refresh: refreshModels,
  }),
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
const genericModels = [
  { id: 'generic-a', name: 'Generic model A', provider: 'openai' },
  { id: 'generic-b', name: 'Generic model B', provider: 'openrouter' },
];
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
    if (url.endsWith('/models')) {
      const models = overrides.models || {
        success: true,
        transport: 'typesafe',
        models: [],
        default_model: 'jev-latest',
        message: null,
      };
      return typeof models === 'function' ? models() : response(models);
    }
    if (url.endsWith('/settings') && options?.method === 'PUT') return response({ settings: {} });
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
async function openSetup({
  onOpenRouterKeySaved,
  onTypesafeKeySaved,
  savedTypesafeKey,
  savedOpenRouterKey,
}: {
  onOpenRouterKeySaved?: (apiKey: string) => void;
  onTypesafeKeySaved?: (apiKey: string) => void;
  savedTypesafeKey?: string;
  savedOpenRouterKey?: string;
} = {}) {
  const user = userEvent.setup();
  render(
    <ContentProtectionSettingsSection
      open
      onToggle={() => {}}
      onOpenRouterKeySaved={onOpenRouterKeySaved}
      onTypesafeKeySaved={onTypesafeKeySaved}
      savedTypesafeKey={savedTypesafeKey}
      savedOpenRouterKey={savedOpenRouterKey}
    />,
  );
  await user.click(await screen.findByRole('button', { name: 'Configure classifier' }));
  return user;
}
async function selectGenericLlmModel(
  user: ReturnType<typeof userEvent.setup>,
  providerName: 'OpenAI' | 'OpenRouter' = 'OpenAI',
) {
  await user.click(
    screen.getByRole('button', { name: /^(Select a generic model|OpenAI Generic model A)/ }),
  );
  await user.click(screen.getByRole('button', { name: providerName }));
}
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  availableModels.models = [];
});
describe('ContentProtectionSettingsSection', () => {
  it('binds saved keys according to the selected Jev provider', async () => {
    installFetch();
    const user = await openSetup({
      savedTypesafeKey: 'ts_saved-typesafe',
      savedOpenRouterKey: 'sk-or-saved-openrouter',
    });
    const key = screen.getByLabelText('API key') as HTMLInputElement;

    expect(key.value).toBe('ts_saved-typesafe');
    await user.selectOptions(screen.getByLabelText('Provider'), 'openrouter');
    expect(key.value).toBe('sk-or-saved-openrouter');
    await user.selectOptions(screen.getByLabelText('Provider'), 'typesafe');
    expect(key.value).toBe('ts_saved-typesafe');
  });

  it('keeps an unrecognized preferred TypeSafe key server-side for auto model fetches', async () => {
    const fetch = installFetch();
    const user = await openSetup({
      savedTypesafeKey: 'legacy-typesafe-key',
      savedOpenRouterKey: 'sk-or-saved-openrouter',
    });

    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe('');
    expect(
      (screen.getByRole('button', { name: 'Fetch Models' }) as HTMLButtonElement).disabled,
    ).toBe(false);
    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));

    await waitFor(() =>
      expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/models'))).toBe(true),
    );
    const modelRequest = fetch.mock.calls.find(([url]) => String(url).endsWith('/models'));
    expect(JSON.parse(modelRequest?.[1]?.body as string)).toMatchObject({
      transport: 'auto',
      api_key: null,
    });
  });

  it('uses the saved OpenRouter key in auto mode only when TypeSafe is absent', async () => {
    installFetch();
    await openSetup({ savedOpenRouterKey: 'sk-or-saved-openrouter' });

    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
      'sk-or-saved-openrouter',
    );
  });

  it('keeps an edited Jev key when the provider changes or catalog refreshes', async () => {
    installFetch();
    const user = await openSetup({
      savedTypesafeKey: 'ts_saved-typesafe',
      savedOpenRouterKey: 'sk-or-saved-openrouter',
    });
    const key = screen.getByLabelText('API key');

    await user.clear(key);
    await user.type(key, 'ts_edited-key');
    await user.selectOptions(screen.getByLabelText('Provider'), 'openrouter');
    expect((key as HTMLInputElement).value).toBe('ts_edited-key');

    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));
    await waitFor(() => expect((key as HTMLInputElement).value).toBe('ts_edited-key'));
  });

  it('does not replace an edited Jev draft when saved-key props refresh', async () => {
    installFetch();
    const user = userEvent.setup();
    const view = render(
      <ContentProtectionSettingsSection
        open
        onToggle={() => {}}
        savedTypesafeKey="ts_saved-typesafe"
      />,
    );
    await user.click(await screen.findByRole('button', { name: 'Configure classifier' }));
    const key = screen.getByLabelText('API key');
    await user.clear(key);
    await user.type(key, 'ts_edited-key');

    view.rerender(
      <ContentProtectionSettingsSection
        open
        onToggle={() => {}}
        savedTypesafeKey="ts_refreshed-typesafe"
      />,
    );
    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe('ts_edited-key');
  });

  it('does not replace a newer Jev draft when an earlier key save resolves', async () => {
    let resolveSave!: (response: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, options?: RequestInit) => {
        if (url.endsWith('/settings') && options?.method === 'PUT') {
          return new Promise<Response>((resolve) => {
            resolveSave = resolve;
          });
        }
        return response(url.endsWith('/catalog') ? catalog : config);
      }),
    );
    const user = await openSetup();
    const key = screen.getByLabelText('API key');
    await user.type(key, 'ts_saved-key');
    await user.click(screen.getByRole('button', { name: 'Save key' }));
    await user.clear(key);
    await user.type(key, 'ts_newer-draft');
    await act(async () => resolveSave(new Response(JSON.stringify({ settings: {} }))));

    await waitFor(() => {
      expect((key as HTMLInputElement).value).toBe('ts_newer-draft');
      expect((screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement).disabled).toBe(
        false,
      );
    });
  });

  it('retains successful saved keys for provider switches and reopening setup', async () => {
    installFetch();
    const user = await openSetup();
    const key = screen.getByLabelText('API key');

    await user.type(key, 'ts_saved-after-save');
    await user.click(screen.getByRole('button', { name: 'Save key' }));
    await waitFor(() => expect((key as HTMLInputElement).value).toBe('ts_saved-after-save'));

    await user.selectOptions(screen.getByLabelText('Provider'), 'openrouter');
    await user.clear(key);
    await user.type(key, 'sk-or-saved-after-save');
    await user.click(screen.getByRole('button', { name: 'Save key' }));
    await waitFor(() => expect((key as HTMLInputElement).value).toBe('sk-or-saved-after-save'));

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Configure classifier' }));
    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
      'ts_saved-after-save',
    );
    await user.selectOptions(screen.getByLabelText('Provider'), 'openrouter');
    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
      'sk-or-saved-after-save',
    );
  });

  it('reports a successfully saved trimmed OpenRouter key to the parent', async () => {
    installFetch();
    const onOpenRouterKeySaved = vi.fn();
    const user = await openSetup({ onOpenRouterKeySaved });

    await user.type(screen.getByLabelText('API key'), ' sk-or-shared-key ');
    await user.click(screen.getByRole('button', { name: 'Save key' }));

    await waitFor(() => expect(onOpenRouterKeySaved).toHaveBeenCalledWith('sk-or-shared-key'));
  });

  it('reports a successfully saved TypeSafe key to the parent', async () => {
    installFetch();
    const onTypesafeKeySaved = vi.fn();
    const user = await openSetup({ onTypesafeKeySaved });

    await user.type(screen.getByLabelText('API key'), 'ts_saved-typesafe');
    await user.click(screen.getByRole('button', { name: 'Save key' }));

    await waitFor(() => expect(onTypesafeKeySaved).toHaveBeenCalledWith('ts_saved-typesafe'));
  });

  it('does not report an OpenRouter key when saving it fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, options?: RequestInit) => {
        if (url.endsWith('/settings') && options?.method === 'PUT') {
          return Promise.resolve(
            new Response(JSON.stringify({ detail: 'offline' }), { status: 503 }),
          );
        }
        return response(url.endsWith('/catalog') ? catalog : config);
      }),
    );
    const onOpenRouterKeySaved = vi.fn();
    const user = await openSetup({ onOpenRouterKeySaved });

    await user.type(screen.getByLabelText('API key'), 'sk-or-shared-key');
    await user.click(screen.getByRole('button', { name: 'Save key' }));

    expect(await screen.findByText('offline')).toBeTruthy();
    expect(onOpenRouterKeySaved).not.toHaveBeenCalled();
    expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe('sk-or-shared-key');
  });

  it('does not report a TypeSafe key save to the parent', async () => {
    installFetch();
    const onOpenRouterKeySaved = vi.fn();
    const user = await openSetup({ onOpenRouterKeySaved });

    await user.type(screen.getByLabelText('API key'), 'ts_typesafe-key');
    refreshModels.mockClear();
    await user.click(screen.getByRole('button', { name: 'Save key' }));

    await waitFor(() =>
      expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe('ts_typesafe-key'),
    );
    expect(refreshModels).not.toHaveBeenCalled();
    expect(onOpenRouterKeySaved).not.toHaveBeenCalled();
  });

  it('saves an Auto-detected apikey TypeSafe key to the TypeSafe setting', async () => {
    const fetch = installFetch();
    const user = await openSetup();

    await user.type(screen.getByLabelText('API key'), 'apikey_synthetic-typesafe-key');
    const saveKey = screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement;
    expect(saveKey.disabled).toBe(false);
    await user.click(saveKey);

    await waitFor(() =>
      expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/settings'))).toBe(true),
    );
    const settingsWrite = fetch.mock.calls.find(([url]) => String(url).endsWith('/settings'));
    expect(JSON.parse(settingsWrite?.[1]?.body as string)).toEqual({
      typesafe_api_key: 'apikey_synthetic-typesafe-key',
    });
  });

  it('creates distinct category and access-level IDs without crypto.randomUUID and saves them', async () => {
    let randomValue = 0;
    vi.stubGlobal('crypto', {
      getRandomValues: vi.fn((array: Uint8Array) => {
        array.fill(randomValue++);
        return array;
      }),
    });
    const fetch = installFetch();
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: 'Add category' }));
    await user.click(screen.getByRole('button', { name: 'Add category' }));
    await user.click(screen.getByRole('tab', { name: 'Access levels' }));
    await user.click(screen.getByRole('button', { name: 'Add access level' }));
    await user.click(screen.getByRole('button', { name: 'Add access level' }));
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));

    await waitFor(() =>
      expect(fetch.mock.calls.some(([, options]) => options?.method === 'PUT')).toBe(true),
    );
    const write = fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
    const savedConfig = JSON.parse(write?.[1]?.body as string).config;
    const categoryIds = savedConfig.categories
      .map((item: { id: string }) => item.id)
      .filter((id: string) => id.startsWith('cat_'));
    const levelIds = savedConfig.access_levels
      .map((item: { id: string }) => item.id)
      .filter((id: string) => id.startsWith('level_'));
    expect(categoryIds).toHaveLength(2);
    expect(levelIds).toHaveLength(2);
    expect(new Set(categoryIds).size).toBe(2);
    expect(new Set(levelIds).size).toBe(2);
    expect(categoryIds.every((id: string) => /^cat_[a-z0-9]{12}$/.test(id))).toBe(true);
    expect(levelIds.every((id: string) => /^level_[a-z0-9]{12}$/.test(id))).toBe(true);
  });

  it('hides system categories and returns focus to a visible category after deletion', async () => {
    const categories = {
      ...config,
      categories: [
        { ...config.categories[0], id: 'operational', name: 'Operational' },
        ...config.categories,
      ],
    };
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => response(url.endsWith('/catalog') ? catalog : categories)),
    );
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));

    expect(screen.queryByRole('button', { name: /Rule override/ })).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Delete Credentials' }));

    expect(screen.queryByRole('button', { name: /^Credentials\b/ })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull();
    expect(screen.queryByText('Rule override')).toBeNull();
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: /^Operational\b/ })),
    );
  });

  it('keeps the system category in config while capping all categories at 24 total', async () => {
    const editableCategories = Array.from({ length: 23 }, (_, index) => ({
      ...config.categories[0],
      id: `category-${index}`,
      name: `Category ${index}`,
    }));
    const atCapacity = { ...config, categories: [...editableCategories, config.categories[1]] };
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => response(url.endsWith('/catalog') ? catalog : atCapacity)),
    );
    const user = await openSetup();

    expect(screen.getByText('23 editable categories')).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    expect(
      (screen.getByRole('button', { name: 'Add category' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(screen.queryByText('Rule override')).toBeNull();
  });

  it('uses readable strictness labels in category rows without rounding custom values', async () => {
    const configured = {
      ...config,
      categories: [
        { ...config.categories[0], threshold_override: 0.25 },
        {
          ...config.categories[0],
          id: 'custom',
          name: 'Custom category',
          threshold_override: 0.3333,
        },
        config.categories[1],
      ],
    };
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => response(url.endsWith('/catalog') ? catalog : configured)),
    );
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    expect(screen.getByText('Strict (≥25%)')).toBeTruthy();
    expect(screen.getByText('Custom (≥33.33%)')).toBeTruthy();
    expect(screen.queryByText('Threshold 0.3333')).toBeNull();
  });

  it('deletes a granted category and removes it from access-level grants on save', async () => {
    const granted = {
      ...config,
      categories: [
        { ...config.categories[0], id: 'operational', name: 'Operational' },
        ...config.categories,
      ],
      access_levels: [{ ...config.access_levels[0], granted_category_ids: ['credentials'] }],
    };
    const fetch = vi.fn((url: string, _options?: RequestInit) => {
      if (url.endsWith('/catalog')) return response(catalog);
      if (url.endsWith('/config')) return response(granted);
      return response({ items: [] });
    });
    vi.stubGlobal('fetch', fetch);
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));

    const trash = screen.getByRole('button', { name: 'Delete Credentials' });
    expect(trash.getAttribute('aria-disabled')).toBeNull();
    await user.click(trash);
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));

    await waitFor(() =>
      expect(fetch.mock.calls.some(([, options]) => options?.method === 'PUT')).toBe(true),
    );
    const write = fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
    const savedConfig = JSON.parse(write?.[1]?.body as string).config;
    expect(savedConfig.categories.map((item: { id: string }) => item.id)).toEqual([
      'operational',
      'rule_override',
    ]);
    expect(savedConfig.access_levels[0].granted_category_ids).toEqual([]);
  });

  it('blocks saving enabled protection after every regular category is deleted', async () => {
    const fetch = installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: 'Delete Credentials' }));
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));

    expect(
      await screen.findByText('Add at least one category before saving enabled protection.'),
    ).toBeTruthy();
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'PUT')).toBe(false);
  });

  it('explains why the default access level cannot be deleted from the rail', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Access levels' }));

    const trash = screen.getByRole('button', { name: 'Delete Standard' });
    expect(trash.getAttribute('aria-disabled')).toBe('true');
    expect(trash.getAttribute('title')).toBe(
      'Cannot delete the default level. Assign a different default first.',
    );
    await user.click(trash);

    expect(
      await screen.findByText(
        'Cannot delete the default level. Assign a different default first.',
        {
          selector: ':not([hidden])',
        },
      ),
    ).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Standard\b/ })).toBeTruthy();
  });

  it('cancels a category edit with Escape without closing the setup draft', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));
    const detail = document.querySelector<HTMLElement>('[data-category-detail]')!;
    await user.click(within(detail).getByRole('button', { name: 'Edit category name' }));
    const name = within(detail).getByRole('textbox', { name: 'Category name' });
    await user.clear(name);
    await user.type(name, 'Uncommitted category name');
    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(within(detail).getByRole('heading', { name: 'Credentials' })).toBeTruthy();
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
    await user.click(screen.getByRole('tab', { name: 'Classifier' }));
    expect(document.querySelector('#content-protection-strictness')).toBeNull();
    expect(screen.queryByLabelText('Strictness')).toBeNull();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    const coverageFieldset = document.querySelector<HTMLElement>(
      'fieldset#content-protection-coverage-fieldset',
    )!;
    expect(coverageFieldset.querySelector('legend')?.textContent).toContain('Coverage');
    expect(within(coverageFieldset).getByRole('button', { name: 'Coverage help' })).toBeTruthy();
    const strictness = document.querySelector<HTMLElement>('#content-protection-strictness')!;
    expect(strictness.previousElementSibling).toBe(coverageFieldset);
    expect(coverageFieldset.lastElementChild?.id).toBe('content-protection-policy-links');
    expect(
      within(coverageFieldset).getByText('Configure specific policies (opens in a new tab):'),
    ).toBeTruthy();
    await user.selectOptions(screen.getByLabelText('Strictness'), 'balanced');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    expect(screen.queryByLabelText('Strictness')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    const savedBody = JSON.parse(
      fetch.mock.calls.find(([, options]) => options?.method === 'PUT')?.[1]?.body as string,
    );
    expect(savedBody).toMatchObject({
      expected_revision: 3,
      config: { strictness: 'balanced', revision: 3 },
    });
    expect(screen.getByRole('tab', { name: 'Review & save' }).matches(':disabled')).toBe(true);
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
    await user.click(
      within(screen.getByRole('dialog')).getByLabelText('Enable content protection'),
    );
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
  it('shows five numbered, freely navigable setup steps without changing inspect tabs', async () => {
    installFetch();
    const user = await openSetup();

    const setupSteps = screen.getByRole('tablist', { name: 'Configure content protection steps' });
    expect(within(setupSteps).getAllByRole('tab')).toHaveLength(5);
    expect(
      Array.from(setupSteps.querySelectorAll('.step-number')).map((step) => step.textContent),
    ).toEqual(['1', '2', '3', '4', '5']);
    expect(
      Array.from(setupSteps.querySelectorAll('.step-number')).every(
        (step) => step.getAttribute('aria-hidden') === 'true',
      ),
    ).toBe(true);

    await user.click(within(setupSteps).getByRole('tab', { name: 'Review & save' }));
    await user.click(within(setupSteps).getByRole('tab', { name: 'Categories' }));
    expect(
      within(setupSteps).getByRole('tab', { name: 'Categories' }).getAttribute('aria-selected'),
    ).toBe('true');

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Test & inspect' }));
    const inspectTabs = screen.getByRole('tablist', {
      name: 'Test and inspect content protection steps',
    });
    expect(inspectTabs.classList.contains('modal-tabs')).toBe(true);
    expect(inspectTabs.classList.contains('wizard-progress')).toBe(false);
  });
  it('positions a directly opened setup step within its horizontal progress strip', async () => {
    installFetch();
    const scrollToDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollTo');
    const scrollTo = vi.fn();
    const getBoundingClientRect = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockImplementation(function (this: HTMLElement) {
        if (this.hasAttribute('data-content-protection-setup-progress')) {
          return { left: 0, right: 100 } as DOMRect;
        }
        if (this.id === 'content-protection-setup-tab-categories') {
          return { left: 120, right: 180 } as DOMRect;
        }
        return { left: 0, right: 0 } as DOMRect;
      });
    Object.defineProperty(HTMLElement.prototype, 'scrollTo', {
      configurable: true,
      value: scrollTo,
      writable: true,
    });

    try {
      const user = userEvent.setup();
      render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
      await user.click(await screen.findByRole('button', { name: 'Configure categories' }));

      const setupSteps = screen.getByRole('tablist', {
        name: 'Configure content protection steps',
      });
      expect(scrollTo).toHaveBeenCalledWith({ behavior: 'auto', left: 80 });
      expect(scrollTo.mock.instances[0]).toBe(setupSteps);
    } finally {
      getBoundingClientRect.mockRestore();
      if (scrollToDescriptor) {
        Object.defineProperty(HTMLElement.prototype, 'scrollTo', scrollToDescriptor);
      } else {
        delete (HTMLElement.prototype as { scrollTo?: unknown }).scrollTo;
      }
    }
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
  it('labels the system detection score as a bypass attempt', async () => {
    installFetch({
      test: { code: 'ready', verdict: 'allow', probabilities: { rule_override: 0.2 } },
    });
    const user = await openInspect();
    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    await user.type(screen.getByLabelText('Sample content'), 'test');
    await user.click(screen.getByRole('button', { name: 'Test content' }));

    expect(await screen.findByText('Bypass attempt')).toBeTruthy();
    expect(screen.queryByText('Rule override')).toBeNull();
  });
  it('exposes the settings highlight target on the always-rendered accordion header', () => {
    render(<ContentProtectionSettingsSection open={false} onToggle={() => {}} />);
    const target = document.getElementById('setting-content_protection');
    expect(target?.getAttribute('aria-controls')).toBe(
      'settings-accordion-body-content-protection',
    );
    expect(
      target
        ?.closest('[data-settings-accordion-section]')
        ?.getAttribute('data-settings-accordion-section'),
    ).toBe('content-protection');
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
  it('makes manual readiness available only to the generic LLM classifier', async () => {
    installFetch();
    const user = await openSetup();
    expect(screen.queryByRole('button', { name: /Check readiness/ })).toBeNull();
    expect(
      document.querySelector(
        '#content-protection-classifier-summary .content-protection-status-icon',
      ),
    ).toBeNull();
    await user.click(screen.getByLabelText('Use generic LLM'));
    expect(screen.getByRole('button', { name: /Check readiness/ })).toBeTruthy();
  });
  it('requires a generic LLM model before readiness can be checked', async () => {
    installFetch();
    availableModels.models = genericModels;
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));

    expect(
      (screen.getByRole('button', { name: /Check readiness/ }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await selectGenericLlmModel(user);
    expect(
      (screen.getByRole('button', { name: /Check readiness/ }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });
  it('invalidates a generic LLM readiness result when its model changes', async () => {
    installFetch();
    availableModels.models = genericModels;
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    await selectGenericLlmModel(user);
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));
    expect(await screen.findByText('Ready')).toBeTruthy();
    await selectGenericLlmModel(user, 'OpenRouter');
    expect(screen.queryByText('Ready')).toBeNull();
  });
  it('cancels readiness when the setup dialog is cancelled', async () => {
    installFetch();
    availableModels.models = genericModels;
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    await selectGenericLlmModel(user);
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
    availableModels.models = genericModels;
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    await selectGenericLlmModel(user);
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    resolveReadiness?.(new Response(JSON.stringify({ code: 'ready' })));
    await Promise.resolve();
    await user.click(screen.getByRole('button', { name: 'Configure classifier' }));
    expect(screen.queryByText('Ready')).toBeNull();
  });
  it('keeps Jev provider controls disabled when generic LLM is selected', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    expect((screen.getByRole('combobox', { name: 'Provider' }) as HTMLSelectElement).disabled).toBe(
      true,
    );
  });
  it('checks readiness for a generic LLM classifier', async () => {
    const fetch = installFetch();
    availableModels.models = genericModels;
    const user = await openSetup();

    await user.click(screen.getByLabelText('Use generic LLM'));
    await selectGenericLlmModel(user);
    await user.click(screen.getByRole('button', { name: /Check readiness/ }));

    await waitFor(() => {
      const request = fetch.mock.calls.find(([url]) => String(url).endsWith('/readiness'));
      expect(JSON.parse(request?.[1]?.body as string)).toMatchObject({
        config: { classifier: { backend: 'llm' } },
      });
    });
  });
  it('keeps the generic LLM readiness action busy until its request resolves', async () => {
    let resolveReadiness!: (response: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/readiness')) {
          return new Promise<Response>((resolve) => {
            resolveReadiness = resolve;
          });
        }
        return response(url.endsWith('/catalog') ? catalog : config);
      }),
    );
    availableModels.models = genericModels;
    const user = await openSetup();
    await user.click(screen.getByLabelText('Use generic LLM'));
    await selectGenericLlmModel(user);
    const readinessButton = screen.getByRole('button', { name: /Check readiness/ });
    await user.click(readinessButton);

    expect((readinessButton as HTMLButtonElement).disabled).toBe(true);
    await act(async () => resolveReadiness(new Response(JSON.stringify({ code: 'ready' }))));
    expect(
      (screen.getByRole('button', { name: /Check readiness/ }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });
  it('detects a typed provider and sends the current transport with a trimmed draft key to model discovery', async () => {
    const fetch = installFetch({
      models: {
        success: true,
        transport: 'typesafe',
        models: [{ id: 'jev-latest', name: 'Jev latest', description: null, release_date: null }],
        default_model: 'jev-latest',
        message: null,
      },
    });
    const user = await openSetup();
    await user.type(screen.getByLabelText('API key'), ' ts_example ');
    expect((screen.getByRole('combobox', { name: 'Provider' }) as HTMLSelectElement).value).toBe(
      'typesafe',
    );
    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));
    await waitFor(() =>
      expect(
        JSON.parse(
          fetch.mock.calls.find(([url]) => String(url).endsWith('/models'))?.[1]?.body as string,
        ),
      ).toEqual({ transport: 'typesafe', api_key: 'ts_example' }),
    );
    expect(screen.getByRole('combobox', { name: 'Model' })).toBeTruthy();
  });
  it('does not treat a successful Jev model catalog fetch as readiness', async () => {
    const fetch = installFetch({
      models: {
        success: true,
        transport: 'typesafe',
        models: [{ id: 'jev-latest', name: 'Jev latest', description: null, release_date: null }],
        default_model: 'jev-latest',
        message: null,
      },
    });
    const user = await openSetup();
    await user.type(screen.getByLabelText('API key'), 'ts_example');
    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));

    await screen.findByRole('combobox', { name: 'Model' });
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/readiness'))).toBe(false);
    expect(screen.queryByText('Ready')).toBeNull();
  });
  it('shows catalog failures and empty model results without switching to a model select', async () => {
    const fetch = installFetch({
      models: {
        success: false,
        transport: null,
        models: [],
        default_model: 'jev-latest',
        message: 'Catalog unavailable',
      },
    });
    const user = await openSetup();
    await user.type(screen.getByLabelText('API key'), 'ts_example');
    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));
    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.getByRole('textbox', { name: 'Model' })).toBeTruthy();
    expect(fetch).toHaveBeenCalled();
  });
  it('ignores a model response after the draft key changes', async () => {
    let resolveModels!: (value: Response) => void;
    installFetch({
      models: () =>
        new Promise<Response>((resolve) => {
          resolveModels = resolve;
        }),
    });
    const user = await openSetup();
    const key = screen.getByLabelText('API key');
    await user.type(key, 'ts_example');
    await user.click(screen.getByRole('button', { name: 'Fetch Models' }));
    await user.type(key, 'x');
    await act(async () =>
      resolveModels(
        new Response(
          JSON.stringify({
            success: true,
            transport: 'typesafe',
            models: [
              { id: 'jev-latest', name: 'Jev latest', description: null, release_date: null },
            ],
            default_model: 'jev-latest',
            message: null,
          }),
        ),
      ),
    );
    expect(screen.getByRole('textbox', { name: 'Model' })).toBeTruthy();
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
    await user.click(screen.getByRole('button', { name: /^Standard\b/ }));
    expect(await screen.findByText('Unknown (gone)')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Remove unknown group gone' }));
    expect(screen.queryByText('Unknown (gone)')).toBeNull();
  });
  it('saves level group mappings and a selected default level', async () => {
    const configured = {
      ...config,
      access_levels: [
        ...config.access_levels,
        { id: 'elevated', name: 'Elevated', granted_category_ids: ['credentials'], guidance: '' },
      ],
    };
    const configuredCatalog = { ...catalog, groups: [{ id: 'admins', name: 'Admins' }] };
    const fetch = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith('/config') && options?.method === 'PUT') return response(configured);
      return response(url.endsWith('/catalog') ? configuredCatalog : configured);
    });
    vi.stubGlobal('fetch', fetch);
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Access levels' }));
    await user.click(screen.getByRole('button', { name: /^Standard\b/ }));
    await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
    await user.click(screen.getByRole('checkbox', { name: 'Admins' }));
    await user.click(screen.getByRole('button', { name: /^Elevated\b/ }));
    await user.click(screen.getByRole('button', { name: 'Make default level' }));
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));

    await waitFor(() =>
      expect(fetch.mock.calls.some(([, options]) => options?.method === 'PUT')).toBe(true),
    );
    const write = fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
    expect(JSON.parse(write?.[1]?.body as string).config).toMatchObject({
      default_access_level_id: 'elevated',
      group_access_levels: [{ group_id: 'admins', access_level_id: 'standard' }],
    });
  });
  it('keeps category drafts through tab navigation and closes the detail before the dialog', async () => {
    installFetch();
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        document.getElementById('cp-wizard-category-detail-title'),
      ),
    );
    const description = screen.getByLabelText('Description');
    await user.clear(description);
    await user.type(description, 'Updated category description');
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));

    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).value).toBe(
      'Updated category description',
    );
    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(document.querySelector('[data-category-detail]')).toBeNull();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
  });
  it('keeps tab focus when returning to a selected category editor', async () => {
    installFetch();
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));
    const coverage = screen.getByRole('tab', { name: 'Coverage' });
    await user.click(coverage);
    coverage.focus();
    await user.keyboard('{ArrowLeft}{ArrowLeft}');

    const categories = screen.getByRole('tab', { name: 'Categories' });
    await waitFor(() => expect(categories.getAttribute('aria-selected')).toBe('true'));
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    expect(document.activeElement).toBe(categories);
  });
  it('commits a category list edit before Escape closes its detail', async () => {
    installFetch();
    const user = await openSetup();

    await user.click(screen.getByRole('tab', { name: 'Categories' }));
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));
    const includes = screen.getByLabelText('Includes');
    await user.type(includes, 'Unsaved list edit');
    await user.keyboard('{Escape}');

    expect(document.querySelector('[data-category-detail]')).toBeNull();
    await user.click(screen.getByRole('button', { name: /^Credentials\b/ }));
    expect((screen.getByLabelText('Includes') as HTMLTextAreaElement).value).toBe(
      'Unsaved list edit',
    );
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
