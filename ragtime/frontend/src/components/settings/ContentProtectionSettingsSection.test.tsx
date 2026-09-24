import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionSettingsSection } from './ContentProtectionSettingsSection';

const { refreshModels, modelState } = vi.hoisted(() => ({
  refreshModels: vi.fn(),
  modelState: { models: [] as Array<{ id: string; name: string; provider: string }> },
}));
vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({
    models: modelState.models,
    loading: false,
    error: null,
    refresh: refreshModels,
  }),
}));

const config = {
  revision: 3,
  enabled: true,
  classifier_model: null,
  coverage_mode: 'all_supported_traffic' as const,
  profiles: [{ id: 'standard', name: 'Standard', level: 0, scope: 'Operational content' }],
  group_profiles: [],
  requirements: [],
  user_overrides: [],
};
const catalog = {
  users: [{ id: 'u1', name: 'Ada' }],
  groups: [],
  tools: [{ id: 'tool-1', name: 'Finance lookup' }],
  mcp_routes: [{ id: 'route-1', name: 'Finance MCP' }],
  surfaces: [
    { id: 'mcp', name: 'MCP' },
    { id: 'workspace_chat', name: 'Workspace chat' },
  ],
};
function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
function installFetch(
  overrides: Partial<
    Record<
      'config' | 'save' | 'preview' | 'readiness' | 'test' | 'decisions',
      (body?: unknown) => Response | Promise<Response>
    >
  > = {},
) {
  const fetch = vi.fn((url: string, options?: RequestInit) => {
    if (url.endsWith('/config') && options?.method === 'PUT')
      return (
        overrides.save?.(JSON.parse(options.body as string)) ||
        Promise.resolve(json({ ...config, revision: 4 }))
      );
    if (url.endsWith('/config')) return overrides.config?.() || Promise.resolve(json(config));
    if (url.endsWith('/catalog')) return Promise.resolve(json(catalog));
    if (url.endsWith('/preview'))
      return (
        overrides.preview?.(JSON.parse(options?.body as string)) ||
        Promise.resolve(
          json({ required: true, provenance: 'surface_requirement', profiles: [config.profiles] }),
        )
      );
    if (url.endsWith('/readiness'))
      return (
        overrides.readiness?.(JSON.parse(options?.body as string)) ||
        Promise.resolve(json({ code: 'ready', verdict: 'allow' }))
      );
    if (url.endsWith('/test'))
      return (
        overrides.test?.(JSON.parse(options?.body as string)) ||
        Promise.resolve(json({ code: 'ready', verdict: 'allow' }))
      );
    if (url.endsWith('/decisions'))
      return overrides.decisions?.() || Promise.resolve(json({ items: [] }));
    return Promise.resolve(json({ code: 'ready', verdict: 'allow' }));
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}
async function openSetup(user = userEvent.setup()) {
  render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
  await user.click(await screen.findByRole('button', { name: 'Configure protection' }));
  await screen.findByRole('dialog', { name: 'Configure content protection' });
  return user;
}

afterEach(() => {
  cleanup();
  refreshModels.mockClear();
  modelState.models = [];
  vi.unstubAllGlobals();
});

describe('ContentProtectionSettingsSection', () => {
  it('links to specific policies without discarding the coverage draft', async () => {
    installFetch();
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} searchQuery="Tool" />);

    await user.click(await screen.findByRole('button', { name: 'Configure protection' }));
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');

    const links = [
      ['User policies', '?view=users#user-policies'],
      ['Manage groups', '?view=users#manage-groups'],
      ['Tool access', '?view=tools#tools-connections'],
      ['MCP routes', '?view=settings#manage-mcp-routes'],
    ] as const;
    for (const [label, href] of links) {
      const link = screen.getByRole('link', { name: label }) as HTMLAnchorElement;
      expect(link.getAttribute('href')).toBe(href);
      expect(link.target).toBe('_blank');
      expect(link.rel).toBe('noopener noreferrer');
    }
    expect(screen.getByText('Tool', { selector: 'mark' })).toBeTruthy();
    expect(screen.queryByText('Existing per-user overrides remain in effect.')).toBeNull();
    expect(screen.getByText('Opens in a new tab; your draft stays here.')).toBeTruthy();

    await user.click(screen.getByRole('link', { name: 'User policies' }));
    expect((screen.getByRole('combobox', { name: 'Coverage' }) as HTMLSelectElement).value).toBe(
      'selected_scopes',
    );
    expect(screen.getByRole('dialog', { name: 'Configure content protection' })).toBeTruthy();
  });

  it('explains coverage and user-policy precedence in help', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.click(screen.getByRole('button', { name: 'Coverage help' }));

    expect(
      await screen.findByText(
        /All supported traffic applies classification broadly. Selected scopes adds requirements only where configured./,
      ),
    ).toBeTruthy();
    expect(
      screen.getByText(
        /Individual user policies can require or skip classification in either mode./,
      ),
    ).toBeTruthy();
  });

  it('preserves the open draft and its revision when the accordion reloads', async () => {
    let revision = 3;
    const save = vi.fn((body: unknown) => json((body as { config: unknown }).config));
    installFetch({ config: () => json({ ...config, revision }), save });
    const user = userEvent.setup();
    const { rerender } = render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await user.click(await screen.findByRole('button', { name: 'Configure profiles' }));
    await user.click(screen.getByRole('button', { name: 'Edit Standard permitted information' }));
    await user.clear(screen.getByLabelText('Standard scope'));
    await user.type(screen.getByLabelText('Standard scope'), 'Draft scope');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    revision = 8;
    rerender(<ContentProtectionSettingsSection open={false} onToggle={() => {}} />);
    rerender(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await user.click(screen.getByRole('tab', { name: 'Profiles' }));
    expect(screen.getByText('Draft scope')).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          expected_revision: 3,
          config: expect.objectContaining({
            revision: 3,
            profiles: [expect.objectContaining({ scope: 'Draft scope' })],
          }),
        }),
      ),
    );
  });

  it('keeps the setup draft open and clears model search on Escape', async () => {
    modelState.models = [
      { id: 'old', name: 'Old model', provider: 'provider' },
      { id: 'new', name: 'New model', provider: 'provider' },
    ];
    installFetch({ config: () => json({ ...config, classifier_model: 'provider::old' }) });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.click(screen.getByRole('tab', { name: 'Model' }));
    await user.click(screen.getByTitle('Provider Old model'));
    const search = screen.getByLabelText('Filter models');
    await user.type(search, 'New model');

    await user.keyboard('{Escape}');

    expect(screen.getByRole('dialog', { name: 'Configure content protection' })).toBeTruthy();
    expect((search as HTMLInputElement).value).toBe('');
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    expect((screen.getByRole('combobox', { name: 'Coverage' }) as HTMLSelectElement).value).toBe(
      'selected_scopes',
    );
  });

  it('shows a successful session check on the saved model summary without probing on load', async () => {
    const readiness = vi.fn(() => json({ code: 'ready', verdict: 'allow' }));
    installFetch({ config: () => json({ ...config, classifier_model: 'omlx::qwen' }), readiness });
    const user = await openSetup();
    expect(readiness).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Check selected model' }));
    await screen.findByLabelText('Model ready');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByLabelText('Saved model checked')).toBeTruthy();
    expect(readiness).toHaveBeenCalledTimes(1);
  });

  it('refreshes models when opened while keeping the closed SettingsPanel contract safe', async () => {
    const { rerender } = render(
      <ContentProtectionSettingsSection open={false} onToggle={() => {}} />,
    );
    expect(refreshModels).not.toHaveBeenCalled();
    rerender(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await waitFor(() => expect(refreshModels).toHaveBeenCalledTimes(1));
  });

  it('uses a disposable setup draft and saves the original revision once', async () => {
    const fetch = installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Configure protection' }));
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    expect((screen.getByRole('combobox', { name: 'Coverage' }) as HTMLSelectElement).value).toBe(
      'all_supported_traffic',
    );
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await waitFor(() =>
      expect(fetch.mock.calls.filter(([, options]) => options?.method === 'PUT')).toHaveLength(1),
    );
    expect(
      JSON.parse(
        fetch.mock.calls.find(([, options]) => options?.method === 'PUT')?.[1]?.body as string,
      ),
    ).toEqual({ expected_revision: 3, config });
  });

  it('retains the draft after a conflict and offers explicit reload', async () => {
    installFetch({ save: () => Promise.resolve(json({ detail: 'conflict' }, 409)) });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    expect((await screen.findByRole('alert')).textContent).toMatch(/reload saved settings/i);
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    expect((screen.getByRole('combobox', { name: 'Coverage' }) as HTMLSelectElement).value).toBe(
      'selected_scopes',
    );
    expect(screen.getByRole('button', { name: 'Reload and discard draft' })).toBeTruthy();
  });

  it('keeps the conflicted draft when reloading saved settings fails', async () => {
    let configRequests = 0;
    installFetch({
      config: () => {
        configRequests += 1;
        return configRequests === 1
          ? Promise.resolve(json(config))
          : Promise.resolve(json({ detail: 'reload failed' }, 500));
      },
      save: () => Promise.resolve(json({ detail: 'conflict' }, 409)),
    });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await user.click(await screen.findByRole('button', { name: 'Reload and discard draft' }));
    expect(
      await screen.findByRole('dialog', { name: 'Configure content protection' }),
    ).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    expect((screen.getByRole('combobox', { name: 'Coverage' }) as HTMLSelectElement).value).toBe(
      'selected_scopes',
    );
  });

  it('preserves focus and typed profile text across draft rerenders', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Profiles' }));
    await user.click(screen.getByRole('button', { name: 'Edit Standard name' }));
    const name = screen.getByLabelText('Standard name');
    await user.clear(name);
    await user.type(name, 'Restricted');
    expect((name as HTMLInputElement).value).toBe('Restricted');
    expect(document.activeElement).toBe(name);
  });

  it('preserves dormant app-area requirements when all-traffic coverage is saved', async () => {
    const withDormantRequirement = {
      ...config,
      coverage_mode: 'selected_scopes' as const,
      requirements: [
        { scope_kind: 'surface' as const, scope_key: 'mcp', mode: 'require' as const },
      ],
    };
    let savedBody: unknown;
    installFetch({
      config: () => Promise.resolve(json(withDormantRequirement)),
      save: (body) => {
        savedBody = body;
        return Promise.resolve(json({ ...withDormantRequirement, revision: 4 }));
      },
    });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Coverage' }),
      'all_supported_traffic',
    );
    expect(screen.queryByLabelText('Coverage for MCP')).toBeNull();
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await waitFor(() => expect(savedBody).toBeTruthy());
    expect(savedBody).toMatchObject({
      expected_revision: 3,
      config: {
        coverage_mode: 'all_supported_traffic',
        requirements: withDormantRequirement.requirements,
      },
    });
  });

  it('hides app-area requirements in a disabled selected-scopes draft without dropping them', async () => {
    const disabledDraft = {
      ...config,
      enabled: false,
      coverage_mode: 'selected_scopes' as const,
      requirements: [
        { scope_kind: 'surface' as const, scope_key: 'mcp', mode: 'require' as const },
      ],
    };
    let savedBody: unknown;
    installFetch({
      config: () => Promise.resolve(json(disabledDraft)),
      save: (body) => {
        savedBody = body;
        return Promise.resolve(json({ ...disabledDraft, revision: 4 }));
      },
    });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));

    expect(screen.queryByLabelText('Coverage for MCP')).toBeNull();
    expect(
      screen.getByText('Enable protection in Review & save to configure app areas.'),
    ).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));
    await waitFor(() => expect(savedBody).toBeTruthy());
    expect(savedBody).toMatchObject({ config: { requirements: disabledDraft.requirements } });
  });

  it('keeps dormant requirements and only sends MCP route or opted-in tool fields', async () => {
    const previewBodies: unknown[] = [];
    installFetch({
      preview: (body) => {
        previewBodies.push(body);
        return Promise.resolve(
          json({ required: true, provenance: 'surface_requirement', profiles: [config.profiles] }),
        );
      },
    });
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Coverage' }));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Coverage' }), 'selected_scopes');
    await user.selectOptions(screen.getByLabelText('Coverage for MCP'), 'require');
    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Coverage' }),
      'all_supported_traffic',
    );
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Test & inspect' }));
    await user.selectOptions(screen.getByLabelText('Where does it run?'), 'workspace_chat');
    expect(screen.queryByLabelText('MCP route')).toBeNull();
    await user.click(screen.getByRole('button', { name: /check saved policy/i }));
    await waitFor(() =>
      expect(previewBodies[0]).toEqual({ surface: 'workspace_chat', public: false }),
    );
    await user.selectOptions(screen.getByLabelText('Where does it run?'), 'mcp');
    await user.selectOptions(screen.getByLabelText('MCP route'), 'route-1');
    await user.click(screen.getByLabelText('Include a tool call'));
    await user.selectOptions(screen.getByLabelText('Tool'), 'tool-1');
    await user.click(screen.getByRole('button', { name: /check saved policy/i }));
    await waitFor(() =>
      expect(previewBodies[1]).toEqual({
        surface: 'mcp',
        mcp_route: 'route-1',
        tool_id: 'tool-1',
        public: false,
      }),
    );
  });

  it('invalidates in-flight request and content results while re-enabling their actions', async () => {
    const previewResponse = deferred<Response>();
    const testResponse = deferred<Response>();
    installFetch({
      config: () => Promise.resolve(json({ ...config, classifier_model: 'provider::model' })),
      preview: () => previewResponse.promise,
      test: () => testResponse.promise,
    });
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await user.click(await screen.findByRole('button', { name: 'Test & inspect' }));
    const check = screen.getByRole('button', { name: /check saved policy/i });
    await user.click(check);
    expect((check as HTMLButtonElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('Where does it run?'), 'workspace_chat');
    expect(
      (screen.getByRole('button', { name: /check saved policy/i }) as HTMLButtonElement).disabled,
    ).toBe(false);
    previewResponse.resolve(
      json({ required: true, provenance: 'stale', profiles: [config.profiles] }),
    );
    await waitFor(() => expect(screen.queryByText('Provenance: stale')).toBeNull());

    await user.click(screen.getByRole('tab', { name: 'Test content' }));
    const sampleInput = screen.getByLabelText('Sample content');
    await user.type(sampleInput, 'first sample');
    const runTest = screen.getByRole('button', { name: 'Test content' });
    await user.click(runTest);
    expect((screen.getByRole('button', { name: 'Testing…' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    await user.type(sampleInput, ' changed');
    expect(
      (screen.getByRole('button', { name: 'Test content' }) as HTMLButtonElement).disabled,
    ).toBe(false);
    testResponse.resolve(json({ code: 'stale-result', verdict: 'deny' }));
    await waitFor(() => expect(screen.queryByText(/stale-result/)).toBeNull());
  });

  it('ignores a late model readiness result after the selected model changes', async () => {
    modelState.models = [
      { id: 'old', name: 'Old model', provider: 'provider' },
      { id: 'new', name: 'New model', provider: 'provider' },
    ];
    const readinessResponse = deferred<Response>();
    installFetch({
      config: () => Promise.resolve(json({ ...config, classifier_model: 'provider::old' })),
      readiness: () => readinessResponse.promise,
    });
    const user = await openSetup();
    await user.click(screen.getByRole('button', { name: 'Check selected model' }));
    expect(screen.getByLabelText('Checking model')).toBeTruthy();
    await user.click(screen.getByTitle('Provider Old model'));
    await user.type(screen.getByLabelText('Filter models'), 'New model');
    await user.click(screen.getByTitle('new'));
    expect(screen.getByLabelText('Model unchecked')).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Check selected model' }) as HTMLButtonElement).disabled,
    ).toBe(false);
    readinessResponse.resolve(json({ code: 'stale-ready', verdict: 'allow' }));
    await waitFor(() => expect(screen.queryByText(/stale-ready/)).toBeNull());
  });

  it('requests failed decisions at most once per dialog and hides a late failure after close', async () => {
    const decisionsResponse = deferred<Response>();
    let decisionRequests = 0;
    installFetch({
      decisions: () => {
        decisionRequests += 1;
        return decisionsResponse.promise;
      },
    });
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await user.click(await screen.findByRole('button', { name: 'Test & inspect' }));
    await user.click(screen.getByRole('tab', { name: 'Recent decisions' }));
    await user.click(screen.getByRole('tab', { name: 'Check a request' }));
    await user.click(screen.getByRole('tab', { name: 'Recent decisions' }));
    expect(decisionRequests).toBe(1);
    await user.click(screen.getByRole('button', { name: 'Close' }));
    await act(async () => {
      decisionsResponse.reject(new Error('decisions unavailable'));
      await Promise.resolve();
    });
    expect(screen.queryByRole('alert')).toBeNull();
    expect(decisionRequests).toBe(1);
  });

  it('adds and deletes an unassigned profile from the setup draft', async () => {
    installFetch();
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Profiles' }));
    await user.click(screen.getByRole('button', { name: 'Add profile' }));

    const newProfile = screen.getByText('New profile').closest('[data-profile-id]');
    expect(newProfile).toBeTruthy();
    await user.click(newProfile!.querySelector('button.btn-danger')!);

    expect(screen.queryByText('New profile')).toBeNull();
  });

  it('guards assigned profile deletion and closes the dialog on Escape', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/config'))
          return Promise.resolve(
            json({ ...config, group_profiles: [{ group_id: 'group-1', profile_id: 'standard' }] }),
          );
        if (url.endsWith('/catalog')) return Promise.resolve(json(catalog));
        return Promise.resolve(json({ items: [] }));
      }),
    );
    const user = await openSetup();
    await user.click(screen.getByRole('tab', { name: 'Profiles' }));
    await user.click(screen.getByRole('button', { name: 'Edit Standard permitted information' }));
    await user.type(screen.getByLabelText('Standard scope'), ' cancelled');
    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog', { name: 'Configure content protection' })).toBeTruthy();
    expect(screen.getByText('Operational content')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Delete (1 groups)' }));
    expect(screen.getByRole('alert').textContent).toMatch(/reassign or clear/i);
    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });
});
