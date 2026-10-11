import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ContentProtectionApiError,
  contentProtectionApi,
  type ContentProtectionConfig,
} from '@/api/contentProtection';
import { ContentProtectionSettingsSection } from './ContentProtectionSettingsSection';

vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({ models: [], loading: false, error: null, refresh: vi.fn() }),
}));

const catalog = {
  users: [],
  groups: [],
  tools: [],
  mcp_routes: [],
  surfaces: [{ id: 'chat', name: 'Chat' }],
  classifier_status: { typesafe_key_configured: true, openrouter_key_configured: true },
};

function config(overrides: Partial<ContentProtectionConfig> = {}): ContentProtectionConfig {
  return {
    schema_version: 2,
    revision: 3,
    enabled: false,
    share_with_assistant: true,
    classifier: {
      backend: 'jev',
      jev: { transport: 'auto', model: 'jev-latest' },
      llm_model: null,
    },
    strictness: 'balanced',
    categories: [
      {
        id: 'credentials',
        name: 'Credentials',
        description: 'Sensitive credentials.',
        includes: [],
        excludes: [],
        examples: [],
        denial_message: 'Restricted.',
        threshold_override: null,
        system: false,
      },
    ],
    access_levels: [
      { id: 'standard', name: 'Standard', granted_category_ids: [], guidance: 'Keep it safe.' },
    ],
    group_access_levels: [{ group_id: 'admins', access_level_id: 'standard' }],
    default_access_level_id: 'standard',
    coverage_mode: 'selected_scopes',
    requirements: [{ scope_kind: 'surface', scope_key: 'chat', mode: 'require' }],
    user_overrides: [{ user_id: 'u1', mode: 'never_classify' }],
    ...overrides,
  };
}

function installApi(initial = config()) {
  let server = structuredClone(initial);
  vi.spyOn(contentProtectionApi, 'getConfig').mockImplementation(async () =>
    structuredClone(server),
  );
  vi.spyOn(contentProtectionApi, 'getCatalog').mockResolvedValue(catalog);
  vi.spyOn(contentProtectionApi, 'saveConfig').mockImplementation(async (_revision, candidate) => {
    server = { ...structuredClone(candidate), revision: server.revision + 1 };
    return structuredClone(server);
  });
  return {
    get server() {
      return server;
    },
    set server(next: ContentProtectionConfig) {
      server = next;
    },
  };
}

async function renderSection(initial = config()) {
  installApi(initial);
  const user = userEvent.setup();
  render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
  await screen.findByLabelText('Enable content protection');
  return user;
}

function toggle(): HTMLInputElement {
  return document.getElementById('content-protection-enabled') as HTMLInputElement;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('ContentProtectionSettingsSection global protection toggle', () => {
  it('renders the persisted enabled value instead of an optimistic value', async () => {
    await renderSection(config({ enabled: true }));

    expect(toggle().checked).toBe(true);
  });

  it('enables the latest config while preserving saved guidance and policy fields', async () => {
    const user = await renderSection();

    await user.click(toggle());

    await waitFor(() => expect(toggle().checked).toBe(true));
    const saved = vi.mocked(contentProtectionApi.saveConfig).mock.calls[0][1];
    expect(saved).toMatchObject({
      enabled: true,
      share_with_assistant: true,
      access_levels: [{ guidance: 'Keep it safe.' }],
      group_access_levels: [{ group_id: 'admins', access_level_id: 'standard' }],
      requirements: [{ scope_kind: 'surface', scope_key: 'chat', mode: 'require' }],
      user_overrides: [{ user_id: 'u1', mode: 'never_classify' }],
    });
  });

  it('keeps the persisted off state and disables actions while enabling is pending', async () => {
    const api = installApi();
    let finishSave!: (value: ContentProtectionConfig) => void;
    vi.mocked(contentProtectionApi.saveConfig).mockImplementation(
      () =>
        new Promise<ContentProtectionConfig>((resolve) => {
          finishSave = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await screen.findByLabelText('Enable content protection');

    await user.click(toggle());

    expect(toggle().checked).toBe(false);
    expect(toggle().disabled).toBe(true);
    expect(screen.getByRole('status').textContent).toContain('Checking classifier…');
    expect(
      (screen.getByRole('button', { name: 'Configure classifier' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await act(async () => {
      api.server = config({ enabled: true, revision: api.server.revision + 1 });
      finishSave(api.server);
    });
    await waitFor(() => expect(toggle().checked).toBe(true));
  });

  it('requires confirmation to turn protection off, cancels with Escape, and restores focus', async () => {
    const user = await renderSection(config({ enabled: true }));
    toggle().focus();

    await user.click(toggle());

    const cancel = screen.getByRole('button', { name: 'Cancel' });
    expect(document.activeElement).toBe(cancel);
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('group', { name: 'Turn off protection?' })).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(toggle()));
    expect(vi.mocked(contentProtectionApi.saveConfig)).not.toHaveBeenCalled();
  });

  it('turns protection off only after confirmation and restores switch focus', async () => {
    const user = await renderSection(config({ enabled: true }));
    toggle().focus();

    await user.click(toggle());
    await user.click(screen.getByRole('button', { name: 'Turn off' }));

    await waitFor(() => expect(toggle().checked).toBe(false));
    await waitFor(() => expect(document.activeElement).toBe(toggle()));
    expect(vi.mocked(contentProtectionApi.saveConfig).mock.calls[0][1]).toMatchObject({
      enabled: false,
      access_levels: [{ guidance: 'Keep it safe.' }],
    });
  });

  it('rejects an invalid latest enable candidate without issuing a PUT', async () => {
    const invalid = config({
      classifier: {
        backend: 'llm',
        jev: { transport: 'auto', model: 'jev-latest' },
        llm_model: null,
      },
    });
    const user = await renderSection(invalid);

    await user.click(toggle());

    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(vi.mocked(contentProtectionApi.saveConfig)).not.toHaveBeenCalled();
    expect(toggle().checked).toBe(false);
  });

  it('offers classifier configuration after a 503 enable failure', async () => {
    const user = await renderSection();
    vi.mocked(contentProtectionApi.saveConfig).mockRejectedValueOnce(
      new ContentProtectionApiError('Classifier unavailable', 503),
    );

    await user.click(toggle());

    expect(await screen.findByRole('alert')).toBeTruthy();
    await user.click(screen.getByRole('alert').querySelector('button')!);
    expect(screen.getByRole('dialog', { name: 'Configure content protection' })).toBeTruthy();
  });

  it('clears the recovery alert after a successful recovery setup save', async () => {
    const user = await renderSection();
    vi.mocked(contentProtectionApi.saveConfig).mockRejectedValueOnce(
      new ContentProtectionApiError('Classifier unavailable', 503),
    );

    await user.click(toggle());
    await user.click(screen.getByRole('alert').querySelector('button')!);
    await user.click(screen.getByRole('tab', { name: 'Review & save' }));
    await user.click(screen.getByRole('button', { name: 'Save protection' }));

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(document.querySelector('[data-content-protection-toggle-error]')).toBeNull();
  });

  it('does not offer misleading recovery after both CAS attempts conflict', async () => {
    const user = await renderSection();
    vi.mocked(contentProtectionApi.saveConfig).mockRejectedValue(
      new ContentProtectionApiError('Conflict', 409),
    );

    await user.click(toggle());

    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(vi.mocked(contentProtectionApi.saveConfig)).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('alert').querySelector('button')).toBeNull();
  });

  it('rejects an invalid latest CAS retry candidate without a second PUT', async () => {
    const valid = config();
    const invalid = config({
      classifier: {
        backend: 'llm',
        jev: { transport: 'auto', model: 'jev-latest' },
        llm_model: null,
      },
    });
    let reads = 0;
    vi.spyOn(contentProtectionApi, 'getConfig').mockImplementation(async () => {
      reads += 1;
      return structuredClone(reads >= 3 ? invalid : valid);
    });
    vi.spyOn(contentProtectionApi, 'getCatalog').mockResolvedValue(catalog);
    vi.spyOn(contentProtectionApi, 'saveConfig').mockRejectedValueOnce(
      new ContentProtectionApiError('Conflict', 409),
    );
    const user = userEvent.setup();
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await screen.findByLabelText('Enable content protection');

    await user.click(toggle());

    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(vi.mocked(contentProtectionApi.saveConfig)).toHaveBeenCalledOnce();
  });

  it('locks the global toggle while the setup modal is open', async () => {
    const user = await renderSection();

    await user.click(screen.getByRole('button', { name: 'Configure classifier' }));

    expect(toggle().disabled).toBe(true);
  });

  it('keeps disable confirmation open when Escape is pressed outside it', async () => {
    const user = await renderSection(config({ enabled: true }));

    await user.click(toggle());
    screen.getByRole('button', { name: /Content protection/ }).focus();
    await user.keyboard('{Escape}');

    expect(screen.getByRole('group', { name: 'Turn off protection?' })).toBeTruthy();
  });

  it('does not replace a successful save with a stale refresh response', async () => {
    const api = installApi();
    let resolveRefresh!: (value: ContentProtectionConfig) => void;
    let configRequests = 0;
    vi.mocked(contentProtectionApi.getConfig).mockImplementation(() => {
      configRequests += 1;
      if (configRequests === 2)
        return new Promise<ContentProtectionConfig>((resolve) => {
          resolveRefresh = resolve;
        });
      return Promise.resolve(structuredClone(api.server));
    });
    const user = userEvent.setup();
    const view = render(
      <ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={0} />,
    );
    await screen.findByLabelText('Enable content protection');

    view.rerender(<ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={1} />);
    await user.click(toggle());
    await waitFor(() => expect(toggle().checked).toBe(true));
    await act(async () => resolveRefresh(config({ enabled: false, revision: 2 })));

    expect(toggle().checked).toBe(true);
  });

  it('shows an unavailable status rather than Disabled when loading fails', async () => {
    vi.spyOn(contentProtectionApi, 'getConfig').mockRejectedValue(new Error('offline'));
    vi.spyOn(contentProtectionApi, 'getCatalog').mockResolvedValue(catalog);
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);

    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.queryByLabelText('Enable content protection')).toBeNull();
    expect(screen.getByText('Unavailable')).toBeTruthy();
  });

  it('does not imply a failure before the collapsed section has loaded', () => {
    installApi();
    render(<ContentProtectionSettingsSection open={false} onToggle={() => {}} />);

    expect(screen.queryByText('Unavailable')).toBeNull();
    expect(contentProtectionApi.getConfig).not.toHaveBeenCalled();
  });

  it('shows Loading in the header until the first configuration load completes', async () => {
    let resolveConfig!: (value: ContentProtectionConfig) => void;
    vi.spyOn(contentProtectionApi, 'getConfig').mockImplementation(
      () =>
        new Promise<ContentProtectionConfig>((resolve) => {
          resolveConfig = resolve;
        }),
    );
    vi.spyOn(contentProtectionApi, 'getCatalog').mockResolvedValue(catalog);
    render(<ContentProtectionSettingsSection open onToggle={() => {}} />);

    expect(screen.getByText('Loading…')).toBeTruthy();
    await act(async () => resolveConfig(config()));
  });

  it('shows a saved-refresh error and unlocks after retry succeeds', async () => {
    const api = installApi();
    const user = userEvent.setup();
    const view = render(
      <ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={0} />,
    );
    await screen.findByLabelText('Enable content protection');
    vi.mocked(contentProtectionApi.getConfig).mockRejectedValueOnce(new Error('offline'));

    view.rerender(<ContentProtectionSettingsSection open onToggle={() => {}} refreshKey={1} />);

    const error = await waitFor(() => {
      const element = document.querySelector('[data-content-protection-saved-refresh-error]');
      expect(element).not.toBeNull();
      return element!;
    });
    expect(toggle().disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(error.isConnected).toBe(false));
    expect(api.server.enabled).toBe(false);
    expect(toggle().disabled).toBe(false);
  });
});
